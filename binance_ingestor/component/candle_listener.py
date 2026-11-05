import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd

from binance_ingestor.api import (get_coin_futures_multi_candlesticks_socket, get_spot_multi_candlesticks_socket,
                                   get_usdt_futures_multi_candlesticks_socket)
from binance_ingestor.bus import TRADE_TYPE_MAP
from binance_ingestor.component.candle_fetcher import BinanceFetcher
from binance_ingestor.utils import convert_interval_to_timedelta, get_logger, create_aiohttp_session
from binance_ingestor.config import KLINE_INTERVAL, FETCH_CONCURRENCY, BATCH_REMEDY_WAIT_SEC
from binance_ingestor.utils.log_kit import logger, divider
from binance_ingestor.utils.time import now_time, async_sleep_until_run_time, next_run_time


def convert_to_dataframe(x, interval_delta):
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_num',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]
    candle_data = [
        pd.to_datetime(int(x['t']), unit='ms', utc=True),
        float(x['o']),
        float(x['h']),
        float(x['l']),
        float(x['c']),
        float(x['v']),
        float(x['q']),
        int(x['n']),
        float(x['V']),
        float(x['Q'])
    ]
    return pd.DataFrame(data=[candle_data], columns=columns, index=[candle_data[0] + interval_delta])


class CandleListener:

    TRADE_TYPE_MAP = {
        'usdt_perp': get_usdt_futures_multi_candlesticks_socket,
        'coin_perp': get_coin_futures_multi_candlesticks_socket,
        'usdt_spot': get_spot_multi_candlesticks_socket
    }

    def __init__(self, type_, symbols, time_interval, que):
        self.trade_type = type_
        self.symbols = set(symbols)
        self.time_interval = time_interval
        self.interval_delta = convert_interval_to_timedelta(time_interval)
        self.que: asyncio.Queue = que
        self.req_reconnect = False
        self._socket = None

    async def start_listen(self):
        if not self.symbols:
            return

        socket_func = self.TRADE_TYPE_MAP[self.trade_type]
        while True:
            socket = socket_func(self.symbols, self.time_interval)
            self._socket = socket
            try:
                async with socket as socket_conn:
                    self._socket = socket_conn
                    while True:
                        if self.req_reconnect:
                            self.req_reconnect = False
                            break
                        try:
                            res = await socket_conn.recv()
                        except asyncio.TimeoutError:
                            get_logger().error('Recv candle ws timeout, reconnecting')
                            break
                        if isinstance(res, dict) and res.get('e') == 'forced_reconnect':
                            self.req_reconnect = False
                            break
                        self.handle_candle_data(res)
            finally:
                self._socket = None

    def handle_candle_data(self, res):
        if 'data' not in res:
            return
        data = res['data']
        if data.get('e', None) != 'kline' or 'k' not in data:
            return
        candle = data['k']
        is_closed = candle.get('x', False)
        if not is_closed:
            return

        df_candle = convert_to_dataframe(candle, self.interval_delta)

        self.que.put_nowait({
            'type': 'candle_data',
            'data': df_candle,
            'closed': is_closed,
            'run_time': df_candle.index[0],
            'symbol': data['s'],
            'time_interval': self.time_interval,
            'trade_type': self.trade_type,
            'recv_time': now_time()
        })

    def add_symbols(self, *symbols):
        for symbol in symbols:
            self.symbols.add(symbol)

    def remove_symbols(self, *symbols):
        for symbol in symbols:
            if symbol in self.symbols:
                self.symbols.remove(symbol)

    def reconnect(self):
        self.req_reconnect = True

    async def force_reconnect(self):
        """Cut the current socket so ``start_listen`` dials a new one immediately."""
        self.req_reconnect = True
        socket = self._socket
        if socket is not None and hasattr(socket, 'force_disconnect'):
            await socket.force_disconnect()


class MarketListener:

    # run_times already handled (written from the websocket batch, or remedied over REST);
    # candles arriving for them afterwards are dropped. Keeps only the most recent ones.
    SETTLED_KEEP = 12

    def __init__(self, market='usdt_perp', data_callback=None, exginfo_callback=None,
                 remedy_callback=None, full_remedy_callback=None, gap_callback=None,
                 remedy_wait_sec=BATCH_REMEDY_WAIT_SEC):
        self.main_queue = asyncio.Queue()
        self.market = market
        self.listeners = {}
        self.symbols = None
        self._data_batches = {}
        self._batch_status = {}
        self._settled: Dict[datetime, str] = {}
        self.data_callback = data_callback
        self.exginfo_callback = exginfo_callback
        self.remedy_callback = remedy_callback
        self.full_remedy_callback = full_remedy_callback
        self.gap_callback = gap_callback
        self.remedy_wait_sec = remedy_wait_sec

    async def build_and_run(self):
        if self.symbols is None:
            self.symbols = set(await self.get_exginfo())

        self.listeners = self.create_listeners(market=self.market, symbols=self.symbols, que=self.main_queue)
        listen_tasks = [
            asyncio.create_task(v.start_listen(), name=f"{self.market}-listener-{k}")
            for k, v in self.listeners.items()
        ]
        dispatcher_task = asyncio.create_task(self.dispatcher(), name=f"{self.market}-dispatcher")
        scheduled_task = asyncio.create_task(self.scheduled_run(), name=f"{self.market}-scheduler")
        tasks = listen_tasks + [dispatcher_task, scheduled_task]
        if self.remedy_callback:
            tasks.append(asyncio.create_task(self.batch_watchdog(), name=f"{self.market}-batch-watchdog"))

        logger.ok(f'{self.market} Market Listener initialized...')

        try:
            await asyncio.gather(*tasks)
        finally:
            await self._cancel_tasks(tasks)

    @staticmethod
    async def _cancel_tasks(tasks):
        pending = [task for task in tasks if not task.done()]
        if not pending:
            return

        for task in pending:
            task.cancel()

        results = await asyncio.gather(*pending, return_exceptions=True)
        cancelled = sum(isinstance(result, asyncio.CancelledError) for result in results)
        logger.warning(f"MarketListener cleanup: cancelled {cancelled}/{len(pending)} pending tasks")

    def _collect_kline_data(self, run_time: datetime, symbol: str, kline_data: Dict[str, Any]) -> None:
        try:
            if run_time not in self._data_batches:
                self._data_batches[run_time] = {}
                self._batch_status[run_time] = {
                    'collected_symbols': set(),
                }

            self._data_batches[run_time][symbol] = kline_data
            self._batch_status[run_time]['collected_symbols'].add(symbol)

            collected = len(self._batch_status[run_time]['collected_symbols'])
            if collected % 100 == 0:
                logger.debug(f"[Websocket] {self.market}, process: {collected} / {len(self.symbols)}")

        except Exception as e:
            logger.error(f"收集kline数据失败: {e}, run_time: {run_time}, symbol: {symbol}")

    def _check_batch_completeness(self, run_time: datetime) -> bool:
        try:
            if run_time not in self._batch_status:
                return False

            status = self._batch_status[run_time]
            collected_symbols = status['collected_symbols']

            is_complete = (len(collected_symbols) == len(self.symbols) and collected_symbols == self.symbols)

            if is_complete:
                logger.ok(f"[Websocket] {self.market} - process done. total: {len(collected_symbols)}")

            return is_complete

        except Exception as e:
            logger.error(f"检查批次完整性失败: {e}, run_time: {run_time}")
            return False

    async def _batch_write_to_db(self, run_time: datetime) -> None:
        if self.data_callback:
            await self.data_callback(run_time, self._data_batches[run_time], self.market)
        else:
            logger.warning('_batch_write_to_db has not callback function.')


    def _cleanup_batch_data(self, run_time: datetime) -> None:
        try:
            if run_time in self._data_batches:
                del self._data_batches[run_time]
            if run_time in self._batch_status:
                del self._batch_status[run_time]
        except Exception as e:
            logger.error(f"清理数据批次失败: {e}, run_time: {run_time}")

    def _mark_settled(self, run_time: datetime, how: str) -> None:
        self._settled[run_time] = how
        while len(self._settled) > self.SETTLED_KEEP:
            del self._settled[next(iter(self._settled))]

    async def dispatcher(self):
        while True:
            req = await self.main_queue.get()
            if req['type'] == 'candle_data':
                await self._handle_candle(req)

    async def _handle_candle(self, req) -> None:
        run_time = req['run_time']
        if run_time in self._settled:
            # bar already written (or remedied over REST); a late candle has nothing to add
            return
        self._collect_kline_data(run_time, req['symbol'], req)
        if self._check_batch_completeness(run_time):
            # settle before the write so the watchdog cannot remedy a bar that is being written
            self._mark_settled(run_time, 'websocket')
            await self._batch_write_to_db(run_time)
            self._cleanup_batch_data(run_time)

    async def batch_watchdog(self):
        """At each close, compare ``run_time`` with the watermark, then either run the
        full remedy immediately or wait ``remedy_wait_sec`` for this bar's batch."""
        wait = timedelta(seconds=self.remedy_wait_sec)
        while True:
            run_time = pd.Timestamp(next_run_time(KLINE_INTERVAL))
            await async_sleep_until_run_time(run_time)
            if await self.on_bar_close(run_time) == 'wait':
                await async_sleep_until_run_time(run_time + wait)
                await self.remedy_incomplete(run_time)

    async def on_bar_close(self, run_time) -> str:
        """Decide this close before waiting for the batch.

        Returns ``full`` when the watermark is more than one bar behind and the full
        remedy was started, ``wait`` when this bar is adjacent and the caller should
        wait for the websocket batch, or ``settled`` when there is nothing to do.
        """
        run_time = pd.Timestamp(run_time)
        if run_time in self._settled or not self.remedy_callback:
            return 'settled'
        if self.gap_callback and self.gap_callback(run_time, self.market):
            self._mark_settled(run_time, 'full')
            self._cleanup_batch_data(run_time)
            logger.warning(f"[Websocket] {self.market} bar {run_time:%Y-%m-%d %H:%M} is more than one "
                           f"interval behind the watermark; running the full REST remedy")
            try:
                if self.full_remedy_callback:
                    await self.full_remedy_callback(run_time, self.market)
            except Exception as e:
                logger.error(f"[Websocket] {self.market} bar {run_time:%Y-%m-%d %H:%M} full remedy failed: {e}")
            return 'full'
        return 'wait'

    async def _reconnect_missing(self, missing) -> list:
        """Cut and redial only the listener groups that own ``missing`` symbols."""
        group_ids = []
        for symbol in sorted(missing):
            group_id = hash(symbol) % FETCH_CONCURRENCY
            if group_id in group_ids or group_id not in self.listeners:
                continue
            group_ids.append(group_id)
            listener = self.listeners[group_id]
            force_reconnect = getattr(listener, 'force_reconnect', None)
            if force_reconnect is not None:
                await force_reconnect()
            else:
                listener.reconnect()
        return group_ids

    async def remedy_incomplete(self, run_time) -> bool:
        """Reconnect the groups still missing candles and hand the partial batch to
        ``remedy_callback``. Returns True when a remedy was attempted."""
        run_time = pd.Timestamp(run_time)
        if run_time in self._settled or not self.remedy_callback:
            return False
        status = self._batch_status.get(run_time)
        collected = set(status['collected_symbols']) if status else set()
        missing = set(self.symbols) - collected
        partial = dict(self._data_batches.get(run_time, {}))
        self._mark_settled(run_time, 'remedy')
        self._cleanup_batch_data(run_time)
        logger.warning(f"[Websocket] {self.market} batch {run_time:%Y-%m-%d %H:%M} incomplete "
                       f"{self.remedy_wait_sec:g}s after close: {len(collected)}/{len(self.symbols)} collected, "
                       f"reconnecting groups and completing {len(missing)} symbols over REST: {sorted(missing)[:5]}...")
        try:
            await self._reconnect_missing(missing)
            await self.remedy_callback(run_time, partial, missing, self.market)
        except Exception as e:
            logger.error(f"[Websocket] {self.market} batch {run_time:%Y-%m-%d %H:%M} remedy failed: {e}")
        return True

    async def scheduled_run(self):
        while True:
            next_time = next_run_time(KLINE_INTERVAL)
            divider(f"[{self.market} Listener] next fetch runtime: {next_time:%Y-%m-%d %H:%M:%S}", display_time=False)
            await async_sleep_until_run_time(next_time)
            symbols_trading = await self.get_exginfo()
            await asyncio.sleep(10)
            await self.update_exginfo(symbols_trading)
            await asyncio.sleep(1)

    async def get_exginfo(self):
        async with create_aiohttp_session(10) as session:
            fetcher = BinanceFetcher(self.market, session)
            if self.exginfo_callback:
                symbols_trading = await self.exginfo_callback(fetcher, self.market)
            else:
                exginfo = await fetcher.get_exchange_info()
                symbols_trading = TRADE_TYPE_MAP[self.market][0](exginfo)
        logger.info(f'Fetched {len(symbols_trading)} trading symbols from exchange info')
        return symbols_trading

    async def update_exginfo(self, symbols_trading):
        delist = self.symbols - set(symbols_trading)
        onboard = set(symbols_trading) - self.symbols

        changed_groups = set()

        if delist:
            logger.warning(f'Symbols delist: {delist}')
            for symbol in delist:
                group_id = hash(symbol) % FETCH_CONCURRENCY
                if group_id in self.listeners:
                    listener: CandleListener = self.listeners[group_id]
                    listener.remove_symbols(symbol)
                    changed_groups.add(group_id)

        if onboard:
            logger.warning(f'Symbols onboard: {onboard}')
            for symbol in onboard:
                group_id = hash(symbol) % FETCH_CONCURRENCY
                if group_id not in self.listeners:
                    self.listeners[group_id] = CandleListener(self.market, [symbol], KLINE_INTERVAL, self.main_queue)
                    logger.info(f'Created new listener group {group_id} for symbol {symbol}')
                else:
                    listener: CandleListener = self.listeners[group_id]
                    listener.add_symbols(symbol)
                changed_groups.add(group_id)

        for group_id in changed_groups:
            if group_id in self.listeners:
                listener: CandleListener = self.listeners[group_id]
                listener.reconnect()

        self.symbols = set(symbols_trading)

    @staticmethod
    def create_listeners(market, symbols, que) -> dict[int, CandleListener]:
        groups = [[] for _ in range(FETCH_CONCURRENCY)]
        for sym in symbols:
            group_id = hash(sym) % FETCH_CONCURRENCY
            groups[group_id].append(sym)
        listeners = {}
        for idx, grp in enumerate(groups):
            num = len(grp)
            if num > 0:
                logger.debug(f'Create WS listen group {idx}, {num} symbols')
                listeners[idx] = CandleListener(market, grp, KLINE_INTERVAL, que)
        return listeners
