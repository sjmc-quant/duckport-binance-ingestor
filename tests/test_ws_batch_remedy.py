"""WebSocket batch remedy. A close more than one bar behind duck_time goes straight to
the full REST remedy. An adjacent close waits ``BATCH_REMEDY_WAIT_SEC``; if the batch
is still incomplete, missing symbols are filled over REST and written only when the
closed bar is complete.

Run with ``PYTHONPATH=. pytest tests``.
"""
import asyncio
import contextlib
from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz

from binance_ingestor import data_jobs
from binance_ingestor.api.ws_basics import ReconnectingWebsocket, WSListenerState
from binance_ingestor.component.candle_fetcher import OptimizedKlineFetcher
from binance_ingestor.component.candle_listener import MarketListener
from binance_ingestor.config import FETCH_CONCURRENCY
from binance_ingestor.data_jobs import WebsocketsDataJobs

RT = pd.Timestamp("2026-11-05 01:40:00", tz="UTC")
INTERVAL = timedelta(minutes=data_jobs.KLINE_INTERVAL_MINUTES)
MARKET = "usdt_perp"


def ws_req(symbol, run_time=RT, close=1.0):
    """A closed-candle queue item as CandleListener.handle_candle_data builds it."""
    df = pd.DataFrame({
        "open_time": [run_time - INTERVAL], "open": [1.0], "high": [1.0], "low": [1.0], "close": [close],
        "volume": [10.0], "quote_volume": [10.0], "trade_num": [3],
        "taker_buy_base_asset_volume": [5.0], "taker_buy_quote_asset_volume": [5.0],
    }, index=[run_time])
    return {"type": "candle_data", "data": df, "closed": True, "run_time": run_time, "symbol": symbol,
            "time_interval": "5m", "trade_type": MARKET, "recv_time": run_time}


def rest_df(symbol, open_times, close=2.0):
    """What BinanceFetcher.get_candle returns: one row per candle, newest last."""
    n = len(open_times)
    return pd.DataFrame({
        "open_time": pd.to_datetime(open_times, utc=True), "symbol": [symbol] * n, "open": [1.0] * n,
        "high": [1.0] * n, "low": [1.0] * n, "close": [close] * n, "volume": [10.0] * n,
        "quote_volume": [20.0] * n, "trade_num": [3] * n, "taker_buy_base_asset_volume": [5.0] * n,
        "taker_buy_quote_asset_volume": [5.0] * n, "avg_price": [2.0] * n,
    })


def make_listener(writes, remedies):
    async def data_cb(run_time, batch, market):
        writes.append((run_time, set(batch)))

    async def remedy_cb(run_time, partial, missing, market):
        remedies.append((run_time, set(partial), set(missing), market))

    listener = MarketListener(MARKET, data_callback=data_cb, remedy_callback=remedy_cb)
    listener.symbols = {"A", "B", "C"}
    return listener


# ── MarketListener ──────────────────────────────────────────────────────────────

def test_incomplete_batch_is_remedied_and_late_candles_dropped():
    writes, remedies = [], []
    listener = make_listener(writes, remedies)

    async def run():
        await listener._handle_candle(ws_req("A"))
        await listener._handle_candle(ws_req("B"))
        # the watchdog computes run_time from the clock (a pytz-UTC datetime, as
        # next_run_time builds it), the batches are keyed by the candle's Timestamp:
        # both must address the same batch
        clock_rt = pytz.UTC.localize(datetime(2026, 11, 5, 1, 40))
        assert await listener.remedy_incomplete(clock_rt) is True
        assert remedies == [(RT, {"A", "B"}, {"C"}, MARKET)]
        assert writes == []
        assert RT not in listener._data_batches and RT not in listener._batch_status
        # the missing candle arriving after the remedy must not start a new batch or write
        await listener._handle_candle(ws_req("C"))
        assert writes == [] and RT not in listener._data_batches
        assert await listener.remedy_incomplete(RT) is False
        assert remedies == [(RT, {"A", "B"}, {"C"}, MARKET)]
        # the next bar is unaffected
        await listener._handle_candle(ws_req("C", RT + INTERVAL))
        assert RT + INTERVAL in listener._data_batches

    asyncio.run(run())


def test_complete_batch_is_written_once_and_not_remedied():
    writes, remedies = [], []
    listener = make_listener(writes, remedies)

    async def run():
        for s in ("A", "B", "C"):
            await listener._handle_candle(ws_req(s))
        assert writes == [(RT, {"A", "B", "C"})]
        assert await listener.remedy_incomplete(RT) is False
        assert remedies == []
        # a duplicate (e.g. after a reconnect) does not write the bar twice
        await listener._handle_candle(ws_req("A"))
        assert writes == [(RT, {"A", "B", "C"})]

    asyncio.run(run())


def test_batch_without_any_candle_is_remedied_in_full():
    writes, remedies = [], []
    listener = make_listener(writes, remedies)

    async def run():
        assert await listener.remedy_incomplete(RT) is True
        assert remedies == [(RT, set(), {"A", "B", "C"}, MARKET)]

    asyncio.run(run())


def test_remedy_errors_do_not_escape():
    calls = []

    async def failing(run_time, partial, missing, market):
        calls.append("fail")
        raise RuntimeError("boom")

    async def run():
        listener = MarketListener(MARKET, remedy_callback=failing)
        listener.symbols = {"A"}
        assert await listener.remedy_incomplete(RT) is True
        assert RT in listener._settled

    asyncio.run(run())
    assert calls == ["fail"]


def test_gap_at_close_runs_full_remedy_and_does_not_wait():
    full_calls = []
    partial_calls = []

    async def full(run_time, market):
        full_calls.append((pd.Timestamp(run_time), market))

    async def partial(run_time, partial_batch, missing, market):
        partial_calls.append(market)

    listener = MarketListener(
        MARKET, remedy_callback=partial, full_remedy_callback=full, gap_callback=lambda run_time, market: True,
    )
    listener.symbols = {"A", "B", "C"}

    async def run():
        await listener._handle_candle(ws_req("A"))
        assert await listener.on_bar_close(RT) == "full"
        assert full_calls == [(RT, MARKET)]
        assert partial_calls == []
        assert listener._settled[RT] == "full"
        assert RT not in listener._data_batches
        await listener._handle_candle(ws_req("B"))
        assert await listener.remedy_incomplete(RT) is False
        assert partial_calls == []

    asyncio.run(run())


def test_adjacent_close_waits_and_does_not_remedy_yet():
    called = []

    async def full(run_time, market):
        called.append("full")

    async def partial(run_time, partial_batch, missing, market):
        called.append("partial")

    listener = MarketListener(
        MARKET, remedy_callback=partial, full_remedy_callback=full, gap_callback=lambda run_time, market: False,
    )
    listener.symbols = {"A"}

    async def run():
        assert await listener.on_bar_close(RT) == "wait"
        assert called == []
        assert RT not in listener._settled

    asyncio.run(run())


class FakeGroup:
    def __init__(self):
        self.forced = 0

    async def force_reconnect(self):
        self.forced += 1


def test_incomplete_batch_reconnects_only_missing_groups():
    remedies = []

    async def remedy_cb(run_time, partial, missing, market):
        remedies.append(set(missing))

    listener = MarketListener(MARKET, remedy_callback=remedy_cb)
    listener.symbols = {"A", "B", "C"}
    groups = {i: FakeGroup() for i in range(FETCH_CONCURRENCY)}
    listener.listeners = groups

    async def run():
        await listener._handle_candle(ws_req("A"))
        await listener._handle_candle(ws_req("B"))
        assert await listener.remedy_incomplete(RT) is True

    asyncio.run(run())
    assert remedies == [{"C"}]
    missing_group = hash("C") % FETCH_CONCURRENCY
    assert groups[missing_group].forced == 1
    assert sum(group.forced for group in groups.values()) == 1


def test_force_disconnect_unblocks_recv_with_sentinel():
    socket = ReconnectingWebsocket(url="wss://example.invalid/", path="btcusdt@kline_5m")

    async def run():
        await socket.force_disconnect()
        assert socket.ws_state == WSListenerState.EXITING
        message = await asyncio.wait_for(socket.recv(), timeout=1)
        assert message == {"e": "forced_reconnect"}

    asyncio.run(run())


def test_settled_bars_are_bounded():
    listener = MarketListener(MARKET, remedy_callback=lambda *a: None)
    for i in range(MarketListener.SETTLED_KEEP + 5):
        listener._mark_settled(RT + i * INTERVAL, "websocket")
    assert len(listener._settled) == MarketListener.SETTLED_KEEP
    assert RT not in listener._settled
    assert RT + (MarketListener.SETTLED_KEEP + 4) * INTERVAL in listener._settled


def test_watchdog_without_remedy_callback_is_not_started():
    listener = MarketListener(MARKET)
    listener.symbols = {"A"}
    assert asyncio.run(listener.remedy_incomplete(RT)) is False


# ── WebsocketsDataJobs._remedy_incomplete_batch ────────────────────────────────

class FakeClient:
    def __init__(self):
        self.written = []

    def write_kline(self, df, market, interval, current_time):
        self.written.append((df, market, current_time))


class FakeFetcher:
    """Stands in for OptimizedKlineFetcher: C has the closed bar, D fails, E is not on REST
    yet, H hangs (a symbol stuck in retry back-off)."""
    requested = []

    def __init__(self, fetcher, max_concurrent):
        pass

    async def get_klines_with_retry(self, symbol, interval, limit):
        FakeFetcher.requested.append((symbol, interval, limit))
        if symbol == "D":
            return {"symbol": symbol, "success": False, "error": "boom"}
        if symbol == "H":
            await asyncio.sleep(10)
        if symbol == "E":
            return {"symbol": symbol, "success": True, "data": rest_df(symbol, [RT - 3 * INTERVAL, RT - 2 * INTERVAL])}
        return {"symbol": symbol, "success": True, "data": rest_df(symbol, [RT - INTERVAL, RT])}


@pytest.fixture
def jobs(monkeypatch):
    @contextlib.asynccontextmanager
    async def fake_session(timeout):
        yield None

    monkeypatch.setattr(data_jobs, "create_aiohttp_session", fake_session)
    monkeypatch.setattr(data_jobs, "BinanceFetcher", lambda market, session: None)
    monkeypatch.setattr(data_jobs, "OptimizedKlineFetcher", FakeFetcher)
    FakeFetcher.requested = []
    j = WebsocketsDataJobs.__new__(WebsocketsDataJobs)
    j._client = FakeClient()
    j.duck_time = {MARKET: RT - INTERVAL}
    return j


def partial_batch(*symbols):
    return {s: ws_req(s) for s in symbols}


def test_remedy_merges_websocket_rows_with_rest_rows(jobs):
    asyncio.run(jobs._remedy_incomplete_batch(MARKET, RT.to_pydatetime(), partial_batch("A", "B"), {"C"}))
    (df, market, current_time), = jobs._client.written
    assert market == MARKET and pd.Timestamp(current_time) == RT
    assert sorted(df["symbol"]) == ["A", "B", "C"]
    assert (df["open_time"] == RT - INTERVAL).all()          # the current, still-open bar is dropped
    assert df.set_index("symbol").loc["C", "close"] == 2.0   # REST row
    assert df.set_index("symbol").loc["A", "close"] == 1.0   # websocket row
    assert FakeFetcher.requested == [("C", data_jobs.KLINE_INTERVAL, 2)]
    assert jobs.duck_time[MARKET] == RT


def test_remedy_with_rest_failures_does_not_write(jobs):
    asyncio.run(jobs._remedy_incomplete_batch(MARKET, RT, partial_batch("A"), {"C", "D", "E"}))
    assert jobs._client.written == []
    assert jobs.duck_time[MARKET] == RT - INTERVAL


def test_remedy_fetch_deadline_does_not_write(jobs, monkeypatch):
    monkeypatch.setattr(data_jobs, "BATCH_REMEDY_FETCH_SEC", 0.05)
    asyncio.run(jobs._remedy_incomplete_batch(MARKET, RT, partial_batch("A"), {"C", "H"}))
    assert jobs._client.written == []
    assert jobs.duck_time[MARKET] == RT - INTERVAL


def test_remedy_with_nothing_to_write_leaves_state_alone(jobs):
    asyncio.run(jobs._remedy_incomplete_batch(MARKET, RT, {}, {"D"}))
    assert jobs._client.written == []
    assert jobs.duck_time[MARKET] == RT - INTERVAL


def test_single_bar_remedy_refuses_a_gap(jobs, monkeypatch):
    calls = []

    async def full(market, current_time, interval):
        calls.append((market, pd.Timestamp(current_time), interval))

    monkeypatch.setattr(jobs, "_fetch_and_insert_binance_data_async", full)
    jobs.duck_time[MARKET] = RT - 2 * INTERVAL
    asyncio.run(jobs._remedy_incomplete_batch(MARKET, RT, partial_batch("A"), {"C"}))
    assert calls == []
    assert jobs._client.written == [] and FakeFetcher.requested == []
    assert jobs.duck_time[MARKET] == RT - 2 * INTERVAL


def test_bar_has_gap_compares_close_times(jobs):
    jobs.duck_time[MARKET] = RT - INTERVAL
    assert jobs._bar_has_gap(RT, MARKET) is False
    jobs.duck_time[MARKET] = RT - 2 * INTERVAL
    assert jobs._bar_has_gap(RT, MARKET) is True


# ── OptimizedKlineFetcher.get_all_klines ───────────────────────────────────────

class FakeBinanceFetcher:
    trade_type = MARKET

    def get_api_limits(self):
        return 2400, 5


def make_all_klines_fetcher(fail_rounds_by_symbol):
    """A fetcher whose symbol fails for the first N rounds given in ``fail_rounds_by_symbol``."""
    fetcher = OptimizedKlineFetcher(FakeBinanceFetcher(), max_concurrent=4)
    calls = {}

    async def fake(symbol, interval, limit, end_timestamp=None):
        calls[symbol] = calls.get(symbol, 0) + 1
        if calls[symbol] <= fail_rounds_by_symbol.get(symbol, 0):
            return {"symbol": symbol, "success": False, "error": "boom"}
        return {"symbol": symbol, "success": True, "data": rest_df(symbol, [RT - INTERVAL]),
                "num": 1, "begin_time": RT - INTERVAL}

    fetcher.get_klines_with_retry = fake
    return fetcher, calls


def test_get_all_klines_retries_only_failed_symbols_and_gives_up():
    fetcher, calls = make_all_klines_fetcher({"B": 1, "D": 99})
    results = asyncio.run(fetcher.get_all_klines(["A", "B", "C", "D"], interval="5m", limit=2))
    ok = sorted(r["symbol"] for r in results if r["success"])
    assert ok == ["A", "B", "C"]
    assert calls == {"A": 1, "C": 1, "B": 2, "D": OptimizedKlineFetcher.MAX_FAILED_ROUNDS}


def test_get_all_klines_all_good_is_one_round():
    fetcher, calls = make_all_klines_fetcher({})
    results = asyncio.run(fetcher.get_all_klines(["A", "B"], interval="5m", limit=2))
    assert all(r["success"] for r in results) and calls == {"A": 1, "B": 1}
