"""Bars around a listing or delisting that the websocket does not deliver.

A new symbol is subscribed only after exchangeInfo shows it TRADING, so its first bar
closes before the subscription exists. A delisted perpetual turns SETTLING about 30
minutes before delivery, leaves the subscription, and keeps trading until delivery.
``DataJobs._fill_listing_edges`` writes those bars over REST without moving the watermark.

Run with ``PYTHONPATH=. pytest tests``.
"""
import asyncio
import contextlib
from datetime import timedelta

import pandas as pd
import pytest

from binance_ingestor import data_jobs
from binance_ingestor.component.candle_fetcher import _parse_usdt_futures_syminfo
from binance_ingestor.component.candle_listener import MarketListener
from binance_ingestor.config import GENESIS_TIME
from binance_ingestor.data_jobs import EDGE_FOLLOW_BARS, WebsocketsDataJobs

NOW = pd.Timestamp("2026-11-27 08:31:00", tz="UTC")
INTERVAL = timedelta(minutes=data_jobs.KLINE_INTERVAL_MINUTES)
MARKET = "usdt_perp"
FAR = pd.Timestamp("2100-12-25 08:00:00", tz="UTC")


def bars(symbol, first, last):
    """REST candles of ``symbol`` from ``first`` to ``last`` inclusive (the last one still open)."""
    times = pd.date_range(first, last, freq=INTERVAL)
    n = len(times)
    return pd.DataFrame({
        "open_time": times, "symbol": [symbol] * n, "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n,
        "close": [1.0] * n, "volume": [10.0] * n, "quote_volume": [10.0] * n, "trade_num": [3] * n,
        "taker_buy_base_asset_volume": [5.0] * n, "taker_buy_quote_asset_volume": [5.0] * n, "avg_price": [1.0] * n,
    })


class FakeClient:
    def __init__(self):
        self.written = []

    def write_kline(self, df, market, interval, current_time):
        self.written.append((df, market, current_time))
        return {"rows_affected": [len(df), 1, len(df)]}


class FakeFetcher:
    rest = {}
    requested = []

    def __init__(self, fetcher, max_concurrent):
        pass

    async def get_klines_with_retry(self, symbol, interval, limit):
        FakeFetcher.requested.append(symbol)
        if symbol not in FakeFetcher.rest:
            return {"symbol": symbol, "success": False, "error": "boom"}
        return {"symbol": symbol, "success": True, "data": FakeFetcher.rest[symbol]}


@pytest.fixture
def jobs(monkeypatch):
    @contextlib.asynccontextmanager
    async def fake_session(timeout):
        yield None

    monkeypatch.setattr(data_jobs, "create_aiohttp_session", fake_session)
    monkeypatch.setattr(data_jobs, "BinanceFetcher", lambda market, session: None)
    monkeypatch.setattr(data_jobs, "OptimizedKlineFetcher", FakeFetcher)
    monkeypatch.setattr(data_jobs, "now_time", lambda: (clock["now"] + timedelta(seconds=10)).to_pydatetime())
    clock["now"] = NOW
    FakeFetcher.rest, FakeFetcher.requested = {}, []
    j = WebsocketsDataJobs.__new__(WebsocketsDataJobs)
    j._client = FakeClient()
    j.duck_time = {MARKET: NOW}
    j._settling = {}
    j._edge_follow = {}
    return j


clock = {"now": NOW}


def fill(jobs, onboard=(), delist=()):
    asyncio.run(jobs._fill_listing_edges(MARKET, set(onboard), set(delist)))


def written(jobs):
    (df, market, current_time), = jobs._client.written
    assert market == MARKET
    return df, pd.Timestamp(current_time)


def test_nothing_to_follow_fetches_nothing(jobs):
    fill(jobs)
    assert FakeFetcher.requested == [] and jobs._client.written == []


def test_new_symbol_gets_the_bars_before_its_subscription(jobs):
    # first trade 08:30, seen at the 08:31 round: the 08:30 bar closed before the subscription
    FakeFetcher.rest["NEW"] = bars("NEW", NOW - INTERVAL, NOW)
    fill(jobs, onboard={"NEW"})
    df, current_time = written(jobs)
    assert list(df["open_time"]) == [NOW - INTERVAL]      # the bar still open is left to the websocket
    assert current_time == NOW                            # the watermark does not move
    assert jobs.duck_time[MARKET] == NOW


def test_bars_past_the_watermark_wait_for_the_market_batch(jobs):
    # the 08:31 market batch is not written yet: the 08:30 bar would be ahead of the watermark
    jobs.duck_time[MARKET] = NOW - INTERVAL
    FakeFetcher.rest["NEW"] = bars("NEW", NOW - INTERVAL, NOW)
    fill(jobs, onboard={"NEW"})
    assert jobs._client.written == []

    # next round the batch is in, and the symbol is still followed
    clock["now"] = NOW + INTERVAL
    jobs.duck_time[MARKET] = NOW + INTERVAL
    FakeFetcher.rest["NEW"] = bars("NEW", NOW - INTERVAL, NOW + INTERVAL)
    fill(jobs)
    df, _ = written(jobs)
    assert list(df["open_time"]) == [NOW - INTERVAL, NOW]


def test_a_followed_symbol_is_dropped_after_a_few_bars(jobs):
    FakeFetcher.rest["NEW"] = bars("NEW", NOW - INTERVAL, NOW)
    fill(jobs, onboard={"NEW"})
    for step in range(1, EDGE_FOLLOW_BARS + 1):
        clock["now"] = NOW + step * INTERVAL
        fill(jobs)
    assert FakeFetcher.requested == ["NEW"] * (EDGE_FOLLOW_BARS + 1)
    clock["now"] = NOW + (EDGE_FOLLOW_BARS + 1) * INTERVAL
    fill(jobs)
    assert FakeFetcher.requested == ["NEW"] * (EDGE_FOLLOW_BARS + 1)
    assert jobs._edge_follow[MARKET] == {}


def test_settling_contract_is_followed_until_delivery(jobs):
    delivery = NOW + 29 * INTERVAL                         # 09:00
    jobs._settling = {MARKET: {"OLD": delivery}}

    # 08:31 round: the symbol just left the trading set
    FakeFetcher.rest["OLD"] = bars("OLD", NOW - 5 * INTERVAL, NOW)
    fill(jobs, delist={"OLD"})
    df, current_time = written(jobs)
    assert df["open_time"].max() == NOW - INTERVAL and current_time == NOW

    # 09:03 round: REST keeps returning flat bars after delivery; they are not written
    jobs._client.written.clear()
    clock["now"] = delivery + 3 * INTERVAL
    jobs.duck_time[MARKET] = delivery + 3 * INTERVAL
    FakeFetcher.rest["OLD"] = bars("OLD", delivery - 5 * INTERVAL, delivery + 3 * INTERVAL)
    fill(jobs)
    df, _ = written(jobs)
    assert df["open_time"].max() == delivery - INTERVAL

    # after the follow window the symbol is no longer fetched
    FakeFetcher.requested.clear()
    clock["now"] = delivery + (EDGE_FOLLOW_BARS + 1) * INTERVAL
    fill(jobs)
    assert FakeFetcher.requested == []


def test_settling_contract_is_picked_up_after_a_restart(jobs):
    # no delist event (the plugin started inside the settlement window): exchangeInfo is enough
    jobs._settling = {MARKET: {"OLD": NOW + 10 * INTERVAL}}
    FakeFetcher.rest["OLD"] = bars("OLD", NOW - 5 * INTERVAL, NOW)
    fill(jobs)
    df, _ = written(jobs)
    assert set(df["symbol"]) == {"OLD"}


def test_long_settled_contract_is_ignored(jobs):
    jobs._settling = {MARKET: {"GONE": NOW - timedelta(days=30)}}
    fill(jobs)
    assert FakeFetcher.requested == []


def test_contract_not_trading_yet_is_ignored(jobs):
    # PENDING_TRADING: not TRADING, delivery time in 2100, and REST has no klines for it
    jobs._settling = {MARKET: {"SOON": FAR}}
    fill(jobs)
    assert FakeFetcher.requested == []


def test_symbol_without_delivery_gets_its_last_closed_bars(jobs):
    # e.g. a spot pair: no bars at or after the round that saw it leave
    FakeFetcher.rest["HALT"] = bars("HALT", NOW - 3 * INTERVAL, NOW - INTERVAL)
    fill(jobs, delist={"HALT"})
    df, _ = written(jobs)
    assert df["open_time"].max() == NOW - INTERVAL


def test_relisted_symbol_is_followed_as_new(jobs):
    FakeFetcher.rest["BACK"] = bars("BACK", NOW - 3 * INTERVAL, NOW)
    fill(jobs, delist={"BACK"})
    clock["now"] = NOW + INTERVAL
    jobs.duck_time[MARKET] = NOW + INTERVAL
    jobs._client.written.clear()
    FakeFetcher.rest["BACK"] = bars("BACK", NOW - 3 * INTERVAL, NOW + INTERVAL)
    fill(jobs, onboard={"BACK"})
    df, _ = written(jobs)
    assert df["open_time"].max() == NOW                    # not capped at the delist round any more


def test_rest_failure_writes_the_other_symbols(jobs):
    FakeFetcher.rest["NEW"] = bars("NEW", NOW - INTERVAL, NOW)
    fill(jobs, onboard={"NEW", "FAILS"})
    df, _ = written(jobs)
    assert set(df["symbol"]) == {"NEW"}


def test_nothing_is_written_before_the_first_watermark(jobs):
    jobs.duck_time[MARKET] = GENESIS_TIME
    FakeFetcher.rest["NEW"] = bars("NEW", NOW - INTERVAL, NOW)
    fill(jobs, onboard={"NEW"})
    assert jobs._client.written == []


# ── exchangeInfo → settling symbols ────────────────────────────────────────────

def um_info(symbol, status, delivery, quote="USDT", contract_type="PERPETUAL"):
    return {"symbol": symbol, "status": status, "quote_asset": quote, "contract_type": contract_type,
            "delivery_time": delivery}


def test_settling_symbols_are_this_markets_contracts_with_a_delivery_time():
    delivery = NOW + 29 * INTERVAL
    exginfo = {i["symbol"]: i for i in [
        um_info("BTCUSDT", "TRADING", FAR),
        um_info("OLDUSDT", "SETTLING", delivery),
        um_info("OLDUSDC", "SETTLING", delivery, quote="USDC"),
        um_info("QTRUSDT", "SETTLING", delivery, contract_type="CURRENT_QUARTER"),
        um_info("NODATE", "SETTLING", None),
    ]}
    assert WebsocketsDataJobs._settling_symbols("usdt_perp", exginfo) == {"OLDUSDT": delivery}
    assert WebsocketsDataJobs._settling_symbols("others_perp", exginfo) == {"OLDUSDC": delivery}


def test_delivery_time_is_parsed_from_exchange_info():
    raw = {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "underlyingType": "COIN", "status": "SETTLING",
           "baseAsset": "OLD", "quoteAsset": "USDT", "marginAsset": "USDT", "onboardDate": 1700000000000,
           "deliveryDate": 1793178000000, "filters": []}
    info = _parse_usdt_futures_syminfo(raw)
    assert info["delivery_time"] == pd.Timestamp("2026-10-28 09:00:00", tz="UTC")


# ── MarketListener calls the edge callback every round ─────────────────────────

class FakeListener:
    def add_symbols(self, *symbols): pass
    def remove_symbols(self, *symbols): pass
    def reconnect(self): pass


def test_listener_reports_edges_every_round_and_survives_a_failing_callback():
    calls = []

    async def edge_cb(market, onboard, delist):
        calls.append((market, set(onboard), set(delist)))
        if len(calls) == 2:
            raise RuntimeError("rest down")

    listener = MarketListener(MARKET, edge_callback=edge_cb)
    listener.symbols = {"A", "B"}
    listener.listeners = {i: FakeListener() for i in range(64)}
    asyncio.run(listener.update_exginfo(["A", "C"]))
    asyncio.run(listener.update_exginfo(["A", "C"]))       # raises inside; the listener goes on
    asyncio.run(listener.update_exginfo(["A", "C"]))
    assert calls == [(MARKET, {"C"}, {"B"}), (MARKET, set(), set()), (MARKET, set(), set())]
    assert listener.symbols == {"A", "C"}
