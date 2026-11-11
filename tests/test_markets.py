"""Markets: DATA_SOURCES validation, the usdt_perp / others_perp split, and the exginfo
writes that two instances of the plugin share.

Run with ``PYTHONPATH=. pytest tests``.
"""
import math
import os
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pytest

from binance_ingestor.bus import TRADE_TYPE_MAP
from binance_ingestor.component.candle_fetcher import BinanceFetcher
from binance_ingestor.data_jobs import WebsocketsDataJobs
from binance_ingestor.duckport_client import DuckportClient
from binance_ingestor.hist.data_lister import get_download_prefix
from binance_ingestor.markets import (
    KLINE_MARKETS, parse_data_sources, select_archive_symbols, venue_of,
)


# ── DATA_SOURCES ──────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("usdt_perp,usdt_spot", ("usdt_perp", "usdt_spot")),
    ("usdt_spot, usdt_perp", ("usdt_perp", "usdt_spot")),      # processing order, not input order
    ("usdt_perp", ("usdt_perp",)),
    ("usdt_spot", ("usdt_spot",)),
    ("others_perp", ("others_perp",)),
    ("", ()),
    (" , ", ()),
])
def test_data_sources_accepted(raw, expected):
    assert parse_data_sources(raw) == expected


@pytest.mark.parametrize("raw", ["usdt_perp,others_perp", "others_perp,usdt_spot",
                                 "usdt_perp,usdt_spot,others_perp"])
def test_usdt_and_other_markets_cannot_share_an_instance(raw):
    with pytest.raises(ValueError, match="不能同时包含"):
        parse_data_sources(raw)


@pytest.mark.parametrize("raw", ["usdc_perp", "usdt_perp,coin_perp", "USDT_PERP", "others"])
def test_unknown_market_is_an_error_not_ignored(raw):
    with pytest.raises(ValueError, match="未知市场"):
        parse_data_sources(raw)


@pytest.mark.parametrize("entry", ["binance_ingestor.main", "binance_ingestor.loadhist"])
def test_both_entry_points_reject_a_mixed_instance_before_connecting(entry, tmp_path):
    """The address points nowhere: a connection attempt would fail differently."""
    env = dict(os.environ, DATA_SOURCES="usdt_perp,others_perp", INGESTOR_ENV_FILE=str(tmp_path / "none.env"),
               DUCKPORT_LISTEN_ADDR="127.0.0.1:1", PYTHONPATH=os.getcwd())
    proc = subprocess.run([sys.executable, "-c", f"import {entry} as m; m.main()"],
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "不能同时包含" in proc.stderr
    assert "Cannot reach" not in proc.stderr + proc.stdout


# ── symbol split ──────────────────────────────────────────────────────

def um_info(symbol, quote, contract="PERPETUAL", status="TRADING"):
    """One /fapi/v1/exchangeInfo symbol entry, reduced to what the parser reads."""
    return {"symbol": symbol, "contractType": contract, "underlyingType": "COIN", "status": status,
            "baseAsset": symbol[:3], "quoteAsset": quote, "marginAsset": quote,
            "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"}]}


UM_EXCHANGE_INFO = [
    um_info("BTCUSDT", "USDT"),
    um_info("XAUUSDT", "USDT", contract="TRADIFI_PERPETUAL"),
    um_info("BTCUSDT_261225", "USDT", contract="CURRENT_QUARTER"),
    um_info("OLDUSDT", "USDT", status="SETTLING"),
    um_info("BTCUSDC", "USDC"),
    um_info("ETHUSDC", "USDC"),
    um_info("BTCUSD1", "USD1"),
    um_info("SPCXUSD1", "USD1", contract="TRADIFI_PERPETUAL"),
    um_info("BTCU", "U"),
    um_info("ETHBTC", "BTC"),
    um_info("OLDUSDC", "USDC", status="SETTLING"),
    um_info("BTCUSDC_261225", "USDC", contract="CURRENT_QUARTER"),
]


def parsed(market):
    fetcher = BinanceFetcher(market, session=None)
    return {i["symbol"]: fetcher.syminfo_parse_func(i) for i in UM_EXCHANGE_INFO}


def test_usdt_perp_symbols_are_what_they_were():
    assert sorted(TRADE_TYPE_MAP["usdt_perp"][0](parsed("usdt_perp"))) == ["BTCUSDT", "XAUUSDT"]


def test_others_perp_takes_every_other_quote():
    assert sorted(TRADE_TYPE_MAP["others_perp"][0](parsed("others_perp"))) == [
        "BTCU", "BTCUSD1", "BTCUSDC", "ETHBTC", "ETHUSDC", "SPCXUSD1"]


def test_the_two_perp_markets_split_the_trading_perpetuals():
    usdt = set(TRADE_TYPE_MAP["usdt_perp"][0](parsed("usdt_perp")))
    others = set(TRADE_TYPE_MAP["others_perp"][0](parsed("others_perp")))
    trading_perps = {i["symbol"] for i in UM_EXCHANGE_INFO
                     if i["status"] == "TRADING" and "PERPETUAL" in i["contractType"]}
    assert not usdt & others
    assert usdt | others == trading_perps


@pytest.mark.parametrize("market", ["usdt_perp", "others_perp"])
def test_exginfo_rows_carry_the_market_being_fetched(market):
    """The row's market used to be the constant 'usdt_perp' for every USD-M symbol."""
    assert {row["market"] for row in parsed(market).values()} == {market}


def test_spot_rows_keep_their_market():
    info = {"symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC", "quoteAsset": "USDT",
            "filters": [], "permissionSets": []}
    assert BinanceFetcher("usdt_spot", session=None).syminfo_parse_func(info)["market"] == "usdt_spot"


def test_unknown_market_has_no_venue():
    with pytest.raises(ValueError):
        venue_of("usdc_perp")
    with pytest.raises(ValueError):
        BinanceFetcher("usdc_perp", session=None)


# ── archive index (loadhist) ──────────────────────────────────────────

ARCHIVE_UM = ["BTCUSDT", "1000SHIBUSDT", "BTCUSDC", "MATICUSDC", "BTCUSD1", "BTCU", "ETHBTC", "BTCBUSD",
              "BTCUSDT_210326", "BTCBUSD_210129", "BNXUSDTSETTLED", "AERGOUSDTSETTLEDSETTLED"]


def test_archive_symbols_usdt_perp_unchanged():
    assert select_archive_symbols("usdt_perp", ARCHIVE_UM) == {"BTCUSDT", "1000SHIBUSDT"}


def test_archive_symbols_others_perp():
    """Delivery contracts and SETTLED leftovers belong to neither market."""
    assert select_archive_symbols("others_perp", ARCHIVE_UM) == {
        "BTCUSDC", "MATICUSDC", "BTCUSD1", "BTCU", "ETHBTC", "BTCBUSD"}


def test_archive_markets_do_not_overlap():
    assert not select_archive_symbols("usdt_perp", ARCHIVE_UM) & select_archive_symbols("others_perp", ARCHIVE_UM)


@pytest.mark.parametrize("market, path", [("usdt_perp", "data/futures/um"), ("others_perp", "data/futures/um"),
                                          ("usdt_spot", "data/spot")])
def test_archive_prefix_follows_the_venue(market, path):
    assert get_download_prefix(market, "klines", "daily", None, "1m") == f"{path}/daily/klines/"
    assert get_download_prefix(market, "klines", "monthly", "btcusdc", "1m") == f"{path}/monthly/klines/BTCUSDC/1m/"


# ── exginfo writes ────────────────────────────────────────────────────

class RecordingClient(DuckportClient):
    """DuckportClient without a server: records what it would send."""

    def __init__(self):
        self.schema = "data"
        self.interval = "1m"
        self.ingestor = "test"
        self.max_lag_seconds = 900
        self.sent = []

    def execute(self, sql):
        self.sent.append(("execute", sql))
        return {}

    def execute_transaction(self, statements):
        self.sent.append(("tx", list(statements)))
        return {}

    def append(self, schema, table, data):
        self.sent.append(("append", table, data.num_rows))
        return {}


def exginfo_df(market, symbols):
    return pd.DataFrame([{
        "market": market, "symbol": s, "status": "TRADING", "base_asset": s[:3], "quote_asset": "X",
        "price_tick": "0.1", "lot_size": "0.001", "min_notional_value": "5", "contract_type": "PERPETUAL",
        "margin_asset": "X", "pre_market": None, "underlying_type": "COIN"} for s in symbols])


def test_each_market_refreshes_exginfo_through_its_own_staging_table():
    """A shared staging table let one instance's TRUNCATE / INSERT hit the other's rows."""
    client = RecordingClient()
    client.save_exginfo(exginfo_df("usdt_perp", ["BTCUSDT"]), "usdt_perp")
    client.save_exginfo(exginfo_df("others_perp", ["BTCUSDC", "BTCU"]), "others_perp")
    text = {m: " ".join(str(x) for x in client.sent[i * 3:(i + 1) * 3])
            for i, m in enumerate(["usdt_perp", "others_perp"])}
    assert "_staging_exginfo_usdt_perp" in text["usdt_perp"]
    assert "_staging_exginfo_others_perp" not in text["usdt_perp"]
    assert "_staging_exginfo_others_perp" in text["others_perp"]
    assert "_staging_exginfo_usdt_perp" not in text["others_perp"]
    for market, sent in text.items():
        assert f"DELETE FROM data.exginfo WHERE market = '{market}'" in sent
        assert "_staging_exginfo " not in sent and "_staging_exginfo\"" not in sent


def test_init_schema_creates_only_the_enabled_markets():
    client = RecordingClient()
    client.init_schema(markets=list(KLINE_MARKETS), interval="1m", data_sources={"others_perp"},
                       start_date="2024-01-01")
    ddl = " ".join(client.sent[0][1])
    assert "CREATE TABLE IF NOT EXISTS data.others_perp_1m" in ddl
    assert "CREATE TABLE _staging_exginfo_others_perp" in ddl
    assert "usdt_perp" not in ddl and "usdt_spot" not in ddl
    specs = " ".join(client.sent[1][1])
    assert "'others_perp_1m'" in specs and "'exginfo'" in specs


# ── zero-volume candles ───────────────────────────────────────────────

def test_websocket_zero_volume_candle_has_null_avg_price():
    """REST and loadhist store NULL for a candle without trades; the websocket path wrote 0."""
    candle = pd.DataFrame({"open_time": [pd.Timestamp("2026-11-05 01:39:00", tz="UTC")], "open": [1.0],
                           "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0], "quote_volume": [0.0],
                           "trade_num": [0], "taker_buy_base_asset_volume": [0.0],
                           "taker_buy_quote_asset_volume": [0.0]})
    df = WebsocketsDataJobs._convert_market_data({"data": candle, "symbol": "BTCU"}, "others_perp")
    assert math.isnan(df["avg_price"].iloc[0])
    table = pa.Table.from_pandas(df, preserve_index=False)
    assert table["avg_price"].null_count == 1


def test_websocket_traded_candle_keeps_its_avg_price():
    candle = pd.DataFrame({"open_time": [pd.Timestamp("2026-11-05 01:39:00", tz="UTC")], "open": [1.0],
                           "high": [1.0], "low": [1.0], "close": [1.0], "volume": [4.0], "quote_volume": [10.0],
                           "trade_num": [2], "taker_buy_base_asset_volume": [1.0],
                           "taker_buy_quote_asset_volume": [2.5]})
    df = WebsocketsDataJobs._convert_market_data({"data": candle, "symbol": "BTCUSDC"}, "others_perp")
    assert df["avg_price"].iloc[0] == 2.5
