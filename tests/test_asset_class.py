"""Asset columns of the perpetual kline tables: contract_type / underlying_type per row.

Run with ``PYTHONPATH=. pytest tests``.
"""
from datetime import datetime

import duckdb
import pandas as pd
import pyarrow as pa
import pytest

from binance_ingestor.asset_class import AssetClassifier, has_asset_columns
from binance_ingestor.component.candle_fetcher import _parse_usdt_futures_syminfo
from binance_ingestor.duckport_client import DuckportClient

CRYPTO = ("PERPETUAL", "COIN")
KLINE = ["open_time", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "trade_num",
         "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "avg_price"]


def info(contract_type, underlying_type, onboard):
    return {"contract_type": contract_type, "underlying_type": underlying_type,
            "onboard_time": pd.Timestamp(onboard) if onboard else None}


EXGINFO = {
    "BTCUSDT": info("PERPETUAL", "COIN", "2019-09-08 17:55"),
    "BTCDOMUSDT": info("PERPETUAL", "INDEX", "2021-06-30 07:00"),
    "XAUUSDT": info("TRADIFI_PERPETUAL", "COMMODITY", "2025-12-11 08:05"),
    # the name of a crypto perpetual delisted in 2024-04, listed again as TradFi pre-IPO
    "ANTUSDT": info("TRADIFI_PERPETUAL", "PREMARKET", "2026-11-17 06:00"),
    # pre-market perpetuals that became standard ones (KNOWN_PHASES has the period)
    "BABYUSDT": info("PERPETUAL", "COIN", "2025-04-05 13:30"),
    "SPCXUSDT": info("TRADIFI_PERPETUAL", "EQUITY", "2026-05-21 04:45"),
    # Litentry until 2025-01, listed again for another token with a pre-market period
    "LITUSDT": info("PERPETUAL", "COIN", "2025-12-23 17:30"),
    "PREDICTUSDT": info("PERPETUAL", "PREMARKET", "2026-10-25 08:15"),
    "MUUSDT": info("TRADIFI_PERPETUAL", "EQUITY", "2026-04-07 13:30"),
}


def classify(classifier, symbol, when):
    table = pa.table({"symbol": [symbol], "open_time": pa.array([pd.Timestamp(when)], type=pa.timestamp("us"))})
    out = classifier.stamp(table)
    return out["contract_type"][0].as_py(), out["underlying_type"][0].as_py()


@pytest.fixture
def clf():
    return AssetClassifier("usdt_perp", EXGINFO)


@pytest.mark.parametrize("symbol, when, expected", [
    ("BTCUSDT", "2021-01-01", CRYPTO),
    ("BTCDOMUSDT", "2024-01-01", ("PERPETUAL", "INDEX")),
    ("XAUUSDT", "2026-01-01", ("TRADIFI_PERPETUAL", "COMMODITY")),
    # a reused name: the earlier listing is not what exchangeInfo describes today
    ("ANTUSDT", "2024-04-01 08:59", CRYPTO),
    ("ANTUSDT", "2026-11-17 10:31", ("TRADIFI_PERPETUAL", "PREMARKET")),
    # past pre-market periods
    ("BABYUSDT", "2025-04-05 13:30", ("PERPETUAL", "PREMARKET")),
    ("BABYUSDT", "2025-04-10 11:09", ("PERPETUAL", "PREMARKET")),
    ("BABYUSDT", "2025-04-10 11:10", CRYPTO),
    ("SPCXUSDT", "2026-06-01", ("TRADIFI_PERPETUAL", "PREMARKET")),
    ("SPCXUSDT", "2026-06-12 16:15", ("TRADIFI_PERPETUAL", "EQUITY")),
    ("LITUSDT", "2021-02-18 07:00", CRYPTO),
    ("LITUSDT", "2025-12-24", ("PERPETUAL", "PREMARKET")),
    ("LITUSDT", "2026-02-01", CRYPTO),
    # still in pre-market: straight from exchangeInfo
    ("PREDICTUSDT", "2026-11-01", ("PERPETUAL", "PREMARKET")),
    # delisted TradFi contract, no longer in exchangeInfo
    ("ANTHROPICUSDT", "2026-08-01", ("TRADIFI_PERPETUAL", "PREMARKET")),
    # delisted, nothing known: crypto
    ("NKNUSDT", "2025-01-01", CRYPTO),
    # the first candle opens a minute before onboardDate
    ("MUUSDT", "2026-04-07 13:29", ("TRADIFI_PERPETUAL", "EQUITY")),
])
def test_row_class(clf, symbol, when, expected):
    assert classify(clf, symbol, when) == expected


def test_segments_of_one_symbol_do_not_overlap(clf):
    for symbol in list(EXGINFO) + ["ANTHROPICUSDT"]:
        segments = clf.segments(symbol)
        for a, b in zip(segments, segments[1:]):
            assert a[1] <= b[0], symbol


def test_symbol_that_left_exchange_info_keeps_its_class(clf):
    """Its last candles are written after it is gone from exchangeInfo."""
    clf.update({k: v for k, v in EXGINFO.items() if k != "XAUUSDT"})
    assert classify(clf, "XAUUSDT", "2026-11-18") == ("TRADIFI_PERPETUAL", "COMMODITY")


def test_update_takes_a_changed_type(clf):
    assert classify(clf, "PREDICTUSDT", "2026-11-20") == ("PERPETUAL", "PREMARKET")
    clf.update({"PREDICTUSDT": info("PERPETUAL", "COIN", "2026-10-25 08:15")})
    assert classify(clf, "PREDICTUSDT", "2026-11-20") == CRYPTO


def test_stamp_many_rows_and_existing_columns():
    clf = AssetClassifier("usdt_perp", EXGINFO)
    times = pd.to_datetime(["2024-01-01", "2026-11-18", "2026-11-18", "2026-11-18"])
    table = pa.table({"open_time": times, "symbol": ["ANTUSDT", "ANTUSDT", "BTCUSDT", None],
                      "contract_type": ["x"] * 4})
    out = clf.stamp(table)
    assert out.column_names == ["open_time", "symbol", "contract_type", "underlying_type"]
    assert out["contract_type"].to_pylist() == ["PERPETUAL", "TRADIFI_PERPETUAL", "PERPETUAL", "PERPETUAL"]
    assert out["underlying_type"].to_pylist() == ["COIN", "PREMARKET", "COIN", "COIN"]
    empty = clf.stamp(table.slice(0, 0))
    assert empty.num_rows == 0 and "underlying_type" in empty.column_names


def test_only_um_markets_have_the_columns():
    assert has_asset_columns("usdt_perp") and has_asset_columns("others_perp")
    assert not has_asset_columns("usdt_spot")
    assert len(DuckportClient.kline_col_defs("usdt_spot")) == 12
    assert len(DuckportClient.kline_col_defs("others_perp")) == 14


def test_exchange_info_parser_keeps_onboard_date():
    raw = {"symbol": "ANTUSDT", "contractType": "TRADIFI_PERPETUAL", "underlyingType": "PREMARKET",
           "status": "TRADING", "baseAsset": "ANT", "quoteAsset": "USDT", "marginAsset": "USDT",
           "filters": [], "onboardDate": 1794895200000}
    assert _parse_usdt_futures_syminfo(raw)["onboard_time"] == pd.Timestamp("2026-11-17 06:00")


# ── the SQL, run on a local DuckDB ────────────────────────────────────

class LocalClient(DuckportClient):
    """DuckportClient whose RPCs run on an in-memory DuckDB."""

    def __init__(self):
        self.schema, self.interval, self.ingestor, self.max_lag_seconds = "data", "1m", "test", 900
        self.append_max_message_bytes = 48 * 1024 * 1024
        self.transactions = []
        self.con = duckdb.connect()
        self.con.execute("CREATE SCHEMA data")
        self.con.execute("CREATE TABLE data.watermark (table_name VARCHAR PRIMARY KEY, ingestor VARCHAR, "
                         "max_lag_seconds INT, time_column VARCHAR, start_time TIMESTAMP, duck_time TIMESTAMP, "
                         "updated_at TIMESTAMP)")

    def execute(self, sql):
        return {"rows_affected": self.con.execute(sql).fetchall()}

    def execute_transaction(self, statements):
        self.transactions.append(statements)
        self.con.execute("BEGIN")
        try:
            affected = []
            for sql in statements:
                done = self.con.execute(sql)
                affected.append(done.fetchall()[0][0] if sql.startswith(("UPDATE", "DELETE")) else 0)
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return {"rows_affected": affected}

    def _query(self, sql):
        return self.con.execute(sql).to_arrow_table()

    def append(self, schema, table, data):
        self.con.register("incoming", data)
        self.con.execute(f"INSERT INTO {schema}.{table} BY NAME SELECT * FROM incoming")
        self.con.unregister("incoming")
        return {"rows_appended": data.num_rows}

    def read_table(self, schema, table):
        return self.con.execute(f"SELECT * FROM {schema}.{table}").to_arrow_table()

    def register_table_specs(self, statements):
        return False

    def rows(self, sql):
        return self.con.execute(sql).fetchall()


def candles(symbol, *times):
    df = pd.DataFrame({c: 1.0 for c in KLINE if c not in ("open_time", "symbol", "trade_num")}, index=range(len(times)))
    df["open_time"], df["symbol"], df["trade_num"] = pd.to_datetime(list(times)), symbol, 1
    return df[KLINE]


def init(client, sources=("usdt_perp", "usdt_spot")):
    client.init_schema(markets=["usdt_perp", "usdt_spot", "others_perp"], interval="1m",
                       data_sources=set(sources), start_date="2021-01-01")


def columns(client, table):
    return [r[0] for r in client.rows(f"SELECT column_name FROM information_schema.columns "
                                      f"WHERE table_schema = 'data' AND table_name = '{table}' ORDER BY ordinal_position")]


def test_new_tables_perp_has_the_columns_spot_does_not():
    client = LocalClient()
    init(client)
    assert columns(client, "usdt_perp_1m") == KLINE + ["contract_type", "underlying_type"]
    assert columns(client, "usdt_spot_1m") == KLINE


def test_upgrade_of_an_existing_table_then_backfill_and_live_write():
    client = LocalClient()
    old_cols = ", ".join(DuckportClient.KLINE_COL_DEFS)
    client.con.execute(f"CREATE TABLE data.usdt_perp_1m ({old_cols})")
    client.con.execute(f"CREATE TABLE main._staging_usdt_perp_1m ({old_cols})")      # as 0.6.0 left it
    old = pd.concat([
        candles("BTCUSDT", "2024-01-01 00:00"),
        candles("ANTUSDT", "2024-04-01 08:59", "2026-11-17 10:31"),
        candles("BABYUSDT", "2025-04-06 00:00", "2025-05-01 00:00"),
        candles("ANTHROPICUSDT", "2026-07-01 00:00"),
        candles("XAUUSDT", "2026-01-01 00:00"),
    ])
    client.con.execute("INSERT INTO data.usdt_perp_1m SELECT * FROM old")

    init(client)
    assert columns(client, "usdt_perp_1m") == KLINE + ["contract_type", "underlying_type"]
    assert client.rows("SELECT DISTINCT contract_type, underlying_type FROM data.usdt_perp_1m") == [CRYPTO]

    client.set_asset_classifier("usdt_perp", AssetClassifier("usdt_perp", EXGINFO))
    client.backfill_asset_columns("usdt_perp", "1m")
    got = {(s, str(t)): (c, u) for s, t, c, u in client.rows(
        "SELECT symbol, open_time, contract_type, underlying_type FROM data.usdt_perp_1m")}
    assert got == {
        ("BTCUSDT", "2024-01-01 00:00:00"): CRYPTO,
        ("ANTUSDT", "2024-04-01 08:59:00"): CRYPTO,
        ("ANTUSDT", "2026-11-17 10:31:00"): ("TRADIFI_PERPETUAL", "PREMARKET"),
        ("BABYUSDT", "2025-04-06 00:00:00"): ("PERPETUAL", "PREMARKET"),
        ("BABYUSDT", "2025-05-01 00:00:00"): CRYPTO,
        ("ANTHROPICUSDT", "2026-07-01 00:00:00"): ("TRADIFI_PERPETUAL", "PREMARKET"),
        ("XAUUSDT", "2026-01-01 00:00:00"): ("TRADIFI_PERPETUAL", "COMMODITY"),
    }

    # live write: one row already there, two new; the new ones carry their class
    live = pd.concat([candles("XAUUSDT", "2026-01-01 00:00", "2026-11-18 00:00"),
                      candles("NEWUSDT", "2026-11-18 00:00")])
    client.write_kline(live, "usdt_perp", "1m", datetime(2026, 11, 18, 0, 1))
    assert client.rows("SELECT count(*) FROM data.usdt_perp_1m") == [(9,)]
    assert client.rows("SELECT symbol, contract_type, underlying_type FROM data.usdt_perp_1m "
                       "WHERE open_time = TIMESTAMP '2026-11-18' ORDER BY 1") == [
        ("NEWUSDT", *CRYPTO), ("XAUUSDT", "TRADIFI_PERPETUAL", "COMMODITY")]


def test_backfill_runs_once():
    """A second run would overwrite what live writes recorded at their own time."""
    client = LocalClient()
    init(client)
    client.set_asset_classifier("usdt_perp", AssetClassifier("usdt_perp", EXGINFO))
    client.backfill_asset_columns("usdt_perp", "1m")
    client.write_kline(candles("PREDICTUSDT", "2026-11-18 00:00"), "usdt_perp", "1m", datetime(2026, 11, 18, 0, 1))
    # PREDICT becomes a standard perpetual; the row written before that keeps PREMARKET
    client.set_asset_classifier("usdt_perp", AssetClassifier(
        "usdt_perp", {**EXGINFO, "PREDICTUSDT": info("PERPETUAL", "COIN", "2026-10-25 08:15")}))
    assert client.backfill_asset_columns("usdt_perp", "1m") is None
    assert client.rows("SELECT underlying_type FROM data.usdt_perp_1m") == [("PREMARKET",)]


def backfill_table(client):
    """A 0.6.0 table with three days of a TradFi contract and of a crypto one, upgraded."""
    old_cols = ", ".join(DuckportClient.KLINE_COL_DEFS)
    client.con.execute(f"CREATE TABLE data.usdt_perp_1m ({old_cols})")
    days = ["2026-03-01 12:00", "2026-03-02 12:00", "2026-03-03 23:59"]
    old = pd.concat([candles("XAUUSDT", *days), candles("BTCUSDT", *days)])      # noqa: F841
    client.con.execute("INSERT INTO data.usdt_perp_1m SELECT * FROM old")
    init(client)
    client.set_asset_classifier("usdt_perp", AssetClassifier("usdt_perp", EXGINFO))


def updates(client):
    return [tx for tx in client.transactions if tx[0].startswith("UPDATE")]


def test_backfill_is_one_transaction_per_day():
    """One UPDATE over a whole table filled the temp directory of a small machine."""
    client = LocalClient()
    backfill_table(client)
    assert client.backfill_asset_columns("usdt_perp", "1m") == 3
    batches = updates(client)
    assert len(batches) == 3
    for day, (update, progress) in zip(("01", "02", "03"), batches):
        assert f"t.open_time >= TIMESTAMP '2026-03-{day} 00:00:00' AND t.open_time < " in update
        assert "asset_columns_backfill_upto:usdt_perp_1m" in progress
    assert client.rows("SELECT symbol, count(*) FROM data.usdt_perp_1m "
                       "WHERE contract_type = 'TRADIFI_PERPETUAL' GROUP BY 1") == [("XAUUSDT", 3)]
    assert client.rows("SELECT key FROM data.config_dict WHERE key LIKE 'asset_columns%'") == [
        ("asset_columns_backfilled:usdt_perp_1m",)]


def test_backfill_goes_on_where_an_interrupted_run_stopped():
    client = LocalClient()
    backfill_table(client)
    real = client.execute_transaction

    def fails_on_the_third_day(statements):
        if "TIMESTAMP '2026-03-03 00:00:00' AND t.open_time <" in statements[0]:
            raise RuntimeError("Out of Memory Error")
        return real(statements)
    client.execute_transaction = fails_on_the_third_day
    with pytest.raises(RuntimeError, match="Out of Memory"):
        client.backfill_asset_columns("usdt_perp", "1m")
    assert client.rows("SELECT value FROM data.config_dict WHERE key LIKE 'asset_columns%'") == [
        ("2026-03-03 00:00:00",)]

    client.execute_transaction = real
    client.transactions.clear()
    assert client.backfill_asset_columns("usdt_perp", "1m") == 1
    assert len(updates(client)) == 1
    assert client.rows("SELECT count(*) FROM data.usdt_perp_1m WHERE underlying_type = 'COMMODITY'") == [(3,)]
    assert client.backfill_asset_columns("usdt_perp", "1m") is None


def test_columns_are_added_one_at_a_time_with_a_checkpoint():
    client = LocalClient()
    calls = []
    real = client.execute
    client.execute = lambda sql: calls.append(sql.split(" DEFAULT")[0]) or real(sql)
    old_cols = ", ".join(DuckportClient.KLINE_COL_DEFS)
    client.con.execute(f"CREATE TABLE data.usdt_perp_1m ({old_cols})")
    init(client)
    assert calls == ["ALTER TABLE data.usdt_perp_1m ADD COLUMN IF NOT EXISTS contract_type VARCHAR", "CHECKPOINT",
                     "ALTER TABLE data.usdt_perp_1m ADD COLUMN IF NOT EXISTS underlying_type VARCHAR", "CHECKPOINT"]
    calls.clear()
    # nothing to add any more, and a new table is created with the columns
    init(client, sources=("usdt_perp", "others_perp"))
    assert calls == []
    assert columns(client, "others_perp_1m")[-2:] == ["contract_type", "underlying_type"]


def test_bulk_write_stamps_and_spot_is_untouched():
    client = LocalClient()
    init(client)
    client.set_asset_classifier("usdt_perp", AssetClassifier("usdt_perp", EXGINFO))
    hist = pa.Table.from_pandas(candles("ANTUSDT", "2023-01-01", "2026-11-18"), preserve_index=False)
    client.bulk_write_kline(hist, "usdt_perp", "1m", "", sync_duck_time=False)
    client.bulk_write_kline(hist, "usdt_perp", "1m", "", sync_duck_time=False)      # idempotent
    assert client.rows("SELECT contract_type FROM data.usdt_perp_1m ORDER BY open_time") == [
        ("PERPETUAL",), ("TRADIFI_PERPETUAL",)]
    client.write_kline(candles("BTCUSDT", "2026-11-18"), "usdt_spot", "1m", datetime(2026, 11, 18, 0, 1))
    assert client.rows("SELECT count(*) FROM data.usdt_spot_1m") == [(1,)]


def test_perp_write_without_a_classifier_is_refused():
    client = LocalClient()
    init(client)
    with pytest.raises(RuntimeError, match="no asset classifier"):
        client.write_kline(candles("BTCUSDT", "2026-11-18"), "usdt_perp", "1m", datetime(2026, 11, 18, 0, 1))
    with pytest.raises(RuntimeError, match="no asset classifier"):
        client.backfill_asset_columns("usdt_perp", "1m")
