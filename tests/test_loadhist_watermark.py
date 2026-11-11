"""loadhist watermark: duck_time stops where every live symbol has rows.

Binance publishes a day's archives symbol by symbol. A loadhist run in between used to
set duck_time to the newest row of any symbol, so the symbols still waiting for their
archive kept a hole that neither the REST backfill (it starts at duck_time) nor a later
loadhist (it reloads one day before duck_time) filled.

Run with ``PYTHONPATH=. pytest tests``.
"""
from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest

from binance_ingestor import loadhist
from binance_ingestor.loadhist import complete_through, save_to_duckport

COLS = ["open_time", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "trade_num",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "avg_price"]


def candles(symbol, start, end):
    """1m candles for ``symbol`` from ``start`` to ``end`` inclusive."""
    times = pd.date_range(start, end, freq="1min")
    df = pd.DataFrame({c: 1.0 for c in COLS if c not in ("open_time", "symbol", "trade_num")}, index=range(len(times)))
    df["open_time"], df["symbol"], df["trade_num"] = times, symbol, 1
    return df[COLS]


def write(tmp_path, month, frames, market="usdt_perp"):
    path = tmp_path / f"{market}_{month}_1M.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def test_all_symbols_have_the_newest_day(tmp_path, con):
    f = write(tmp_path, "2026-11", [candles(s, "2026-11-08", "2026-11-10 23:59") for s in ("AAA", "BBB")])
    complete, newest, behind = complete_through(con, [f])
    assert complete == newest == datetime(2026, 11, 10, 23, 59)
    assert behind == []


def test_symbol_whose_newest_archive_is_not_published(tmp_path, con):
    f = write(tmp_path, "2026-11", [
        candles("AAA", "2026-11-08", "2026-11-10 23:59"),
        candles("LATE", "2026-11-08", "2026-11-09 23:59"),
    ])
    complete, newest, behind = complete_through(con, [f])
    assert newest == datetime(2026, 11, 10, 23, 59)
    assert complete == datetime(2026, 11, 9, 23, 59)
    assert behind == ["LATE"]


def test_symbol_delisted_earlier_does_not_hold_the_watermark(tmp_path, con):
    f = write(tmp_path, "2026-11", [
        candles("AAA", "2026-11-01", "2026-11-10 23:59"),
        candles("GONE", "2026-11-01", "2026-11-05 12:00"),
    ])
    complete, newest, behind = complete_through(con, [f])
    assert complete == newest == datetime(2026, 11, 10, 23, 59)
    assert behind == []


def test_symbol_that_stopped_the_day_before_holds_it_at_its_last_row(tmp_path, con):
    """Delisted yesterday or archive late: indistinguishable here, REST settles it."""
    f = write(tmp_path, "2026-11", [
        candles("AAA", "2026-11-08", "2026-11-10 23:59"),
        candles("STOP", "2026-11-08", "2026-11-09 08:00"),
    ])
    complete, _, behind = complete_through(con, [f])
    assert complete == datetime(2026, 11, 9, 8, 0)
    assert behind == ["STOP"]


def test_symbol_listed_on_the_newest_day_is_ignored(tmp_path, con):
    f = write(tmp_path, "2026-11", [
        candles("AAA", "2026-11-08", "2026-11-10 23:59"),
        candles("NEW", "2026-11-10 10:00", "2026-11-10 20:00"),
    ])
    complete, newest, behind = complete_through(con, [f])
    assert complete == newest
    assert behind == []


def test_last_minutes_trimmed_by_the_clean_step(tmp_path, con):
    """Zero-volume candles after a symbol's last trade are trimmed; the watermark follows."""
    f = write(tmp_path, "2026-11", [
        candles("AAA", "2026-11-08", "2026-11-10 23:59"),
        candles("THIN", "2026-11-08", "2026-11-10 23:41"),
    ])
    complete, _, behind = complete_through(con, [f])
    assert complete == datetime(2026, 11, 10, 23, 41)
    assert behind == []


def test_previous_day_is_in_last_months_file(tmp_path, con):
    files = [
        write(tmp_path, "2026-09", [candles("AAA", "2026-09-29", "2026-09-30 23:59")]),
        write(tmp_path, "2026-10", [candles(s, "2026-10-30", "2026-10-31 23:59") for s in ("AAA", "LATE")]),
        write(tmp_path, "2026-11", [candles("AAA", "2026-11-01", "2026-11-01 23:59")]),
    ]
    complete, newest, behind = complete_through(con, files)
    assert newest == datetime(2026, 11, 1, 23, 59)
    assert complete == datetime(2026, 10, 31, 23, 59)
    assert behind == ["LATE"]


def test_never_more_than_two_days_behind(tmp_path, con):
    f = write(tmp_path, "2026-11", [
        candles("AAA", "2026-11-01", "2026-11-10 23:59"),
        candles("B", "2026-11-01", "2026-11-09 00:00"),
        candles("C", "2026-11-01", "2026-11-08 23:59"),
    ])
    complete, newest, _ = complete_through(con, [f])
    assert complete == datetime(2026, 11, 9, 0, 0)
    assert newest - complete < timedelta(days=2)


def test_no_files(con):
    assert complete_through(con, []) == (None, None, [])


# ── save_to_duckport ──────────────────────────────────────────────────

class FakeClient:
    def __init__(self, duck_time=None):
        self.duck_time = duck_time
        self.rows = 0
        self.replaced = []

    def read_duck_time(self, market):
        return self.duck_time

    def bulk_write_kline(self, arrow, market, interval, duck_time_str, chunk_size=None, sync_duck_time=True):
        self.rows += len(arrow)

    def replace_duck_time(self, market, value):
        self.replaced.append(value)


@pytest.fixture
def pqt(tmp_path, monkeypatch):
    d = tmp_path / "usdt_perp_1m"
    d.mkdir()
    monkeypatch.setattr(loadhist, "PARQUET_DIR", str(tmp_path))
    monkeypatch.setattr(loadhist, "KLINE_INTERVAL", "1m")
    return d


def test_first_load_stops_before_the_unpublished_day(pqt):
    write(pqt, "2026-11", [candles("AAA", "2026-11-08", "2026-11-10 23:59"),
                           candles("LATE", "2026-11-08", "2026-11-09 23:59")])
    client = FakeClient()
    save_to_duckport(client, ["usdt_perp"])
    assert client.rows == 3 * 1440 + 2 * 1440            # every row is still loaded
    assert client.replaced == ["2026-11-09 23:59:00"]


def test_first_load_with_complete_archives_is_unchanged(pqt):
    write(pqt, "2026-11", [candles(s, "2026-11-08", "2026-11-10 23:59") for s in ("AAA", "BBB")])
    client = FakeClient()
    save_to_duckport(client, ["usdt_perp"])
    assert client.replaced == ["2026-11-10 23:59:00"]


def test_duck_time_is_never_moved_back(pqt):
    """The ingestor has run past the archives: its duck_time stays."""
    write(pqt, "2026-11", [candles("AAA", "2026-11-08", "2026-11-10 23:59"),
                           candles("LATE", "2026-11-08", "2026-11-09 23:59")])
    client = FakeClient(duck_time=pd.Timestamp("2026-11-10 12:00", tz="UTC"))
    save_to_duckport(client, ["usdt_perp"])
    assert client.replaced == []
    assert client.rows > 0
