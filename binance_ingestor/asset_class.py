"""Asset class of each perpetual kline row: ``contract_type`` and ``underlying_type``.

The USD-M kline tables (``usdt_perp``, ``others_perp``) carry Binance's two
classification fields on every row, so a query can filter crypto / TradFi / equity ...
without ``data.exginfo``. ``exginfo`` only lists what trades now: it has no delisted
symbols and no history, and a symbol name is reused (ANTUSDT was a crypto perpetual
until 2024-04 and is a TradFi pre-IPO perpetual since 2026-11).

Where a row's values come from:

1. ``KNOWN_PHASES``: periods whose values differ from what exchangeInfo says today.
   These are the past pre-market periods (taken from the Binance announcements) and
   delisted contracts that were not crypto.
2. The symbol's current exchangeInfo entry, for rows from its ``onboardDate`` on.
   Binance resets ``onboardDate`` when a name is listed again, so rows of an earlier
   listing fall through to 3.
3. ``DEFAULT_CLASS`` (crypto): earlier listings and delisted symbols.

Rows written live get the values of that moment. A later change (a pre-market
contract becoming a standard one) therefore shows in the rows written after it;
for loadhist to reproduce the earlier period, add it to ``KNOWN_PHASES``.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from binance_ingestor.markets import MARKETS, VENUE_UM

CONTRACT_TYPE_COL = "contract_type"
UNDERLYING_TYPE_COL = "underlying_type"
DEFAULT_CLASS = ("PERPETUAL", "COIN")
ASSET_COL_DEFS = [
    f"{CONTRACT_TYPE_COL} VARCHAR DEFAULT '{DEFAULT_CLASS[0]}'",
    f"{UNDERLYING_TYPE_COL} VARCHAR DEFAULT '{DEFAULT_CLASS[1]}'",
]

# The first candle of a listing can open a minute before onboardDate.
ONBOARD_TOLERANCE = timedelta(hours=1)

_MIN = datetime(1970, 1, 1)
_MAX = datetime(2200, 1, 1)      # inside datetime64[ns]

Segment = Tuple[datetime, datetime, str, Optional[str]]     # [lo, hi), contract_type, underlying_type


def has_asset_columns(market: str) -> bool:
    return market in MARKETS and MARKETS[market].venue == VENUE_UM


def _t(text: Optional[str]) -> Optional[datetime]:
    return datetime.strptime(text, "%Y-%m-%d %H:%M") if text else None


def _phases(contract_type: str, underlying_type: str, rows) -> Dict[str, List[Segment]]:
    return {symbol: [(_t(start), _t(end) or _MAX, contract_type, underlying_type)]
            for symbol, start, end in rows}


# market -> symbol -> [(start, end, contract_type, underlying_type)], times in UTC.
# start = listing time, end = the conversion time Binance announced (None: until delisting).
KNOWN_PHASES: Dict[str, Dict[str, List[Segment]]] = {
    "usdt_perp": {
        # crypto pre-market perpetuals, converted to standard perpetuals
        **_phases("PERPETUAL", "PREMARKET", [
            ("BABYUSDT", "2025-04-05 13:30", "2025-04-10 11:10"),
            ("INITUSDT", "2025-04-16 06:30", "2025-04-24 11:00"),
            ("SXTUSDT", "2025-05-02 08:30", "2025-05-08 13:00"),
            ("NEWTUSDT", "2025-06-19 14:30", "2025-06-24 14:00"),
            ("PUMPUSDT", "2025-07-10 07:30", "2025-07-14 17:20"),
            ("XPLUSDT", "2025-08-22 09:30", "2025-09-25 13:00"),
            ("WLFIUSDT", "2025-08-23 08:30", "2025-09-01 13:00"),
            ("SOMIUSDT", "2025-08-25 09:30", "2025-09-02 14:30"),
            ("LINEAUSDT", "2025-09-01 08:30", "2025-09-10 16:00"),
            ("0GUSDT", "2025-09-17 15:45", "2025-09-22 10:00"),
            ("MONUSDT", "2025-10-10 07:15", "2025-11-24 15:00"),
            ("YBUSDT", "2025-10-10 14:45", "2025-10-15 11:00"),
            ("METUSDT", "2025-10-11 12:30", "2025-10-23 14:00"),
            ("KITEUSDT", "2025-10-29 10:30", "2025-11-03 13:00"),
            ("CCUSDT", "2025-10-31 12:00", "2025-11-10 11:30"),
            ("STABLEUSDT", "2025-11-06 12:00", "2025-12-08 13:00"),
            ("SENTUSDT", "2025-11-14 12:45", "2026-01-22 13:00"),
            ("LITUSDT", "2025-12-23 17:30", "2026-01-15 15:15"),
            ("BREVUSDT", "2025-12-30 10:45", "2026-01-06 14:00"),
            ("ZAMAUSDT", "2026-01-09 09:00", "2026-02-02 13:00"),
            ("FOGOUSDT", "2026-01-10 14:00", "2026-01-15 14:00"),
            ("MEGAUSDT", "2026-01-30 12:15", "2026-04-30 11:00"),
            ("ESPUSDT", "2026-02-10 08:10", "2026-02-12 13:00"),
            ("AZTECUSDT", "2026-02-11 04:30", "2026-02-12 07:00"),
            ("OPNUSDT", "2026-02-21 15:00", "2026-03-05 13:00"),
            ("KATUSDT", "2026-03-02 05:00", "2026-03-18 13:00"),
            ("EDGEUSDT", "2026-03-19 14:00", "2026-03-31 12:30"),
            ("CHIPUSDT", "2026-04-16 07:15", "2026-04-21 13:30"),
        ]),
        # TradFi pre-IPO perpetuals: converted to standard TradFi perpetuals, or delisted
        **_phases("TRADIFI_PERPETUAL", "PREMARKET", [
            ("SPCXUSDT", "2026-05-21 04:45", "2026-06-12 16:15"),
            ("QNTXUSDT", "2026-05-29 08:15", "2026-06-04 18:00"),
            ("ANTHROPICUSDT", "2026-06-02 04:30", None),      # delisted 2026-11-17
        ]),
    },
}


def _overlay(segments: List[Segment], new: Segment) -> List[Segment]:
    """``segments`` with ``new`` laid on top; the result does not overlap."""
    nlo, nhi = new[0], new[1]
    out = []
    for lo, hi, ct, ut in segments:
        if lo < nlo:
            out.append((lo, min(hi, nlo), ct, ut))
        if hi > nhi:
            out.append((max(lo, nhi), hi, ct, ut))
    out.append(new)
    return sorted(s for s in out if s[0] < s[1])


class AssetClassifier:
    """Classifies one market's kline rows; fed with the market's exchangeInfo."""

    def __init__(self, market: str, exginfo: Optional[dict] = None):
        self.market = market
        self._current: Dict[str, Tuple[str, Optional[str], Optional[datetime]]] = {}
        self._cache: Dict[str, List[Segment]] = {}
        if exginfo:
            self.update(exginfo)

    def update(self, exginfo: dict) -> None:
        """Take a fresh exchangeInfo (``BinanceFetcher.get_exchange_info``).

        A symbol that left exchangeInfo keeps its last values: the candles of its final
        minutes are still written after it is gone.
        """
        for symbol, info in exginfo.items():
            contract_type = info.get("contract_type")
            if not contract_type:
                continue
            underlying = info.get("underlying_type")
            if not underlying and contract_type == DEFAULT_CLASS[0]:
                underlying = DEFAULT_CLASS[1]
            onboard = info.get("onboard_time")
            if onboard is not None:
                onboard = onboard.to_pydatetime() if hasattr(onboard, "to_pydatetime") else onboard
                onboard = onboard.replace(tzinfo=None)
            self._current[symbol] = (contract_type, underlying, onboard)
        self._cache.clear()

    def segments(self, symbol: str) -> List[Segment]:
        """The periods in which ``symbol`` is not ``DEFAULT_CLASS``, in time order."""
        cached = self._cache.get(symbol)
        if cached is not None:
            return cached
        segments: List[Segment] = []
        current = self._current.get(symbol)
        if current is not None:
            contract_type, underlying, onboard = current
            lo = onboard - ONBOARD_TOLERANCE if onboard else _MIN
            segments.append((lo, _MAX, contract_type, underlying))
        for start, end, contract_type, underlying in KNOWN_PHASES.get(self.market, {}).get(symbol, ()):
            segments = _overlay(segments, (start - ONBOARD_TOLERANCE, end, contract_type, underlying))
        segments = [s for s in segments if (s[2], s[3]) != DEFAULT_CLASS]
        self._cache[symbol] = segments
        return segments

    def all_segments(self) -> List[Tuple[str, datetime, datetime, str, Optional[str]]]:
        """``(symbol, lo, hi, contract_type, underlying_type)`` for every non-default period."""
        symbols = set(self._current) | set(KNOWN_PHASES.get(self.market, {}))
        return [(symbol, *segment) for symbol in sorted(symbols) for segment in self.segments(symbol)]

    def stamp(self, table: pa.Table) -> pa.Table:
        """``table`` (needs ``symbol`` and a naive-UTC ``open_time``) with the two columns set."""
        n = table.num_rows
        contract = np.full(n, DEFAULT_CLASS[0], dtype=object)
        underlying = np.full(n, DEFAULT_CLASS[1], dtype=object)
        if n:
            encoded = table["symbol"].combine_chunks().dictionary_encode()
            codes = encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False)
            times = table["open_time"].combine_chunks().to_numpy(zero_copy_only=False)
            for code, symbol in enumerate(encoded.dictionary.to_pylist()):
                segments = self.segments(symbol)
                if not segments:
                    continue
                rows = np.flatnonzero(codes == code)
                at = times[rows]
                for lo, hi, contract_type, underlying_type in segments:
                    hit = rows[(at >= np.datetime64(lo)) & (at < np.datetime64(hi))]
                    contract[hit] = contract_type
                    underlying[hit] = underlying_type
        for name, values in ((CONTRACT_TYPE_COL, contract), (UNDERLYING_TYPE_COL, underlying)):
            column = pa.array(values, type=pa.string())
            index = table.schema.get_field_index(name)
            table = table.append_column(name, column) if index < 0 else table.set_column(index, name, column)
        return table
