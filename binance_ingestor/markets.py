"""Kline markets: what a ``DATA_SOURCES`` entry means.

A market name maps to a venue (which REST / WebSocket / archive endpoints serve it) and
to a quote group. ``usdt_perp`` and ``others_perp`` share the USD-M futures venue and
split its perpetuals by quote asset: USDT in one table, every other quote (USDC, USD1,
U, BTC, ...) together in the other. New quote assets land in ``others_perp`` without a
config change.

This module has no imports from the rest of the package, so ``config`` can use it.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

VENUE_SPOT = 'spot'
VENUE_UM = 'um'      # USD-M futures (fapi)
VENUE_CM = 'cm'      # COIN-M futures (dapi); not a kline market yet


@dataclass(frozen=True)
class Market:
    name: str
    venue: str
    usdt_quoted: bool


MARKETS = {
    m.name: m for m in (
        Market('usdt_perp', VENUE_UM, True),
        Market('usdt_spot', VENUE_SPOT, True),
        Market('others_perp', VENUE_UM, False),
    )
}

# Every market the ingestor can write, in processing order.
KLINE_MARKETS: Tuple[str, ...] = tuple(MARKETS)

# Names used by the API layer that are not kline markets.
_EXTRA_VENUES = {'coin_perp': VENUE_CM}


def venue_of(market: str) -> str:
    if market in MARKETS:
        return MARKETS[market].venue
    if market in _EXTRA_VENUES:
        return _EXTRA_VENUES[market]
    raise ValueError(f'Market {market} not supported')


def parse_data_sources(raw: str) -> Tuple[str, ...]:
    """Validate ``DATA_SOURCES`` and return the enabled markets in processing order.

    Raises ValueError for a name that is not a market, and for a mix of USDT and
    non-USDT markets: one instance serves one of the two groups, so that a new market's
    loadhist or stale watermark never stops the other group's ingestion.
    """
    names = [s.strip() for s in (raw or '').split(',') if s.strip()]
    unknown = sorted(set(n for n in names if n not in MARKETS))
    if unknown:
        raise ValueError(
            f"DATA_SOURCES 含未知市场 {', '.join(unknown)}；可用: {', '.join(KLINE_MARKETS)}"
        )
    usdt = sorted(set(n for n in names if MARKETS[n].usdt_quoted))
    others = sorted(set(n for n in names if not MARKETS[n].usdt_quoted))
    if usdt and others:
        raise ValueError(
            f"DATA_SOURCES 不能同时包含 USDT 市场（{', '.join(usdt)}）和非 USDT 市场"
            f"（{', '.join(others)}）；请用 duckport install binance-ingestor 另装一个实例"
        )
    return tuple(m for m in KLINE_MARKETS if m in names)


def is_other_quote_um_symbol(symbol: str) -> bool:
    """Archive-index rule for ``others_perp``: a USD-M perpetual not quoted in USDT.

    The archive index only has names. USDT perpetuals are the names ending in ``USDT``
    (the rule loadhist has always used for ``usdt_perp``); delivery contracts carry a
    ``_YYMMDD`` suffix and settled leftovers end in ``SETTLED``. Everything else is
    another quote asset.
    """
    return not (symbol.endswith('USDT') or '_' in symbol or symbol.endswith('SETTLED'))


def select_archive_symbols(market: str, symbols: Iterable[str]) -> set:
    """Pick ``market``'s symbols out of an archive listing of its venue."""
    if MARKETS[market].usdt_quoted:
        return {s for s in symbols if s.endswith('USDT')}
    return {s for s in symbols if is_other_quote_um_symbol(s)}
