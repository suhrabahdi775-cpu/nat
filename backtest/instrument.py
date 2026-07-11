"""
Backtest instrument definition with REALISTIC Binance fees.

nautilus_trader's TestInstrumentProvider.btcusdt_perp_binance() models the
taker fee at 0.018% - but real Binance USDT-M VIP0 taker is 0.05% (maker
0.02%). This strategy uses MARKET orders (taker) both ways, so the provider
under-models round-trip costs ~2.8x (0.036% vs 0.10% of notional).

That difference is decisive for tight-target profiles: at a 0.30% stop and
1R target, breakeven win rate is 56% under the test fees but 66.7% under
real fees. Any validation done with the test instrument silently favors
scalping profiles that lose money live. Always use THIS instrument for
backtests and sweeps.
"""

from decimal import Decimal

from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity

# Binance USDT-M futures VIP0, no BNB discount (conservative-realistic)
REAL_MAKER_FEE = Decimal("0.000200")  # 0.02%
REAL_TAKER_FEE = Decimal("0.000500")  # 0.05%


def btcusdt_perp_real_fees() -> CryptoPerpetual:
    """BTCUSDT-PERP.BINANCE with real VIP0 fees."""
    return CryptoPerpetual(
        instrument_id=InstrumentId(
            symbol=Symbol("BTCUSDT-PERP"),
            venue=Venue("BINANCE"),
        ),
        raw_symbol=Symbol("BTCUSDT"),
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=1,
        price_increment=Price.from_str("0.1"),
        size_precision=3,
        size_increment=Quantity.from_str("0.001"),
        max_quantity=Quantity.from_str("1000.000"),
        min_quantity=Quantity.from_str("0.001"),
        max_notional=None,
        min_notional=Money(10.00, USDT),
        max_price=Price.from_str("809484.0"),
        min_price=Price.from_str("261.1"),
        margin_init=Decimal("0.0500"),
        margin_maint=Decimal("0.0250"),
        maker_fee=REAL_MAKER_FEE,
        taker_fee=REAL_TAKER_FEE,
        ts_event=1646199312128000000,
        ts_init=1646199342953849862,
    )
