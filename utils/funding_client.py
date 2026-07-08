"""
Binance Futures Funding Rate Fetcher

Provides the current funding rate for a perpetual contract. Funding is a
direct carry cost/income for held positions and a useful crowding signal:
strongly positive funding means longs pay shorts (crowded long), and
vice versa.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests


class FundingRateFetcher:
    """
    Fetches the current funding rate from Binance Futures premium index.

    Results are cached briefly to avoid hammering the endpoint when
    analysis runs on every bar.
    """

    API_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"

    def __init__(self, symbol: str = "BTCUSDT", cache_seconds: int = 300):
        """
        Parameters
        ----------
        symbol : str
            Binance futures symbol (default: "BTCUSDT")
        cache_seconds : int
            How long to reuse the last successful fetch (default: 300)
        """
        self.symbol = symbol
        self.cache_seconds = cache_seconds
        self._cached: Optional[Dict[str, Any]] = None
        self._cached_at: float = 0.0
        self._last_attempt: float = 0.0
        self._consecutive_failures: int = 0

    def fetch(self) -> Optional[Dict[str, Any]]:
        """
        Fetch current funding data.

        Returns
        -------
        Dict or None
            {
                'funding_rate': float,        # e.g. 0.0001 = 0.01%
                'mark_price': float,
                'next_funding_time': str,     # ISO timestamp
            }
        """
        now = time.time()
        if self._cached is not None and (now - self._cached_at) < self.cache_seconds:
            return self._cached

        # Negative caching: don't hammer (and block on) an unreachable
        # endpoint - back off harder with each consecutive failure. Without
        # this, every analysis cycle pays the full HTTP timeout when the
        # endpoint is unreachable (geo-block, backtest, outage).
        if self._consecutive_failures > 0:
            backoff = min(self.cache_seconds * self._consecutive_failures, 3600)
            if (now - self._last_attempt) < backoff:
                return self._cached

        self._last_attempt = now

        try:
            response = requests.get(
                self.API_URL, params={"symbol": self.symbol}, timeout=5
            )
            response.raise_for_status()
            data = response.json()

            next_funding = ""
            next_funding_ms = int(data.get("nextFundingTime", 0))
            if next_funding_ms > 0:
                next_funding = datetime.fromtimestamp(
                    next_funding_ms / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC")

            result = {
                "funding_rate": float(data.get("lastFundingRate", 0.0)),
                "mark_price": float(data.get("markPrice", 0.0)),
                "next_funding_time": next_funding,
            }

            self._cached = result
            self._cached_at = now
            self._consecutive_failures = 0
            return result

        except Exception:
            # Stale cache is better than nothing
            self._consecutive_failures += 1
            return self._cached
