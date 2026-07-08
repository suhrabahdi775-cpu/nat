"""
Cached DeepSeek Analyzer - measure the AI's edge with zero money at risk.

Wraps DeepSeekAnalyzer with a persistent on-disk cache keyed by bar
timestamp. First backtest run calls the real API and records every signal;
subsequent runs replay from cache instantly and deterministically.

This answers "does DeepSeek beat the rule-based baseline?" using real AI
signals against real historical data, before risking a single dollar.

Notes
-----
- Cache key is the bar timestamp only. The prompt also contains position
  state, which can differ between runs with different risk configs - cached
  signals reflect the position context of the run that CREATED them. For
  signal-quality measurement this is acceptable and keeps replays
  deterministic.
- max_api_calls caps spend per run: once exhausted, uncached bars get a
  HOLD fallback (marked "budget_exhausted") instead of an API call.
"""

import json
import os
from typing import Any, Dict, Optional

from utils.deepseek_client import DeepSeekAnalyzer


class CachedDeepSeekAnalyzer:
    """Drop-in analyzer: real DeepSeek signals with on-disk caching."""

    def __init__(
        self,
        api_key: str,
        cache_path: str,
        model: str = "deepseek-v4-pro",
        temperature: float = 0.1,
        max_retries: int = 2,
        max_api_calls: int = 0,
        nautilus_logger=None,
    ):
        self._inner = DeepSeekAnalyzer(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            nautilus_logger=nautilus_logger,
        )
        self._log = nautilus_logger
        self.cache_path = cache_path
        self.max_api_calls = max_api_calls
        self.api_calls = 0
        self.cache_hits = 0
        self.budget_exhausted_bars = 0

        self.cache: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    @property
    def signal_history(self):
        return self._inner.signal_history

    def analyze(
        self,
        price_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None,
        current_position: Optional[Dict[str, Any]] = None,
        trade_history: Optional[list] = None,
        funding_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        key = str(price_data.get("timestamp", ""))

        # Cache hit: replay, keeping the inner signal history coherent so
        # any subsequent API calls see the same "previous signal" context.
        if key in self.cache:
            self.cache_hits += 1
            signal = dict(self.cache[key])
            self._inner.signal_history.append(signal)
            if len(self._inner.signal_history) > 30:
                self._inner.signal_history.pop(0)
            return signal

        # Budget guard
        if self.max_api_calls and self.api_calls >= self.max_api_calls:
            self.budget_exhausted_bars += 1
            return {
                "signal": "HOLD",
                "confidence": "LOW",
                "reason": "API budget exhausted (cached run cap reached)",
                "budget_exhausted": True,
            }

        # Real API call
        self.api_calls += 1
        signal = self._inner.analyze(
            price_data=price_data,
            technical_data=technical_data,
            sentiment_data=sentiment_data,
            current_position=current_position,
            trade_history=trade_history,
            funding_data=funding_data,
        )

        # Don't poison the cache with transient failures
        if not signal.get("is_fallback"):
            self.cache[key] = signal
            self._save()

        return signal

    def _save(self):
        try:
            directory = os.path.dirname(self.cache_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            if self._log:
                self._log.warning(f"Failed to persist signal cache: {e}")

    def stats(self) -> str:
        return (
            f"CachedDeepSeekAnalyzer: {self.api_calls} API calls, "
            f"{self.cache_hits} cache hits, {len(self.cache)} cached signals, "
            f"{self.budget_exhausted_bars} budget-exhausted bars"
        )
