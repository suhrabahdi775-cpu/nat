"""
CryptoOracle Sentiment Data Fetcher for NautilusTrader

Fetches market sentiment indicators from CryptoOracle API.
"""

import os
import time

import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class SentimentDataFetcher:
    """
    Fetches BTC market sentiment data from CryptoOracle API.

    Provides positive/negative sentiment ratios and net sentiment scores.

    Sentiment is OPTIONAL context (the AI prompt explicitly handles it being
    unavailable). This fetcher is therefore built to fail fast and quietly:
    a short timeout, and negative-cache backoff so a slow/unreachable
    endpoint does not block the analysis cycle every 15 minutes.
    """

    API_URL = "https://service.cryptoracle.network/openapi/v2.1/endpoint/klines"

    @property
    def API_KEY(self) -> str:
        # Read from environment at call time (after load_dotenv), never
        # hardcode API keys in the repo
        return os.getenv("CRYPTORACLE_API_KEY", "")

    def __init__(self, lookback_hours: int = 4, timeframe: str = "15m",
                 timeout: float = 5.0, logger=None):
        """
        Initialize sentiment data fetcher.

        Parameters
        ----------
        lookback_hours : int
            How many hours of historical data to fetch (default: 4)
        timeframe : str
            Time interval for data (default: "15m")
        timeout : float
            HTTP timeout in seconds (default: 5 - fail fast; sentiment is
            optional and must not stall the trading loop)
        logger : optional
            NautilusTrader logger (self.log). Falls back to print().
        """
        self.lookback_hours = lookback_hours
        self.timeframe = timeframe
        self.timeout = timeout
        self._log = logger
        # Negative-cache backoff: skip calls for a while after failures so a
        # dead endpoint doesn't cost `timeout` seconds every single cycle.
        self._consecutive_failures = 0
        self._last_attempt = 0.0
        self._backoff_base = 900.0  # 15 min per failure, capped at 1h

    def _warn(self, msg: str):
        if self._log:
            self._log.warning(msg)
        else:
            print(msg)

    def fetch(self, token: str = "BTC") -> Optional[Dict[str, Any]]:
        """
        Fetch sentiment data for specified token.

        Parameters
        ----------
        token : str
            Token symbol (default: "BTC")

        Returns
        -------
        Dict or None
            Sentiment data with structure:
            {
                'positive_ratio': float,
                'negative_ratio': float,
                'net_sentiment': float,
                'data_time': str,
                'data_delay_minutes': int
            }
        """
        if not self.API_KEY:
            return None  # sentiment simply disabled; the AI handles its absence

        # Negative-cache backoff: after failures, wait before retrying so a
        # slow/unreachable endpoint doesn't block the loop each cycle.
        now = time.time()
        if self._consecutive_failures > 0:
            backoff = min(self._backoff_base * self._consecutive_failures, 3600)
            if (now - self._last_attempt) < backoff:
                return None
        self._last_attempt = now

        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.lookback_hours)

            # Build request
            request_body = {
                "apiKey": self.API_KEY,
                "endpoints": ["CO-A-02-01", "CO-A-02-02"],  # Positive and Negative ratios
                "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "timeType": self.timeframe,
                "token": [token]
            }

            headers = {
                "Content-Type": "application/json",
                "X-API-KEY": self.API_KEY
            }

            # Make request with a short timeout to prevent stalling the loop
            response = requests.post(
                self.API_URL, json=request_body, headers=headers, timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("code") == 200 and data.get("data"):
                    self._consecutive_failures = 0  # success resets backoff
                    return self._parse_sentiment_data(data)

            self._consecutive_failures += 1
            self._warn(
                f"⚠️ CryptoOracle unexpected response {response.status_code} "
                f"- continuing without sentiment (backing off)"
            )
            return None

        except Exception as e:
            self._consecutive_failures += 1
            # Only log the first couple of failures to avoid spamming; after
            # that the backoff mostly suppresses attempts anyway.
            if self._consecutive_failures <= 2:
                self._warn(
                    f"⚠️ Sentiment fetch failed ({type(e).__name__}) - continuing "
                    f"without sentiment. Backing off {int(self._backoff_base/60)}min×failures."
                )
            return None

    def _parse_sentiment_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse sentiment data from API response."""
        try:
            time_periods = data["data"][0]["timePeriods"]

            # Find first time period with valid data
            for period in time_periods:
                period_data = period.get("data", [])

                sentiment = {}
                valid_data_found = False

                for item in period_data:
                    endpoint = item.get("endpoint")
                    value = item.get("value", "").strip()

                    if value:  # Only process non-empty values
                        try:
                            if endpoint in ["CO-A-02-01", "CO-A-02-02"]:
                                sentiment[endpoint] = float(value)
                                valid_data_found = True
                        except (ValueError, TypeError):
                            continue

                # If found valid data with both positive and negative
                if valid_data_found and "CO-A-02-01" in sentiment and "CO-A-02-02" in sentiment:
                    positive = sentiment['CO-A-02-01']
                    negative = sentiment['CO-A-02-02']
                    net_sentiment = positive - negative

                    # Calculate data delay
                    data_delay = int(
                        (datetime.now() - datetime.strptime(
                            period['startTime'], '%Y-%m-%d %H:%M:%S'
                        )).total_seconds() // 60
                    )

                    print(f"✅ Using sentiment data from: {period['startTime']} (delay: {data_delay} minutes)")

                    return {
                        'positive_ratio': positive,
                        'negative_ratio': negative,
                        'net_sentiment': net_sentiment,
                        'data_time': period['startTime'],
                        'data_delay_minutes': data_delay
                    }

            print("❌ All time periods have empty data")
            return None

        except Exception as e:
            print(f"❌ Sentiment data parsing failed: {e}")
            return None

    def format_for_display(self, sentiment_data: Optional[Dict[str, Any]]) -> str:
        """
        Format sentiment data for logging/display.

        Parameters
        ----------
        sentiment_data : Dict or None
            Sentiment data from fetch()

        Returns
        -------
        str
            Formatted sentiment string
        """
        if not sentiment_data:
            return "Market Sentiment: Data unavailable"

        sign = '+' if sentiment_data['net_sentiment'] >= 0 else ''
        return (
            f"Market Sentiment: "
            f"Bullish {sentiment_data['positive_ratio']:.1%} | "
            f"Bearish {sentiment_data['negative_ratio']:.1%} | "
            f"Net {sign}{sentiment_data['net_sentiment']:.3f} | "
            f"Delay {sentiment_data['data_delay_minutes']}min"
        )
