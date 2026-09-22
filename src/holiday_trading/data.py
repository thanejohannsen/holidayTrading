"""Download and cache daily index data from Yahoo Finance."""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={start}&period2={end}&interval=1d"
)
# Yahoo rejects the default urllib agent.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _fetch_json(url: str, attempts: int = 4) -> dict:
    """GET a URL, retrying with exponential backoff on transient proxy resets."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except Exception as error:  # noqa: BLE001 - network layer is opaque here
            last_error = error
            time.sleep(5 * 2**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def download(symbol: str, start: dt.date = dt.date(1927, 1, 1)) -> pd.DataFrame:
    """Return a daily OHLCV frame for ``symbol`` indexed by session date."""
    start_epoch = int(
        dt.datetime.combine(start, dt.time()).replace(tzinfo=dt.timezone.utc).timestamp()
    )
    end_epoch = int(time.time()) + 86400
    payload = _fetch_json(
        CHART_URL.format(
            symbol=urllib.parse.quote(symbol), start=start_epoch, end=end_epoch
        )
    )
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        },
        index=pd.to_datetime(result["timestamp"], unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None),
    )
    frame.index.name = "date"
    return frame[frame["close"].notna()]


def load(symbol: str = "^GSPC", refresh: bool = False) -> pd.DataFrame:
    """Load ``symbol`` from the local cache, downloading it if needed."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{symbol.lstrip('^').lower()}_daily.csv"
    if cache.exists() and not refresh:
        frame = pd.read_csv(cache, index_col="date", parse_dates=["date"])
    else:
        frame = download(symbol)
        # The session in progress is incomplete; keep only settled closes.
        today = pd.Timestamp(dt.date.today())
        frame = frame[frame.index < today]
        frame.to_csv(cache)
    frame["ret"] = frame["close"].pct_change()
    return frame
