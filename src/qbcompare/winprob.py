"""Per-play win probability from ESPN.

The bulk feed ships `wpa` / `wp_before` / `wp_after` columns, but they are
corrupted: routine plays (a punt, a four-yard run) swing home win probability by
more than 30 points on 5.8% of 2025 snaps and 12.7% of 2026 snaps, and timeouts
carry non-zero EPA. Its EPA column is sound — play types order correctly and the
league mean is sane — so EPA is taken from the bulk feed and win probability is
taken from ESPN instead.

Using one win-probability model everywhere matters more than which one: mixing
two models across a comparison makes the numbers incommensurable.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={game_id}"
CACHE = Path(__file__).resolve().parents[2] / "data" / "espn"


def _fetch(url: str, attempts: int = 4) -> dict | None:
    """GET via curl, which honours this sandbox's proxy where urllib does not."""
    for attempt in range(attempts):
        result = subprocess.run(
            ["curl", "-sS", "--max-time", "45", "-H", "User-Agent: Mozilla/5.0", url],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip().startswith("{"):
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        time.sleep(2 * 2**attempt)
    return None


def game_win_probability(game_id: int, refresh: bool = False) -> pd.DataFrame:
    """Return one row per play: its id, the home win probability, and the swing."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{game_id}.json"
    if cached.exists() and not refresh:
        payload = json.loads(cached.read_text())
    else:
        payload = _fetch(SUMMARY.format(game_id=game_id))
        if payload is None:
            return pd.DataFrame()
        cached.write_text(json.dumps(payload))

    probabilities = {
        entry["playId"]: entry["homeWinPercentage"]
        for entry in payload.get("winprobability", [])
        if "playId" in entry and entry.get("homeWinPercentage") is not None
    }
    if not probabilities:
        return pd.DataFrame()

    rows = []
    for drive in payload.get("drives", {}).get("previous", []):
        for play in drive.get("plays", []):
            play_id = play.get("id")
            if play_id not in probabilities:
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "play_id": play_id,
                    "sequence": int(play.get("sequenceNumber", 0) or 0),
                    "period": play.get("period", {}).get("number"),
                    "home_wp": probabilities[play_id],
                    "text": play.get("text", ""),
                }
            )
    frame = pd.DataFrame(rows).sort_values("sequence").reset_index(drop=True)
    if frame.empty:
        return frame

    home = payload["header"]["competitions"][0]["competitors"]
    frame.attrs["home_team"] = next(
        c["team"]["displayName"] for c in home if c["homeAway"] == "home"
    )
    # The swing a play produced, from the home team's point of view.
    frame["home_wp_delta"] = frame["home_wp"].diff().fillna(0)
    return frame


def load_many(game_ids: list[int], pause: float = 0.4) -> pd.DataFrame:
    """Fetch win probability for several games, caching each one."""
    frames = []
    for game_id in game_ids:
        frame = game_win_probability(game_id)
        if not frame.empty:
            frame["home_team"] = frame.attrs.get("home_team")
            frames.append(frame)
            time.sleep(pause)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
