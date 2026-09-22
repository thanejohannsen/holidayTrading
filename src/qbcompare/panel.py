"""Assemble the analysis panel: one row per quarterback play, with context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import ingest

DATA = Path(__file__).resolve().parents[2] / "data"

COLUMNS = [
    "game_id", "week", "pos_team", "def_pos_team", "play_type", "play_text",
    "down", "distance", "yards_to_goal", "yards_gained", "EPA", "success", "ppa",
    "period", "clock_minutes", "clock_seconds", "pos_score_diff_start",
    "pos_team_score", "def_pos_team_score",
    "home_team", "away_team", "home_team_pregame_elo", "away_team_pregame_elo",
    "home_team_division", "away_team_division", "spread", "penalty_no_play",
    "completion", "int", "sack", "rz_play", "middle_8", "id_play",
]

# Plays that put the quarterback's own performance on the field. Kneels and
# spikes are excluded: they are clock management, not attempts to gain.
SCRIMMAGE = {"pass", "sack", "rush"}


def load_season(year: int) -> pd.DataFrame:
    return pq.read_table(DATA / f"pbp_{year}.parquet", columns=COLUMNS).to_pandas()


def _attach_win_probability(season: pd.DataFrame, model) -> pd.DataFrame:
    """Score every play in the season, then difference within each game."""
    from . import wpmodel

    frame = wpmodel._prepare(season)
    usable = frame[wpmodel.FEATURES].notna().all(axis=1)
    frame["wp_pos"] = np.nan
    frame.loc[usable, "wp_pos"] = model.predict_proba(frame.loc[usable, wpmodel.FEATURES])[:, 1]
    # Put every play on the home team's scale so a change of possession does not
    # flip the sign when consecutive plays are differenced.
    frame["wp_home"] = np.where(frame["is_home"] == 1, frame["wp_pos"], 1 - frame["wp_pos"])
    frame = frame.sort_values(
        ["game_id", "period", "seconds_left"], ascending=[True, True, False]
    )
    frame["wp_home_delta"] = frame.groupby("game_id")["wp_home"].shift(-1) - frame["wp_home"]
    frame["wpa_own"] = np.where(
        frame["is_home"] == 1, frame["wp_home_delta"], -frame["wp_home_delta"]
    )
    return frame


def build(years=(2025, 2026), model=None) -> pd.DataFrame:
    """Return every Manning and Stockton scrimmage play across ``years``.

    When a win-probability ``model`` is supplied, win probability is computed on
    the whole season frame *before* the quarterback's plays are pulled out. That
    ordering matters: the feed's ``id_play`` is stored as a float and loses
    precision on 18-digit identifiers, so 293,200 rows collapse to 230,529
    distinct values. Any merge on it silently duplicates rows, which is why the
    context is attached in one pass instead.
    """
    frames = []
    for year in years:
        season = load_season(year)
        if model is not None:
            season = _attach_win_probability(season, model)
        for quarterback in ingest.QUARTERBACKS:
            plays = ingest.extract(season, quarterback)
            plays = plays[plays["play_kind"].isin(SCRIMMAGE)].copy()
            plays["season"] = year
            frames.append(plays)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel["EPA"].notna()].copy()

    team = panel["pos_team"]
    is_home = panel["home_team"] == team
    panel["opponent"] = panel["away_team"].where(is_home, panel["home_team"])
    panel["opponent_elo"] = panel["away_team_pregame_elo"].where(
        is_home, panel["home_team_pregame_elo"]
    )
    panel["opponent_division"] = panel["away_team_division"].where(
        is_home, panel["home_team_division"]
    )
    panel["is_fcs_opponent"] = panel["opponent_division"].fillna("").str.lower().eq("fcs")

    if "seconds_left" not in panel:
        panel["seconds_left"] = (
            (4 - panel["period"].clip(upper=4)) * 900
            + panel["clock_minutes"].fillna(0) * 60
            + panel["clock_seconds"].fillna(0)
        )
    if "score_diff" not in panel:
        panel["score_diff"] = panel["pos_score_diff_start"]
    panel["is_dropback"] = panel["play_kind"].isin({"pass", "sack"})
    return panel.reset_index(drop=True)


def frames(panel: pd.DataFrame, season: int | None = None, dropbacks_only: bool = True,
           drop_fcs: bool = False, drop_nullified: bool = True) -> dict[str, pd.DataFrame]:
    """Split the panel into one frame per quarterback under a given filter."""
    view = panel
    if season is not None:
        view = view[view["season"] == season]
    if dropbacks_only:
        view = view[view["is_dropback"]]
    if drop_fcs:
        view = view[~view["is_fcs_opponent"]]
    if drop_nullified:
        view = view[~view["nullified"]]
    return {qb: view[view["quarterback"] == qb].copy() for qb in ingest.QUARTERBACKS}
