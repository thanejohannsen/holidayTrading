"""A win-probability model fit from the play-by-play corpus.

Neither shipped source is usable for this. The bulk feed's win-probability
columns are corrupted (routine plays swing home WP by 30+ points on 5.8% of 2025
snaps and 12.7% of 2026 snaps), and ESPN's endpoint rate-limits. So the model is
fit here from game states and realized outcomes, which also means one consistent
model covers every play in the comparison — the property that matters most when
the whole point is to compare two quarterbacks on a common scale.

Leverage is then defined league-wide, from the dispersion of realized win-
probability swings within a game state. Because it is estimated from all teams,
it cannot be inflated by either quarterback's own play.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .panel import DATA

TRAIN_COLUMNS = [
    "game_id", "pos_team", "home_team", "away_team", "pos_team_score",
    "def_pos_team_score", "period", "clock_minutes", "clock_seconds",
    "down", "distance", "yards_to_goal",
]
FEATURES = [
    "score_diff", "seconds_left", "down", "distance", "yards_to_goal",
    "is_home", "score_diff_per_min", "is_overtime",
]


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Add model features and the realized outcome for each play's offense."""
    out = frame.copy()
    out["score_diff"] = out["pos_team_score"] - out["def_pos_team_score"]
    regulation = out["period"].clip(upper=4)
    out["seconds_left"] = (
        (4 - regulation) * 900
        + out["clock_minutes"].fillna(0) * 60
        + out["clock_seconds"].fillna(0)
    ).clip(lower=0)
    out["is_overtime"] = (out["period"] > 4).astype(int)
    out["is_home"] = (out["pos_team"] == out["home_team"]).astype(int)
    # The quantity that actually drives win probability: how big the lead is
    # relative to how much time is left to erase it.
    out["score_diff_per_min"] = out["score_diff"] / np.sqrt(out["seconds_left"] / 60 + 1)
    return out


def _outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Final home/away scores per game, taken from the last row of each game."""
    last = frame.sort_values(["game_id", "period", "clock_minutes"], ascending=[True, True, False])
    last = last.groupby("game_id").tail(1)
    home_score = np.where(
        last["pos_team"] == last["home_team"], last["pos_team_score"], last["def_pos_team_score"]
    )
    away_score = np.where(
        last["pos_team"] == last["home_team"], last["def_pos_team_score"], last["pos_team_score"]
    )
    return pd.DataFrame(
        {"game_id": last["game_id"].to_numpy(), "home_final": home_score, "away_final": away_score}
    )


def training_set(years=(2025, 2026)) -> pd.DataFrame:
    frames = []
    for year in years:
        season = pq.read_table(DATA / f"pbp_{year}.parquet", columns=TRAIN_COLUMNS).to_pandas()
        prepared = _prepare(season)
        prepared = prepared.merge(_outcomes(season), on="game_id", how="left")
        prepared["home_won"] = (prepared["home_final"] > prepared["away_final"]).astype(int)
        # Ties leave the target undefined; they are rare and dropped.
        prepared = prepared[prepared["home_final"] != prepared["away_final"]]
        prepared["pos_won"] = np.where(
            prepared["is_home"] == 1, prepared["home_won"], 1 - prepared["home_won"]
        )
        prepared["season"] = year
        frames.append(prepared)
    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=FEATURES + ["pos_won"])


def fit(data: pd.DataFrame) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=6, min_samples_leaf=200, random_state=20260922
    )
    model.fit(data[FEATURES], data["pos_won"])
    return model


def validate(data: pd.DataFrame, n_splits: int = 4) -> dict:
    """Out-of-fold calibration, grouped by game so no game spans the split."""
    splitter = GroupKFold(n_splits=n_splits)
    predictions = np.zeros(len(data))
    for train_idx, test_idx in splitter.split(data, groups=data["game_id"]):
        model = fit(data.iloc[train_idx])
        predictions[test_idx] = model.predict_proba(data.iloc[test_idx][FEATURES])[:, 1]

    actual = data["pos_won"].to_numpy()
    observed, expected = calibration_curve(actual, predictions, n_bins=10, strategy="quantile")
    return {
        "brier": float(np.mean((predictions - actual) ** 2)),
        # A model that always guessed the base rate, for reference.
        "brier_baseline": float(np.mean((actual.mean() - actual) ** 2)),
        "max_calibration_error": float(np.max(np.abs(observed - expected))),
        "curve": pd.DataFrame({"predicted": expected, "actual": observed}),
        "oof_predictions": predictions,
    }


def leverage_table(data: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """League-wide leverage: how much win probability is at stake in a state.

    Leverage is the standard deviation of the win-probability change that
    actually occurred across all teams in a comparable state, normalised so that
    1.0 is an average snap. It is deliberately computed without reference to
    either quarterback.
    """
    frame = data.copy()
    frame["wp"] = predictions
    frame = frame.sort_values(["game_id", "period", "seconds_left"], ascending=[True, True, False])
    frame["wp_next"] = frame.groupby("game_id")["wp"].shift(-1)
    frame["swing"] = (frame["wp_next"] - frame["wp"]).abs()

    frame["state"] = list(
        zip(
            frame["down"].fillna(0).astype(int),
            pd.cut(frame["distance"], [-1, 3, 7, 12, 100], labels=False).fillna(0).astype(int),
            pd.cut(frame["yards_to_goal"], [-1, 20, 50, 80, 100], labels=False).fillna(0).astype(int),
            pd.cut(frame["score_diff"], [-100, -17, -9, -4, 0, 4, 9, 17, 100], labels=False).fillna(0).astype(int),
            pd.cut(frame["seconds_left"], [-1, 120, 420, 900, 1800, 3600], labels=False).fillna(0).astype(int),
        )
    )
    table = frame.groupby("state")["swing"].agg(["mean", "size"]).reset_index()
    table = table[table["size"] >= 30]
    # Normalise so that leverage 1.0 is an average *play*, not an average
    # *state*. Rare states carry large swings, so an unweighted mean across
    # states would push every ordinary snap below 1.
    play_weighted = np.average(table["mean"], weights=table["size"])
    table["leverage"] = table["mean"] / play_weighted
    return table[["state", "leverage", "size"]]
