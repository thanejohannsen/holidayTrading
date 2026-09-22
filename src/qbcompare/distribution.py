"""Compare two quarterbacks by the shape of their play outcomes, not their average.

The question is not "who has the higher mean EPA" — the means turn out to be
nearly identical — but whether one quarterback's value arrives steadily and the
other's arrives in bursts. That is a question about distributions, so the tools
here work on quantiles and tails rather than on central tendency.

Two statistical hazards drive the design:

* Plays are nested in games, so a play-level standard error understates
  uncertainty badly. Every interval here is a **game-clustered bootstrap**.
* Selecting each quarterback's own top 5% and comparing them guarantees
  regression to the mean. The tail test is therefore **cross-validated**: the
  threshold is set on one half of a quarterback's plays and the exceedance
  measured on the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260922)
N_BOOT = 4000


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic,
    n_boot: int = N_BOOT,
    cluster: str = "game_id",
) -> tuple[float, float, float]:
    """Resample whole games with replacement and return (estimate, lo, hi).

    Resampling games rather than plays preserves the within-game correlation
    that makes play-level intervals far too narrow.
    """
    observed = statistic(frame)
    groups = [group for _, group in frame.groupby(cluster)]
    if len(groups) < 2:
        return observed, np.nan, np.nan

    draws = []
    for _ in range(n_boot):
        picked = RNG.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in picked], ignore_index=True)
        value = statistic(sample)
        if value is not None and np.isfinite(value):
            draws.append(value)
    if not draws:
        return observed, np.nan, np.nan
    return observed, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summary(frame: pd.DataFrame, column: str = "EPA") -> dict:
    """Central tendency, floor and tail for one quarterback-season."""
    values = frame[column].dropna()
    if values.empty:
        return {}

    mean, mean_lo, mean_hi = cluster_bootstrap(frame, lambda f: f[column].mean())
    return {
        "n": int(len(values)),
        "games": int(frame["game_id"].nunique()),
        "mean": float(values.mean()),
        "mean_lo": mean_lo,
        "mean_hi": mean_hi,
        "median": float(values.median()),
        "sd": float(values.std(ddof=1)),
        # Floor, on absolute thresholds so the two quarterbacks are comparable.
        # Within-player quantiles would define the floor differently for each.
        "negative_rate": float((values < 0).mean()),
        "bad_rate": float((values < -0.5).mean()),
        "disaster_rate": float((values < -1.5).mean()),
        # Tail, likewise on fixed thresholds.
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "explosive_rate": float((values > 1).mean()),
        "big_rate": float((values > 2).mean()),
        "huge_rate": float((values > 3).mean()),
    }


def shift_function(
    a: pd.Series, b: pd.Series, quantiles=np.arange(0.05, 0.96, 0.05)
) -> pd.DataFrame:
    """Quantile-by-quantile difference between two quarterbacks.

    This is the single most informative view of the question: it shows *where*
    in the distribution the two differ, instead of collapsing the difference to
    one number. A curve that is negative at the bottom and positive at the top
    is the signature of a boom-bust quarterback against a steady one.
    """
    a_clean, b_clean = a.dropna(), b.dropna()
    rows = []
    for q in quantiles:
        qa, qb = a_clean.quantile(q), b_clean.quantile(q)
        rows.append({"quantile": round(float(q), 2), "a": qa, "b": qb, "diff": qa - qb})
    return pd.DataFrame(rows)


def shift_bands(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    column: str = "EPA",
    quantiles=np.arange(0.05, 0.96, 0.05),
    n_boot: int = 1500,
) -> pd.DataFrame:
    """Shift function with game-clustered bootstrap bands."""
    groups_a = [g for _, g in frame_a.groupby("game_id")]
    groups_b = [g for _, g in frame_b.groupby("game_id")]
    draws = []
    for _ in range(n_boot):
        sa = pd.concat([groups_a[i] for i in RNG.integers(0, len(groups_a), len(groups_a))])
        sb = pd.concat([groups_b[i] for i in RNG.integers(0, len(groups_b), len(groups_b))])
        draws.append(
            [sa[column].quantile(q) - sb[column].quantile(q) for q in quantiles]
        )
    spread = np.array(draws)
    base = shift_function(frame_a[column], frame_b[column], quantiles)
    base["lo"] = np.percentile(spread, 2.5, axis=0)
    base["hi"] = np.percentile(spread, 97.5, axis=0)
    base["excludes_zero"] = (base["lo"] > 0) | (base["hi"] < 0)
    return base


def tail_share(frame: pd.DataFrame, column: str = "EPA", top=(0.01, 0.05, 0.10)) -> dict:
    """How much of a quarterback's total value comes from his best plays.

    This is the literal form of the hypothesis that one quarterback is carried
    by a small fraction of brilliant snaps. Positive EPA is used as the base so
    the share is not distorted by negatives cancelling out.
    """
    values = frame[column].dropna().sort_values(ascending=False)
    positive_total = values[values > 0].sum()
    out = {}
    for fraction in top:
        k = max(int(round(len(values) * fraction)), 1)
        out[f"top{int(fraction * 100)}pct_share"] = float(values.head(k).sum() / positive_total)
        out[f"top{int(fraction * 100)}pct_mean"] = float(values.head(k).mean())
    return out


def crossval_tail(
    frame: pd.DataFrame, column: str = "EPA", percentile: float = 95, n_splits: int = 800
) -> dict:
    """Out-of-sample tail strength, free of max-selection bias.

    Setting the threshold and measuring the exceedance on the same plays
    guarantees an inflated answer. Here the threshold comes from one random half
    and the exceedance rate and conditional mean are measured on the other.
    """
    values = frame[column].dropna().to_numpy()
    if len(values) < 40:
        return {}
    rates, means = [], []
    for _ in range(n_splits):
        order = RNG.permutation(len(values))
        half = len(values) // 2
        train, test = values[order[:half]], values[order[half:]]
        threshold = np.percentile(train, percentile)
        exceed = test[test > threshold]
        rates.append(len(exceed) / len(test))
        if len(exceed):
            means.append(exceed.mean())
    return {
        "oos_exceed_rate": float(np.mean(rates)),
        "oos_tail_mean": float(np.mean(means)) if means else np.nan,
    }


def decompose_gap(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame, column: str = "EPA",
    floor: float = -0.5, ceiling: float = 1.0,
) -> pd.DataFrame:
    """Split the mean difference into floor, middle and tail contributions.

    Each band contributes P(band) x E[value | band] to the mean, so differencing
    those products says how much of the overall gap is the bad plays, the
    routine plays, and the explosive plays. This puts "consistent" and
    "brilliant" on one common scale, which is what the comparison needs.
    """
    bands = {"floor": (-np.inf, floor), "middle": (floor, ceiling), "tail": (ceiling, np.inf)}
    rows = []
    for name, (low, high) in bands.items():
        contributions = {}
        for label, frame in (("a", frame_a), ("b", frame_b)):
            values = frame[column].dropna()
            mask = (values > low) & (values <= high)
            share = mask.mean()
            conditional = values[mask].mean() if mask.any() else 0.0
            contributions[label] = share * conditional
            contributions[f"{label}_share"] = share
            contributions[f"{label}_mean"] = conditional
        rows.append({"band": name, **contributions, "contribution_diff": contributions["a"] - contributions["b"]})
    return pd.DataFrame(rows)


def min_detectable_effect(sd: float, n_per_side: int, plays_per_game: float, icc: float = 0.03) -> dict:
    """The smallest EPA/play gap this sample could detect, after clustering.

    Plays within a game are correlated, which inflates the effective standard
    error by the design effect. Reporting this first keeps every null result in
    the study honest: it says what the data could not have seen.
    """
    design_effect = 1 + (plays_per_game - 1) * icc
    se_difference = sd * np.sqrt(2 * design_effect / n_per_side)
    return {
        "sd": sd,
        "n_per_side": n_per_side,
        "design_effect": design_effect,
        "se_diff": se_difference,
        # 2.8 = z(0.975) + z(0.80), the usual 80%-power constant.
        "mde_80pct": 2.8 * se_difference,
    }
