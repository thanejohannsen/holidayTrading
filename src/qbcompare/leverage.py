"""Does a quarterback's value arrive when it matters?

EPA treats every snap alike; a game does not. This module converts plays into
win-probability terms and then asks the question that actually separates "good
statistics" from "impactful": is one quarterback's production concentrated in
snaps that decided games?

Two traps are handled explicitly.

**The opportunity trap.** A quarterback on a trailing team is handed more
high-leverage snaps automatically. Total win probability added therefore rewards
being behind, not being good. So total WPA is decomposed into how many snaps a
quarterback took, how leveraged they were (opportunity), and how well he played
per unit of leverage (skill).

**The self-fulfilling trap.** To test whether the coupling between situation and
outcome is real, outcomes are permuted *within* a quarterback across *his own*
situations, holding the situation mix exactly fixed. Only the situation-outcome
pairing is destroyed. If the observed clutch statistic sits inside that null,
his win-probability total is explained by the chances he got, not by what he did
with them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260922)


def add_leverage(panel: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Join league-wide leverage onto each quarterback play by game state."""
    out = panel.copy()
    out["state"] = list(
        zip(
            out["down"].fillna(0).astype(int),
            pd.cut(out["distance"], [-1, 3, 7, 12, 100], labels=False).fillna(0).astype(int),
            pd.cut(out["yards_to_goal"], [-1, 20, 50, 80, 100], labels=False).fillna(0).astype(int),
            pd.cut(out["score_diff"], [-100, -17, -9, -4, 0, 4, 9, 17, 100], labels=False).fillna(0).astype(int),
            pd.cut(out["seconds_left"], [-1, 120, 420, 900, 1800, 3600], labels=False).fillna(0).astype(int),
        )
    )
    lookup = dict(zip(table["state"], table["leverage"]))
    out["leverage"] = out["state"].map(lookup)
    # States too rare to estimate fall back to an average snap.
    out["leverage"] = out["leverage"].fillna(1.0)
    out["leverage_bucket"] = pd.qcut(
        out["leverage"], 4, labels=["low", "medium", "high", "highest"], duplicates="drop"
    )
    return out


def decompose_wpa(frame: pd.DataFrame) -> dict:
    """Split total win probability added into opportunity and skill.

    Total WPA is approximately plays x mean leverage x performance per unit of
    leverage, plus the covariance between leverage and performance. Reporting
    the pieces separately is what answers "was he better when it mattered, or
    just more often in spots that mattered".
    """
    valid = frame.dropna(subset=["wpa_own", "leverage"])
    if valid.empty:
        return {}
    leverage = valid["leverage"].to_numpy()
    wpa = valid["wpa_own"].to_numpy()
    performance = wpa / np.where(leverage == 0, np.nan, leverage)
    return {
        "plays": int(len(valid)),
        "total_wpa": float(wpa.sum()),
        "mean_leverage": float(leverage.mean()),
        "performance_per_leverage": float(np.nanmean(performance)),
        # Positive covariance is the clutch term: playing better precisely when
        # more was at stake.
        "clutch_covariance": float(np.cov(leverage, wpa)[0, 1]),
    }


def leverage_gradient(frame: pd.DataFrame, column: str = "EPA") -> pd.DataFrame:
    """Performance by leverage bucket, relative to the league in that bucket."""
    rows = []
    for bucket, group in frame.groupby("leverage_bucket", observed=True):
        rows.append(
            {
                "bucket": bucket,
                "plays": len(group),
                "mean_epa": group[column].mean(),
                "mean_wpa": group["wpa_own"].mean(),
                "mean_leverage": group["leverage"].mean(),
            }
        )
    return pd.DataFrame(rows)


def permutation_clutch(frame: pd.DataFrame, n_permutations: int = 5000) -> dict:
    """Is the leverage-outcome coupling more than chance?

    Outcomes are shuffled within coarse situation strata, so a third-and-one
    result never lands in a third-and-fifteen slot. The quarterback's situation
    mix — his opportunity — is held exactly fixed; only the pairing moves.
    """
    valid = frame.dropna(subset=["wpa_own", "leverage", "EPA"]).copy()
    if len(valid) < 50:
        return {}

    valid["stratum"] = list(
        zip(
            valid["down"].fillna(0).astype(int),
            pd.cut(valid["distance"], [-1, 3, 7, 12, 100], labels=False).fillna(0).astype(int),
        )
    )
    observed = float(np.cov(valid["leverage"], valid["EPA"])[0, 1])

    draws = []
    groups = [g.index.to_numpy() for _, g in valid.groupby("stratum")]
    epa = valid["EPA"].to_numpy()
    leverage = valid["leverage"].to_numpy()
    position = {idx: i for i, idx in enumerate(valid.index)}
    for _ in range(n_permutations):
        shuffled = epa.copy()
        for members in groups:
            slots = [position[i] for i in members]
            shuffled[slots] = RNG.permutation(epa[slots])
        draws.append(np.cov(leverage, shuffled)[0, 1])

    draws = np.array(draws)
    return {
        "observed_cov": observed,
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std(ddof=1)),
        "p_value": float(np.mean(np.abs(draws - draws.mean()) >= abs(observed - draws.mean()))),
    }
