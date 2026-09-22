"""Does a boom-bust passing profile help a team come back?

The Ohio State game is a single observation, and it was chosen precisely because
Texas won it — selecting on the outcome guarantees a flattering answer. So the
hypothesis is moved to a population where it can actually be tested: every team
in the corpus that trailed by two scores or more entering the fourth quarter.

The question then becomes answerable with real power. Holding a team's average
passing efficiency constant, does having a **fatter right tail** — more snaps
that produce a big gain — raise the chance of completing a comeback? That is the
general form of "is Arch's brilliance more impactful than Gunner's consistency",
and it is estimated from hundreds of comeback attempts rather than one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

from .panel import DATA

COLUMNS = [
    "game_id", "pos_team", "def_pos_team", "home_team", "away_team", "period",
    "pos_team_score", "def_pos_team_score", "play_type", "EPA", "clock_minutes",
    "home_team_division", "away_team_division",
]
PASS_TYPES = {"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack"}
RUSH_TYPES = {"Rush", "Rushing Touchdown"}


def _load(years) -> pd.DataFrame:
    frames = []
    for year in years:
        season = pq.read_table(DATA / f"pbp_{year}.parquet", columns=COLUMNS).to_pandas()
        season["season"] = year
        frames.append(season)
    return pd.concat(frames, ignore_index=True)


def team_profiles(plays: pd.DataFrame) -> pd.DataFrame:
    """Passing profile per team-season: average efficiency and tail weight.

    Profiles are computed per season, not pooled, because a roster five years
    apart is a different team.

    ``tail_rate`` is the share of dropbacks producing more than +1 EPA. It is
    the distributional shape term; ``mean_epa`` is the level term. Separating
    them is the whole point — the hypothesis is about shape at a given level.
    """
    passing = plays[plays["play_type"].isin(PASS_TYPES)].dropna(subset=["EPA"])
    rushing = plays[plays["play_type"].isin(RUSH_TYPES)].dropna(subset=["EPA"])

    pass_profile = passing.groupby(["pos_team", "season"])["EPA"].agg(
        pass_epa="mean", pass_tail=lambda s: (s > 1).mean(),
        pass_floor=lambda s: (s < -0.5).mean(), pass_sd="std", pass_n="size",
    )
    rush_profile = rushing.groupby(["pos_team", "season"])["EPA"].agg(rush_epa="mean", rush_n="size")
    profile = pass_profile.join(rush_profile, how="inner")
    # A high pass-volume threshold would quietly drop run-first teams such as the
    # service academies, which is exactly the wrong selection for a study about
    # how offensive shape affects comebacks. The floor is therefore low enough to
    # keep them while still excluding teams with too few snaps to estimate a tail.
    return profile[(profile["pass_n"] >= 100) & (profile["rush_n"] >= 100)].reset_index()


def comeback_attempts(plays: pd.DataFrame, deficit: int = 14, fbs_only: bool = True) -> pd.DataFrame:
    """Every team that entered the fourth quarter trailing by ``deficit`` or more.

    The corpus contains FCS and lower-division opponents. Restricting to games
    where both sides are FBS keeps the comparison to one competitive population;
    leaving them in produced an implausible 50% comeback rate among teams such as
    Bentley and Cumberland, which is a data artefact rather than a finding.
    """
    if fbs_only:
        both_fbs = (
            plays["home_team_division"].fillna("").str.lower().eq("fbs")
            & plays["away_team_division"].fillna("").str.lower().eq("fbs")
        )
        plays = plays[both_fbs]
    # Final score per game, from the last row.
    ordered = plays.sort_values(["game_id", "period", "clock_minutes"], ascending=[True, True, False])
    last = ordered.groupby("game_id").tail(1)
    home_final = np.where(
        last["pos_team"] == last["home_team"], last["pos_team_score"], last["def_pos_team_score"]
    )
    away_final = np.where(
        last["pos_team"] == last["home_team"], last["def_pos_team_score"], last["pos_team_score"]
    )
    finals = pd.DataFrame(
        {"game_id": last["game_id"].to_numpy(), "home_final": home_final, "away_final": away_final}
    )

    # Score state at the first play of the fourth quarter.
    fourth = ordered[ordered["period"] == 4].groupby("game_id").head(1)
    rows = []
    for row in fourth.itertuples():
        is_home = row.pos_team == row.home_team
        home_score = row.pos_team_score if is_home else row.def_pos_team_score
        away_score = row.def_pos_team_score if is_home else row.pos_team_score
        for team, own, opponent in (
            (row.home_team, home_score, away_score),
            (row.away_team, away_score, home_score),
        ):
            rows.append(
                {"game_id": row.game_id, "season": row.season, "team": team,
                 "margin": own - opponent, "is_home": team == row.home_team}
            )
    states = pd.DataFrame(rows).merge(finals, on="game_id", how="left")
    states["won"] = np.where(
        states["is_home"], states["home_final"] > states["away_final"],
        states["away_final"] > states["home_final"],
    ).astype(int)
    return states[states["margin"] <= -deficit].reset_index(drop=True)


def fit_model(years=(2022, 2023, 2024, 2025, 2026), deficit: int = 10) -> dict:
    """Estimate whether tail weight predicts comebacks, holding the mean fixed."""
    plays = _load(years)
    profiles = team_profiles(plays)
    attempts = comeback_attempts(plays, deficit).merge(
        profiles, left_on=["team", "season"], right_on=["pos_team", "season"], how="inner"
    )
    if attempts.empty:
        return {}

    # Standardise so coefficients are comparable in units of a standard deviation.
    predictors = ["pass_epa", "pass_tail", "rush_epa"]
    design = attempts[predictors].copy()
    for column in predictors:
        design[column] = (design[column] - design[column].mean()) / design[column].std()
    design["deficit"] = (attempts["margin"] - attempts["margin"].mean()) / attempts["margin"].std()
    design["is_home"] = attempts["is_home"].astype(int)

    model = sm.Logit(attempts["won"], sm.add_constant(design)).fit(disp=False)
    return {
        "n_attempts": int(len(attempts)),
        "n_wins": int(attempts["won"].sum()),
        "base_rate": float(attempts["won"].mean()),
        "model": model,
        "attempts": attempts,
        "profiles": profiles,
    }
