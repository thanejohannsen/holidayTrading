"""Regenerate every result in the study. Usage: python -m qbcompare.report"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import comeback, distribution as dist, ingest, leverage, panel as pn, wpmodel

RESULTS = Path(__file__).resolve().parents[2] / "results"
SEASONS_ALL = (2022, 2023, 2024, 2025, 2026)


def _frames(panel, season):
    return pn.frames(panel, season=season, dropbacks_only=True, drop_fcs=True)


def build() -> dict:
    RESULTS.mkdir(exist_ok=True)
    train = pd.read_parquet("data/wp_train.parquet")
    oof = np.load("data/wp_oof.npy")
    model = wpmodel.fit(train)
    table = wpmodel.leverage_table(train, oof)

    panel = pn.build(model=model)
    panel = leverage.add_leverage(panel, table)
    panel.to_parquet("data/qb_panel_lev.parquet")

    out: dict = {}
    tables: dict[str, pd.DataFrame] = {}

    # --- Attribution audit: the gate everything else rests on.
    audits = []
    for year in (2025, 2026):
        season = pn.load_season(year)
        for quarterback in ingest.QUARTERBACKS:
            audits.append({"season": year, **ingest.audit(season, quarterback)})
    tables["attribution_audit"] = pd.DataFrame(audits)

    # --- Distribution, per season.
    rows, shifts, decomps = [], [], []
    for season in (2025, 2026):
        frames = _frames(panel, season)
        for name, frame in frames.items():
            if len(frame) < 10:
                continue
            rows.append({"season": season, "qb": name, **dist.summary(frame),
                         **dist.tail_share(frame), **dist.crossval_tail(frame)})
        if all(len(f) >= 40 for f in frames.values()):
            shift = dist.shift_bands(frames["Manning"], frames["Stockton"])
            shift["season"] = season
            shifts.append(shift)
            decomp = dist.decompose_gap(frames["Manning"], frames["Stockton"])
            decomp["season"] = season
            decomps.append(decomp)
    tables["distribution"] = pd.DataFrame(rows)
    tables["shift_function"] = pd.concat(shifts, ignore_index=True)
    tables["gap_decomposition"] = pd.concat(decomps, ignore_index=True)

    # --- Power: what this sample could not have seen.
    frames = _frames(panel, 2025)
    pooled_sd = pd.concat([f["EPA"] for f in frames.values()]).std()
    plays_per_game = np.mean([len(f) / f["game_id"].nunique() for f in frames.values()])
    power = dist.min_detectable_effect(
        pooled_sd, min(len(f) for f in frames.values()), plays_per_game
    )
    power["observed_gap_2025"] = frames["Stockton"]["EPA"].mean() - frames["Manning"]["EPA"].mean()
    tables["power"] = pd.DataFrame([power])

    # --- Leverage.
    grads, decs, perms = [], [], []
    for season in (2025, 2026):
        for name, frame in _frames(panel, season).items():
            if len(frame) < 40:
                continue
            grad = leverage.leverage_gradient(frame)
            grad["qb"], grad["season"] = name, season
            grads.append(grad)
            decs.append({"season": season, "qb": name, **leverage.decompose_wpa(frame)})
            result = leverage.permutation_clutch(frame)
            if result:
                perms.append({"season": season, "qb": name, **result})
    tables["leverage_gradient"] = pd.concat(grads, ignore_index=True)
    tables["wpa_decomposition"] = pd.DataFrame(decs)
    tables["clutch_permutation"] = pd.DataFrame(perms)

    # --- Adjustment ladder and how much weight 2026 deserves.
    ladder = []
    for season in (2025, 2026):
        for label, kwargs in [
            ("raw", dict(dropbacks_only=False, drop_fcs=False)),
            ("dropbacks", dict(dropbacks_only=True, drop_fcs=False)),
            ("dropbacks + FCS removed", dict(dropbacks_only=True, drop_fcs=True)),
        ]:
            for name, frame in pn.frames(panel, season=season, **kwargs).items():
                if len(frame) < 5:
                    continue
                est, lo, hi = dist.cluster_bootstrap(frame, lambda f: f["EPA"].mean(), n_boot=3000)
                ladder.append({"season": season, "rung": label, "qb": name,
                               "n": len(frame), "mean_epa": est, "lo": lo, "hi": hi})
    tables["adjustment_ladder"] = pd.DataFrame(ladder)

    weights = []
    for name in ingest.QUARTERBACKS:
        prior = _frames(panel, 2025)[name]
        recent = _frames(panel, 2026)[name]
        # Game-clustered standard errors: the unit of independent information is
        # a game, not a snap.
        se_prior = prior["EPA"].std() / np.sqrt(prior["game_id"].nunique())
        se_recent = recent["EPA"].std() / np.sqrt(max(recent["game_id"].nunique(), 1))
        w_prior, w_recent = 1 / se_prior**2, 1 / se_recent**2
        weights.append({
            "qb": name, "mean_2025": prior["EPA"].mean(), "n_2025": len(prior),
            "mean_2026": recent["EPA"].mean(), "n_2026": len(recent),
            "weight_2026": w_recent / (w_prior + w_recent),
            "posterior": (prior["EPA"].mean() * w_prior + recent["EPA"].mean() * w_recent)
            / (w_prior + w_recent),
        })
    tables["precision_weighting"] = pd.DataFrame(weights)

    # --- Comeback capability across five seasons.
    plays = comeback._load(SEASONS_ALL)
    rates = []
    for deficit in (7, 10, 14, 17, 20):
        attempts = comeback.comeback_attempts(plays, deficit)
        rates.append({"deficit": deficit, "attempts": len(attempts),
                      "comebacks": int(attempts["won"].sum()),
                      "rate": float(attempts["won"].mean())})
    tables["comeback_rates"] = pd.DataFrame(rates)

    models = []
    for deficit in (7, 10):
        fitted = comeback.fit_model(years=SEASONS_ALL, deficit=deficit)
        if not fitted:
            continue
        m = fitted["model"]
        interval = m.conf_int()
        for term in m.params.index:
            models.append({"deficit": deficit, "term": term, "coef": m.params[term],
                           "se": m.bse[term], "p": m.pvalues[term],
                           "odds_ratio": float(np.exp(m.params[term])),
                           "or_lo": float(np.exp(interval.loc[term, 0])),
                           "or_hi": float(np.exp(interval.loc[term, 1])),
                           "n": fitted["n_attempts"], "events": fitted["n_wins"]})
    tables["comeback_model"] = pd.DataFrame(models)

    for name, frame in tables.items():
        frame.to_csv(RESULTS / f"{name}.csv", index=False)
    out["tables"] = {k: v.to_dict("records") for k, v in tables.items()}
    (RESULTS / "results.json").write_text(json.dumps(out, indent=1, default=str))
    return tables


if __name__ == "__main__":
    for name, frame in build().items():
        print(f"wrote results/{name}.csv ({len(frame)} rows)")
