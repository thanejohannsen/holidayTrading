"""Statistical tests for holiday effects in daily index returns.

Two confounds make a naive holiday-vs-rest comparison meaningless, and both are
structural rather than incidental:

1. Seasonality. The Tishrei holidays always land in September/October and the
   Nisan holidays in March/April. September is historically the weakest month of
   the year, so any Tishrei sample inherits that drift before a single holiday
   effect is considered.
2. Weekday. The Hebrew calendar's postponement rules (dechiyot) forbid Rosh
   Hashanah from falling on Sunday, Wednesday or Friday, which propagates a
   fixed weekday skew through every holiday in the year.

Every test here therefore compares a holiday against control days matched on
both calendar month and weekday, drawn from other years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260922)
N_DRAWS = 20_000


def build_panel(prices: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach holiday labels and conditioning variables to the price frame."""
    panel = prices.copy()
    panel["year"] = panel.index.year
    panel["month"] = panel.index.month
    panel["weekday"] = panel.index.weekday
    panel["abs_ret"] = panel["ret"].abs()
    # Volume trends up by orders of magnitude over a century, so only volume
    # relative to its own recent level is comparable across eras.
    trailing = panel["volume"].rolling(60, min_periods=30).median().shift(1)
    panel["rel_volume"] = panel["volume"] / trailing.replace(0, np.nan)
    # Parkinson-style intraday range, again scaled by its own recent level.
    span = np.log(panel["high"] / panel["low"])
    panel["rel_range"] = span / span.rolling(60, min_periods=30).median().shift(1).replace(0, np.nan)

    merged = labels.drop_duplicates("date").set_index("date")
    panel["holiday"] = merged["holiday"].reindex(panel.index)
    panel["category"] = merged["category"].reindex(panel.index)
    return panel


def matched_test(
    panel: pd.DataFrame,
    mask: pd.Series,
    column: str = "ret",
    exclude: pd.Series | None = None,
    n_draws: int = N_DRAWS,
) -> dict:
    """Compare ``column`` on the masked days against month- and weekday-matched days.

    The null distribution is built by repeatedly redrawing, for each treated day,
    one control day from the same calendar month and weekday in a different year.
    This preserves the seasonal and weekday footprint of the treated sample, so
    the resulting p-value reflects the holiday itself rather than the time of
    year it happens to occupy.
    """
    values = panel[column]
    treated = panel.index[mask.fillna(False)]
    treated = treated[values.reindex(treated).notna()]
    if len(treated) < 5:
        return {"n": len(treated), "mean": np.nan, "p_value": np.nan}

    # Control universe: never a labelled holiday of any kind, so the null is not
    # contaminated by the effect being tested.
    ineligible = exclude if exclude is not None else panel["category"].notna()
    pool = panel.loc[~ineligible.fillna(False) & values.notna()]

    buckets: dict[tuple[int, int], pd.DataFrame] = {
        key: group for key, group in pool.groupby(["month", "weekday"])
    }

    draw_columns = []
    usable = []
    for timestamp in treated:
        bucket = buckets.get((timestamp.month, timestamp.weekday()))
        if bucket is None:
            continue
        candidates = bucket[bucket["year"] != timestamp.year][column].to_numpy()
        if candidates.size < 10:
            continue
        usable.append(timestamp)
        draw_columns.append(RNG.choice(candidates, size=n_draws, replace=True))

    if not draw_columns:
        return {"n": len(treated), "mean": np.nan, "p_value": np.nan}

    observed_values = values.reindex(usable).to_numpy()
    observed = observed_values.mean()
    null_means = np.mean(np.column_stack(draw_columns), axis=1)
    control_mean = null_means.mean()

    # Two-sided p-value: how often chance alone produces a gap this large.
    gap = observed - control_mean
    p_value = float(np.mean(np.abs(null_means - control_mean) >= abs(gap)))

    return {
        "n": len(usable),
        "mean": float(observed),
        "control_mean": float(control_mean),
        "diff": float(gap),
        "p_value": p_value,
        "win_rate": float((observed_values > 0).mean()),
        "std": float(observed_values.std(ddof=1)),
        "null_std": float(null_means.std(ddof=1)),
    }


def by_category(panel: pd.DataFrame, column: str = "ret") -> pd.DataFrame:
    """Run the matched test once per holiday category."""
    rows = []
    for category in ["erev", "yom_tov", "chol_hamoed", "minor"]:
        mask = panel["category"] == category
        result = matched_test(panel, mask, column=column)
        rows.append({"group": category, **result})
    return pd.DataFrame(rows)


def by_holiday(panel: pd.DataFrame, min_count: int = 20, column: str = "ret") -> pd.DataFrame:
    """Run the matched test once per named holiday with enough observations."""
    rows = []
    counts = panel["holiday"].value_counts()
    for name in counts[counts >= min_count].index:
        mask = panel["holiday"] == name
        result = matched_test(panel, mask, column=column)
        category = panel.loc[mask, "category"].iloc[0]
        rows.append({"holiday": name, "category": category, **result})
    frame = pd.DataFrame(rows)
    return frame.sort_values("diff").reset_index(drop=True)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """FDR-adjusted p-values; with this many holidays, raw ones mislead."""
    clean = p_values.dropna()
    order = clean.sort_values()
    m = len(order)
    adjusted = (order.to_numpy() * m / np.arange(1, m + 1)).clip(max=1.0)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    return pd.Series(adjusted, index=order.index).reindex(p_values.index)
