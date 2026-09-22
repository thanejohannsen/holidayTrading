"""Regenerate every table in REPORT.md from scratch.

Usage:  python -m holiday_trading.report  [--refresh]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyluach import dates as hdates
from scipy import stats

from . import analysis, calendars, data

RESULTS = Path(__file__).resolve().parents[2] / "results"
BP = 1e4


def _tidy(frame: pd.DataFrame, bp_columns=("mean", "control_mean", "diff")) -> pd.DataFrame:
    out = frame.copy()
    for column in bp_columns:
        if column in out:
            out[column + "_bp"] = (out[column] * BP).round(2)
    return out.drop(columns=[c for c in bp_columns if c in out]).round(4)


def build() -> dict[str, pd.DataFrame]:
    prices = data.load()
    jewish = calendars.jewish_holidays(prices.index)
    secular = calendars.secular_holidays(prices.index)
    pre = calendars.pre_holiday(prices.index)

    # One combined label set defines the control universe, so a control day is
    # never itself some other kind of holiday.
    combined = pd.concat([jewish, secular, pre]).drop_duplicates("date")
    universe = analysis.build_panel(prices, combined)
    not_control = universe["category"].notna()

    jewish_panel = analysis.build_panel(prices, jewish)
    secular_panel = analysis.build_panel(prices, secular)
    pre_panel = analysis.build_panel(prices, pre)
    for panel in (jewish_panel, secular_panel, pre_panel):
        panel["_ex"] = not_control.reindex(panel.index)

    tables: dict[str, pd.DataFrame] = {}

    # 1. Category-level returns, volume and volatility.
    rows = []
    for column in ["ret", "rel_volume", "rel_range"]:
        for category in ["erev", "yom_tov", "chol_hamoed", "minor"]:
            result = analysis.matched_test(
                jewish_panel,
                jewish_panel["category"] == category,
                column=column,
                exclude=jewish_panel["_ex"],
            )
            rows.append({"measure": column, "group": category, **result})
    tables["category_summary"] = pd.DataFrame(rows).round(5)

    # 2. Per-holiday returns and volume, FDR-corrected across the family.
    for column, name in [("ret", "holiday_returns"), ("rel_volume", "holiday_volume")]:
        rows = []
        for holiday in jewish_panel["holiday"].value_counts().loc[lambda s: s >= 20].index:
            result = analysis.matched_test(
                jewish_panel,
                jewish_panel["holiday"] == holiday,
                column=column,
                exclude=jewish_panel["_ex"],
            )
            rows.append(
                {
                    "holiday": holiday,
                    "category": jewish_panel.loc[
                        jewish_panel["holiday"] == holiday, "category"
                    ].iloc[0],
                    **result,
                }
            )
        frame = pd.DataFrame(rows).dropna(subset=["p_value"])
        frame["fdr_p"] = analysis.benjamini_hochberg(frame["p_value"])
        bp = ("mean", "control_mean", "diff") if column == "ret" else ()
        tables[name] = _tidy(frame.sort_values("diff"), bp).reset_index(drop=True)

    # 3. Secular days the NYSE keeps open.
    for column, name in [("ret", "secular_returns"), ("rel_volume", "secular_volume")]:
        rows = []
        for holiday in secular["holiday"].unique():
            result = analysis.matched_test(
                secular_panel,
                secular_panel["holiday"] == holiday,
                column=column,
                exclude=secular_panel["_ex"],
            )
            rows.append({"holiday": holiday, **result})
        frame = pd.DataFrame(rows).dropna(subset=["p_value"])
        frame["fdr_p"] = analysis.benjamini_hochberg(frame["p_value"])
        bp = ("mean", "control_mean", "diff") if column == "ret" else ()
        tables[name] = _tidy(frame.sort_values("diff"), bp).reset_index(drop=True)

    # 4. Era splits: do the effects survive into the modern market?
    eras = [(1928, 1969), (1970, 1989), (1990, 2009), (2010, 2026)]
    rows = []
    high_holy = jewish_panel["holiday"].isin(["Yom Kippur", "Rosh Hashanah I"])
    for low, high in eras:
        window = jewish_panel[(jewish_panel["year"] >= low) & (jewish_panel["year"] <= high)]
        pre_window = pre_panel[(pre_panel["year"] >= low) & (pre_panel["year"] <= high)]
        specs = [
            ("high_holy_days_volume", window, high_holy.reindex(window.index).fillna(False), "rel_volume"),
            ("high_holy_days_return", window, high_holy.reindex(window.index).fillna(False), "ret"),
            ("yom_tov_return", window, window["category"] == "yom_tov", "ret"),
            ("pre_closure_return", pre_window, pre_window["category"] == "pre_closure", "ret"),
        ]
        for label, panel, mask, column in specs:
            result = analysis.matched_test(
                panel, mask, column=column, exclude=panel["_ex"]
            )
            rows.append({"era": f"{low}-{high}", "test": label, **result})
    tables["era_stability"] = pd.DataFrame(rows).round(5)

    # 5. Sell Rosh Hashanah / buy Yom Kippur.
    tables["rosh_hashanah_yom_kippur"] = _rh_yk(prices)

    # 6. How large an effect could this sample even see?
    daily_sd = jewish_panel["ret"].std()
    rows = [
        {
            "sample": label,
            "n": n,
            "daily_sd_bp": round(daily_sd * BP, 1),
            # 2.8 = z(0.975) + z(0.80), the usual 80%-power constant.
            "min_detectable_bp": round(2.8 * daily_sd / np.sqrt(n) * BP, 1),
        }
        for label, n in [
            ("Yom Kippur only", 68),
            ("High Holy Days", 137),
            ("All yom tov", 865),
            ("All Jewish holidays", len(jewish)),
        ]
    ]
    tables["power"] = pd.DataFrame(rows)

    return tables


def _rh_yk(prices: pd.DataFrame) -> pd.DataFrame:
    """Return the 'sell Rosh Hashanah, buy Yom Kippur' window for every year."""
    index = prices.index
    close = prices["close"]

    def session_before(stamp):
        earlier = index[index < stamp]
        return earlier[-1] if len(earlier) else None

    def session_on_or_before(stamp):
        earlier = index[index <= stamp]
        return earlier[-1] if len(earlier) else None

    rows = []
    for hebrew_year in range(5688, 5788):
        try:
            rosh = pd.Timestamp(hdates.HebrewDate(hebrew_year, 7, 1).to_pydate())
            kippur = pd.Timestamp(hdates.HebrewDate(hebrew_year, 7, 10).to_pydate())
        except Exception:  # noqa: BLE001 - out of the library's supported range
            continue
        entry, exit_ = session_before(rosh), session_on_or_before(kippur)
        if entry is None or exit_ is None or exit_ <= entry:
            continue
        if entry < index[0] or exit_ > index[-1]:
            continue
        rows.append(
            {
                "hebrew_year": hebrew_year,
                "year": entry.year,
                "entry": entry.date(),
                "exit": exit_.date(),
                "sessions": int(((index > entry) & (index <= exit_)).sum()),
                "window_return": close[exit_] / close[entry] - 1,
            }
        )
    return pd.DataFrame(rows)


def summarise(tables: dict[str, pd.DataFrame]) -> None:
    trade = tables["rosh_hashanah_yom_kippur"]
    test = stats.ttest_1samp(trade["window_return"], 0)
    print("\n=== Sell Rosh Hashanah / buy Yom Kippur ===")
    print(f"years              : {len(trade)} ({trade['year'].min()}-{trade['year'].max()})")
    print(f"mean window return : {trade['window_return'].mean() * 100:+.3f}%")
    print(f"years market fell  : {(trade['window_return'] < 0).mean() * 100:.1f}%")
    print(f"t vs zero          : t={test.statistic:.2f}, p={test.pvalue:.3f}")

    volume = tables["holiday_volume"]
    print("\n=== Largest attendance drops (volume vs matched control) ===")
    top = volume.nsmallest(4, "diff")[["holiday", "n", "mean", "control_mean", "diff", "fdr_p"]]
    print(top.to_string(index=False))

    returns = tables["holiday_returns"]
    survivors = returns[returns["fdr_p"] < 0.05]
    print(f"\nJewish holidays with an FDR-significant return effect: {len(survivors)}")

    print("\n=== Era stability ===")
    era = tables["era_stability"]
    for test_name in era["test"].unique():
        subset = era[era["test"] == test_name]
        # Returns are reported in basis points; volume is a ratio, so it is
        # reported as a percentage change against the matched control.
        is_volume = test_name.endswith("volume")
        line = "  ".join(
            f"{row.era}: "
            + (
                f"{(row.mean / row.control_mean - 1) * 100:+5.1f}%"
                if is_volume
                else f"{row.diff * BP:+6.1f}bp"
            )
            + f" (p={row.p_value:.3f})"
            for row in subset.itertuples()
        )
        print(f"{test_name:24s} {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    args = parser.parse_args()
    if args.refresh:
        data.load(refresh=True)

    RESULTS.mkdir(exist_ok=True)
    tables = build()
    for name, frame in tables.items():
        frame.to_csv(RESULTS / f"{name}.csv", index=False)
        print(f"wrote results/{name}.csv  ({len(frame)} rows)")
    summarise(tables)


if __name__ == "__main__":
    main()
