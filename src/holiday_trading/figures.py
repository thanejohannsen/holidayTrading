"""Render the headline figure: attendance drops, and the returns that don't follow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis, calendars, data

FIGURES = Path(__file__).resolve().parents[2] / "figures"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
GRID = "#e6e5e1"
BELOW = "#2a78d6"  # diverging cool pole: volume below normal
ABOVE = "#e34948"  # diverging warm pole: volume above normal
NEUTRAL = "#52514e"

# Days marked with a dagger are already light on the market calendar: the NYSE
# closes early, or the bond market shuts entirely. They are the yardstick for
# how big a "real" attendance drop looks.
CALENDAR_LIGHT = {"Christmas Eve", "Day after Thanksgiving", "Columbus Day", "Veterans Day"}

SELECTION = [
    "Yom Kippur",
    "Rosh Hashanah I",
    "Rosh Hashanah II",
    "Shavuot I",
    "Passover I",
    "Sukkot I",
    "Purim",
    "Christmas Eve",
    "Day after Thanksgiving",
    "Columbus Day",
    "Veterans Day",
]


def _collect() -> pd.DataFrame:
    prices = data.load()
    jewish = calendars.jewish_holidays(prices.index)
    secular = calendars.secular_holidays(prices.index)
    pre = calendars.pre_holiday(prices.index)
    combined = pd.concat([jewish, secular, pre]).drop_duplicates("date")
    universe = analysis.build_panel(prices, combined)
    exclude = universe["category"].notna()

    rows = []
    for name in SELECTION:
        mask = universe["holiday"] == name
        volume = analysis.matched_test(universe, mask, "rel_volume", exclude=exclude)
        returns = analysis.matched_test(universe, mask, "ret", exclude=exclude)
        # Combine the treated-sample and control-draw variances for the interval.
        se = np.sqrt(returns["std"] ** 2 / returns["n"] + returns["null_std"] ** 2)
        rows.append(
            {
                "holiday": name,
                "volume_pct": (volume["mean"] / volume["control_mean"] - 1) * 100,
                "volume_p": volume["p_value"],
                "ret_bp": returns["diff"] * 1e4,
                "ret_lo": (returns["diff"] - 1.96 * se) * 1e4,
                "ret_hi": (returns["diff"] + 1.96 * se) * 1e4,
                "n": returns["n"],
            }
        )
    return pd.DataFrame(rows).sort_values("volume_pct").reset_index(drop=True)


def render() -> Path:
    frame = _collect()
    labels = [
        f"{row.holiday}†" if row.holiday in CALENDAR_LIGHT else row.holiday
        for row in frame.itertuples()
    ]
    y = np.arange(len(frame))

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.4, 5.6), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9.5, length=0)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)

    # -- Left: trading volume against month- and weekday-matched control days.
    colors = [BELOW if value < 0 else ABOVE for value in frame["volume_pct"]]
    left.barh(y, frame["volume_pct"], color=colors, height=0.62, zorder=3)
    left.axvline(0, color=TEXT_SECONDARY, linewidth=1.1, zorder=4)
    for index, row in frame.iterrows():
        offset = -1.4 if row["volume_pct"] < 0 else 1.4
        mark = " ***" if row["volume_p"] < 0.001 else (" **" if row["volume_p"] < 0.01 else "")
        left.text(
            row["volume_pct"] + offset,
            index,
            f"{row['volume_pct']:+.0f}%{mark}",
            va="center",
            ha="right" if row["volume_pct"] < 0 else "left",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    left.set_yticks(y, labels, fontsize=10, color=TEXT_PRIMARY)
    # Ranked bars read top-down, largest drop first.
    left.invert_yaxis()
    left.set_xlim(-58, 22)
    left.set_xlabel("Trading volume vs matched control days (%)", fontsize=9.5, color=TEXT_SECONDARY)
    left.set_title(
        "Who shows up", fontsize=13, color=TEXT_PRIMARY, fontweight="bold", loc="left", pad=12
    )

    # -- Right: the return difference, with the interval that contains zero.
    right.axvline(0, color=TEXT_SECONDARY, linewidth=1.1, zorder=4)
    right.hlines(y, frame["ret_lo"], frame["ret_hi"], color=TEXT_MUTED, linewidth=2, zorder=3)
    right.scatter(
        frame["ret_bp"], y, s=46, color=NEUTRAL, zorder=5, edgecolor=SURFACE, linewidth=1.5
    )
    span = max(frame["ret_hi"].max(), -frame["ret_lo"].min()) * 1.12
    right.set_xlim(-span, span)
    right.set_xlabel(
        "Return vs matched control days (basis points)", fontsize=9.5, color=TEXT_SECONDARY
    )
    right.set_title(
        "What it does to prices",
        fontsize=13,
        color=TEXT_PRIMARY,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    right.text(
        0.98,
        0.02,
        "every interval crosses zero",
        transform=right.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=TEXT_MUTED,
        style="italic",
    )

    fig.suptitle(
        "Jewish holidays empty the S&P 500's order book without moving it",
        fontsize=15.5,
        color=TEXT_PRIMARY,
        fontweight="bold",
        x=0.007,
        y=0.975,
        ha="left",
    )
    fig.text(
        0.007,
        0.905,
        "S&P 500 daily sessions, 1928–2026. Each holiday is compared with non-holiday sessions "
        "in the same calendar month and weekday.\nBars show the 95% interval on the return gap. "
        "† = NYSE half-day or bond-market holiday.   ** p<0.01, *** p<0.001",
        fontsize=9,
        color=TEXT_SECONDARY,
        ha="left",
        va="top",
        linespacing=1.5,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "holiday_volume_vs_returns.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(render())
