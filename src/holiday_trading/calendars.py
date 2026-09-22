"""Holiday calendars: Jewish holy days plus secular days the NYSE stays open.

The Jewish day runs from sunset to nightfall, so a holiday "on" Gregorian date D
covers the whole 9:30-16:00 session of date D. Mapping a Hebrew date onto its
Gregorian date therefore lines the holiday up with the session it affects.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from pyluach import dates as hdates
from pyluach import hebrewcal

# Hebrew month numbers as pyluach counts them (Nisan=1 ... Adar/Adar II=12/13,
# Tishrei=7). Days on which melacha is forbidden in the Diaspora: an observant
# trader cannot work, so these are the absence days the study is built around.
YOM_TOV = {
    (7, 1): "Rosh Hashanah I",
    (7, 2): "Rosh Hashanah II",
    (7, 10): "Yom Kippur",
    (7, 15): "Sukkot I",
    (7, 16): "Sukkot II",
    (7, 22): "Shemini Atzeret",
    (7, 23): "Simchat Torah",
    (1, 15): "Passover I",
    (1, 16): "Passover II",
    (1, 21): "Passover VII",
    (1, 22): "Passover VIII",
    (3, 6): "Shavuot I",
    (3, 7): "Shavuot II",
}

# Work is permitted on the intermediate days of Passover and Sukkot, but many
# observant traders still take them off. Kept as a separate, weaker bucket.
CHOL_HAMOED = {
    **{(1, d): f"Passover CH {d - 16}" for d in range(17, 21)},
    **{(7, d): f"Sukkot CH {d - 16}" for d in range(17, 22)},
}

# Work is fully permitted on these; they are the placebo group. If absenteeism
# drives any effect, these days should look ordinary.
MINOR_HOLIDAYS = {
    (12, 14): "Purim",  # 14 Adar in a common year; Purim Katan in a leap year
    (13, 14): "Purim",  # 14 Adar II in a leap year
    (5, 9): "Tisha B'Av",
    (9, 25): "Hanukkah I",
    (11, 15): "Tu BiShvat",
    (2, 18): "Lag BaOmer",
}


def _hebrew_key(date: dt.date) -> tuple[int, int]:
    hebrew = hdates.HebrewDate.from_pydate(date)
    key = (hebrew.month, hebrew.day)
    # In a leap year Adar I holds Purim Katan, a non-event; the real Purim sits
    # in Adar II. Map the decoy away so it is not counted.
    if key == (12, 14) and hebrewcal.Year(hebrew.year).leap:
        return (0, 0)
    return key


def jewish_holidays(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Label each date in ``index`` with its Jewish holiday, if any.

    Returns one row per labelled date with the holiday name and its category.
    ``erev`` marks the day before a yom tov (an afternoon many observant traders
    leave early, and the day the "sell Rosh Hashanah" adage refers to).
    """
    records = []
    labelled: dict[pd.Timestamp, None] = {}
    for timestamp in index:
        date = timestamp.date()
        key = _hebrew_key(date)
        if key in YOM_TOV:
            records.append((timestamp, YOM_TOV[key], "yom_tov"))
            labelled[timestamp] = None
        elif key in CHOL_HAMOED:
            records.append((timestamp, CHOL_HAMOED[key], "chol_hamoed"))
            labelled[timestamp] = None
        elif key in MINOR_HOLIDAYS:
            records.append((timestamp, MINOR_HOLIDAYS[key], "minor"))
            labelled[timestamp] = None

    # Erev days: the calendar day before a yom tov, whether or not that yom tov
    # itself fell on a trading day.
    for timestamp in index:
        if timestamp in labelled:
            continue
        tomorrow = timestamp.date() + dt.timedelta(days=1)
        key = _hebrew_key(tomorrow)
        if key in YOM_TOV and not YOM_TOV[key].endswith((" II", " VIII")):
            records.append((timestamp, "Erev " + YOM_TOV[key], "erev"))

    frame = pd.DataFrame(records, columns=["date", "holiday", "category"])
    return frame.sort_values("date").reset_index(drop=True)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (Anonymous / Meeus algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


def secular_holidays(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Non-Jewish days that are culturally marked but leave the NYSE open."""
    years = range(index.min().year, index.max().year + 1)
    by_date: dict[dt.date, str] = {}
    for year in years:
        candidates = {
            dt.date(year, 2, 14): "Valentine's Day",
            dt.date(year, 3, 17): "St. Patrick's Day",
            dt.date(year, 4, 15): "Tax Day",
            dt.date(year, 5, 5): "Cinco de Mayo",
            dt.date(year, 10, 31): "Halloween",
            dt.date(year, 11, 11): "Veterans Day",
            dt.date(year, 12, 24): "Christmas Eve",
            _nth_weekday(year, 10, 0, 2): "Columbus Day",
            _easter(year) + dt.timedelta(days=1): "Easter Monday",
            _nth_weekday(year, 11, 3, 4) + dt.timedelta(days=1): "Day after Thanksgiving",
        }
        by_date.update(candidates)

    trading = set(index)
    records = [
        (pd.Timestamp(date), name, "secular")
        for date, name in by_date.items()
        if pd.Timestamp(date) in trading
    ]
    frame = pd.DataFrame(records, columns=["date", "holiday", "category"])
    return frame.sort_values("date").reset_index(drop=True)


def pre_holiday(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sessions immediately before an NYSE closure (the classic holiday effect).

    A closure is inferred from the calendar itself: a weekday gap in the trading
    index. This picks up the market's own holidays without hardcoding a list
    that has changed over a century.
    """
    ordered = pd.DatetimeIndex(sorted(index))
    records = []
    for current, following in zip(ordered[:-1], ordered[1:]):
        gap = pd.bdate_range(current + pd.Timedelta(days=1), following - pd.Timedelta(days=1))
        if len(gap) > 0:
            records.append((current, "Pre-holiday", "pre_closure"))
    frame = pd.DataFrame(records, columns=["date", "holiday", "category"])
    return frame.sort_values("date").reset_index(drop=True)
