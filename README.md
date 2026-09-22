# holidayTrading

Does it pay to trade on holidays the stock market stays open for — Jewish holidays
in particular?

**Findings: [REPORT.md](REPORT.md)**

Short version: no return edge survives correction for multiple testing, but trading
volume on Yom Kippur and Rosh Hashanah is 11–15% below comparable sessions, a drop
larger than Columbus Day or Veterans Day. It is an execution fact, not a signal.

![Volume and returns on open holidays](figures/holiday_volume_vs_returns.png)

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m holiday_trading.report      # rebuilds every table in results/
PYTHONPATH=src python -m holiday_trading.figures     # rebuilds the figure
```

`--refresh` re-downloads prices from Yahoo Finance; otherwise the cached copy in
`data/` is used, so results are reproducible without network access.

## Layout

| Path | What it holds |
|---|---|
| `src/holiday_trading/data.py` | Yahoo Finance download and CSV cache |
| `src/holiday_trading/calendars.py` | Hebrew-calendar holidays, secular open holidays, pre-closure sessions |
| `src/holiday_trading/analysis.py` | Matched-control randomization test, FDR correction |
| `src/holiday_trading/report.py` | Rebuilds every table in `results/` |
| `src/holiday_trading/figures.py` | Rebuilds the figure |
| `data/gspc_daily.csv` | S&P 500 daily OHLCV, 1927-12-30 to 2026-09-21 |
| `results/*.csv` | Generated result tables |

## Method in one paragraph

Jewish holidays cluster in September/October and March/April, and the Hebrew
calendar's postponement rules mean Yom Kippur can only ever fall on a Monday,
Wednesday or Thursday. Comparing holidays against "all other days" therefore measures
the season and the weekday, not the holiday. Every test here instead draws controls
from non-holiday sessions in the **same calendar month and weekday in other years**,
and builds the null distribution by redrawing 20,000 times. Holidays are classified
by whether work is forbidden (*yom tov*), permitted but commonly taken off
(*chol hamoed*), or ordinary (*minor*) — so observance strength, not just the
calendar, can be read off the results. Because the study runs 32 holiday tests at
once, p-values are reported both raw and Benjamini-Hochberg corrected.

## Caveats

The S&P 500 before 1957 is a backfilled reconstruction; volume is NYSE composite
volume and starts in 1950; returns exclude dividends. The sample is small in the way
that matters — 68 Yom Kippurs — so the smallest detectable effect is about 40 bp/day.
"No effect" here means "nothing large enough to trade", not "exactly zero".
