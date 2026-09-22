# Manning vs. Stockton — contextual QB impact

Does a boom-bust quarterback beat a steady one? Compares **Arch Manning** (Texas) and
**Gunner Stockton** (Georgia) across 2025 and 2026, adjusting for the things a
quarterback does not control.

**[Interactive dashboard →](https://claude.ai/artifact/Gzvq7Jj4bvmZVQgwpc6rw9)**

## Headline

In 2025 the two produced nearly the same value per dropback (+0.077 vs +0.175 EPA, a gap
well inside this sample's detection limit). The *shape* differed sharply, and the shape is
what decides games.

| 2025, dropbacks, FCS removed | Manning | Stockton |
|---|---|---|
| Mean EPA/dropback | +0.077 | +0.175 |
| SD | 1.764 | 1.582 |
| Bad-play rate (EPA < −0.5) | 45.1% | 39.0% |
| 95th-percentile play | **+3.42** | +2.91 |
| Highest-leverage quartile | **−0.133** | **+0.655** |
| Clutch permutation test | p = 0.958 | **p = 0.010** |

Manning's brilliance is real and worth **+0.048** EPA/dropback. His floor costs
**−0.146**. Across 2,376 real comeback attempts (2022–2026), tail weight does **not**
predict completing a comeback once average efficiency is held constant (odds ratio 0.85,
95% CI 0.60–1.21).

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m qbcompare.report     # rebuilds every table in results/
```

Play-by-play parquet files are downloaded once into `data/` from the
[cfbfastR bulk release](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfbfastR_cfb_pbp)
(no API key needed). `data/wp_train.parquet` and `data/wp_oof.npy` are produced by
`qbcompare.wpmodel` and cached.

## Layout

| Path | Purpose |
|---|---|
| `src/qbcompare/ingest.py` | Attribute plays to a QB from `play_text` |
| `src/qbcompare/panel.py` | Assemble the play panel with context |
| `src/qbcompare/wpmodel.py` | Win-probability model and league leverage table |
| `src/qbcompare/distribution.py` | Shift function, tails, floor, power |
| `src/qbcompare/leverage.py` | WPA decomposition, clutch permutation test |
| `src/qbcompare/comeback.py` | Comeback-capability model, 2022–2026 |
| `src/qbcompare/report.py` | Regenerates everything in `results/` |
| `site/index.html` | The published dashboard |

## Three data problems worth knowing about

**The bulk feed's win-probability columns are corrupted.** Routine plays — a punt, a
four-yard run — swing home win probability by more than 30 points on 5.8% of 2025 snaps
and 12.7% of 2026 snaps, and timeouts carry non-zero EPA. Its **EPA column is sound** (play
types order correctly, league mean is sane) and is used as shipped. Win probability is
instead modelled here from game states across 370,022 plays, calibrated out of fold by
game: Brier 0.154 against a 0.250 baseline, max calibration error 0.017.

**`passer_player_name` is mangled** on ~65% of rows when play text uses the verbose scorer
format, so attribution is parsed from `play_text` with the QB required to be in the *actor*
slot. Validated against official lines: Manning 404 attempts (official 404), Stockton 385
(386), sacks exact for both.

**`id_play` is stored as a float**, so 18-digit identifiers lose precision and 293,200 rows
collapse to 230,529 distinct values. Joining on it silently inflated Manning's 2025
dropbacks from 426 to 636. Context is attached before extraction, never merged on that key.

## Limits

- **Receiver drops are in no free source** — zero occurrences across 375,000 plays. PFF
  reports 21 drops for Manning in 2025 (4th-most in the SEC) but publishes no Georgia
  equivalent, so the adjustment cannot be applied symmetrically and is not applied at all.
- **Adjusted completion % is not computable** — every component of its denominator (drops,
  throwaways, batted balls, spikes) is absent.
- **Pressure is a lower bound.** Scorer-credited "QB hurried" fires on ~3% of plays and
  varies by venue; PFF charts true pressure near 35% of dropbacks.
- **QB and team are collinear.** Each took nearly all his team's snaps, so nothing here
  separates the quarterback from his line, scheme and receivers.
- **Minimum detectable effect is 0.476 EPA/play** after game clustering. Differences
  smaller than that are described, not inferred.
