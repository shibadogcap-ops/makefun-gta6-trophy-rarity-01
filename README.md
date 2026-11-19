# makefun-gta6-trophy-rarity-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/shibadogcap-ops/makefun-gta6-trophy-rarity-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/shibadogcap-ops/makefun-gta6-trophy-rarity-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic forecaster for the earn rate of every Grand Theft Auto VI PlayStation 5 trophy on Exophase at a future target time. Given the house's recording of the trophy page up to a checkpoint — each read's players-tracked count and per-trophy earn rates — it collapses the reads into distinct page states, forecasts the tracked-player count with log-time fits, and for each trophy blends a logit-scale drift model with an earner-count model (per-trophy least-squares slopes against log-time, shrunk toward tier means, the blended move damped toward persistence). Quantiles are built on the logit scale with a horizon-anchored spread, and structural constraints are applied at the end: story trophies stay ordered along the mission chain, and the platinum never exceeds the rarest other trophy.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` reads every `*.in` case file in the input directory and writes one `<id>.out` forecast per case to the output directory: for each trophy, the 0.05, 0.25, 0.5, 0.75 and 0.95 quantiles of its earn rate at the case's target. Python 3.13, standard library only, no network.
