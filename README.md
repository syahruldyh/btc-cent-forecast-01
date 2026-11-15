# btc-cent-forecast-01

[![make.fun · btc-cent-forecast-01](https://make.fun/badge/btc-cent-forecast-01.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01) [![rank](https://make.fun/badge/btc-cent-forecast-01/syahruldyh/btc-cent-forecast-01/rank.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01) [![reward](https://make.fun/badge/btc-cent-forecast-01/syahruldyh/btc-cent-forecast-01/reward.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)

Entry for [make.fun/bounty/btc-cent-forecast-01](https://make.fun/bounty/btc-cent-forecast-01): forecast Bitcoin's US-dollar price 4 hours ahead, 12 times over 48 hours. The program reads 4,320 one-minute BTC closes (72 hours) up to a checkpoint and predicts the close 240 minutes later. Score is the mean absolute error in cents — lower is better.

## Method

Walk-forward LAD-ridge ensemble, built at runtime from each case's own history (no pre-trained model, no external data):

- 12 features per time step: multi-horizon returns (5m, 15m, 60m, 240m, 1440m), distance to 4h/24h moving averages and 12h median, short/long volatility, 12h OLS slope, 240m range.
- ~176 walk-forward training samples per case (every 15 minutes, no lookahead), target = 240-minute forward return.
- 5 LAD-ridge models fit by IRLS (the absolute-error objective matches the MAE score, and is robust to BTC's one-minute jumps): λ = 0.1 / 1 / 10 on all features, λ = 1 on momentum features only, λ = 1 on mean-reversion features only.
- Ensemble prediction = median of the 5 models, shrunk 0.85 toward persistence (the last price), clamped to the price grid.

Backtest on the kit's practice cases: **MAE $325.38 vs persistence baseline $470.55** (5-case). Rolling validation on the reconstructed series: **$396.76 vs persistence $530.32**, model beats persistence on 3 of 4 checkpoints.

## Run

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out` per case: `{"forecasts": [{"case": "<id>", "price": <number>}]}`. Python 3.13, standard library only, deterministic, ~0.5 s per case.

## Referral

Invite code: `4cebaz6p` — https://make.fun/r/4cebaz6p
