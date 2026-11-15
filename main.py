# BTC 4-hour-ahead forecaster: walk-forward LAD-ridge ensemble, median, shrunk to persistence.
# Stdlib only, deterministic, no network. Reads every *.in in the input dir and writes one *.out
# per case with {"forecasts": [{"case": id, "price": number}]}.
import json
import math
import os
import sys
from pathlib import Path

HORIZON = 240
SHRINK = 0.85
EPS = 1e-6


def clamp_price(x):
    if not (x == x) or x in (float("inf"), float("-inf")):
        return None
    x = max(0.00001, min(9999999.99999, x))
    return round(x, 5)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def solve_linear(a, b):
    # Gaussian elimination with partial pivoting; a is n x n, b is length n.
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            m[piv][col] = 1e-12
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / pv
            if f != 0.0:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / m[i][i]
    return x


def fit_lad_ridge(xs, ys, lam, max_iter=50):
    # IRLS for min sum |y - x.b| + lam * ||b[1:]||^2. xs rows include intercept as x[0]=1.
    n = len(xs)
    k = len(xs[0])
    beta = [0.0] * k
    # init with plain ridge (weights all 1)
    for _ in range(max_iter):
        resid = [ys[i] - sum(xs[i][j] * beta[j] for j in range(k)) for i in range(n)]
        w = [1.0 / max(abs(r), EPS) for r in resid]
        a = [[0.0] * k for _ in range(k)]
        b = [0.0] * k
        for i in range(n):
            wi = w[i]
            xi = xs[i]
            yi = ys[i]
            for j in range(k):
                b[j] += wi * xi[j] * yi
                xj = xi[j]
                for l in range(j, k):
                    a[j][l] += wi * xj * xi[l]
        for j in range(k):
            for l in range(j):
                a[j][l] = a[l][j]
        for j in range(1, k):
            a[j][j] += lam
        new_beta = solve_linear(a, b)
        delta = max(abs(new_beta[j] - beta[j]) for j in range(k))
        beta = new_beta
        if delta < 1e-8:
            break
    return beta


def features_at(prices, t):
    # Features from data <= t only. prices are closes; index t is the t-th minute (0-based).
    p = prices[t]
    if p == 0:
        return None

    def ret(lag):
        return (p - prices[t - lag]) / p if t - lag >= 0 else 0.0

    def ma(win):
        return mean(prices[t - win + 1:t + 1]) if t - win + 1 >= 0 else p

    def med(win):
        return median(prices[t - win + 1:t + 1]) if t - win + 1 >= 0 else p

    def vol(win):
        if t - win + 1 < 1:
            return 0.0
        rets = [(prices[i] - prices[i - 1]) / prices[i - 1] if prices[i - 1] != 0 else 0.0
                for i in range(t - win + 1, t + 1)]
        return std(rets)

    def slope(win):
        if t - win + 1 < 0:
            return 0.0
        ys = prices[t - win + 1:t + 1]
        n = len(ys)
        mx = (n - 1) / 2.0
        my = mean(ys)
        num = sum((i - mx) * (ys[i] - my) for i in range(n))
        den = sum((i - mx) ** 2 for i in range(n))
        s = num / den if den != 0 else 0.0
        return s * HORIZON / p

    window = prices[t - 239:t + 1] if t - 239 >= 0 else prices[:t + 1]
    rng = (max(window) - min(window)) / p if window else 0.0
    return [
        ret(5), ret(15), ret(60), ret(240), ret(1440),
        (p - ma(240)) / p, (p - ma(1440)) / p, (p - med(720)) / p,
        vol(240), vol(1440), slope(720), rng,
    ]


# Feature index groups: momentum (ret_5, ret_15, ret_60, ret_240, ret_1440, slope_720)
MOMENTUM = [0, 1, 2, 3, 4, 10]
# mean-reversion (dist_ma240, dist_ma1440, dist_med720, vol_240, vol_1440)
REVERSION = [5, 6, 7, 8, 9]
ALL12 = list(range(12))


def build_samples(prices):
    n = len(prices)
    samples = []
    t = 1440
    # 0-based: last sample's target index t + 240 must stay inside the 4320 closes.
    while t <= n - HORIZON - 1:
        feats = features_at(prices, t)
        if feats is not None:
            y = (prices[t + HORIZON] - prices[t]) / prices[t] if prices[t] != 0 else 0.0
            samples.append((feats, y))
        t += 15
    return samples


def standardize(samples, idxs):
    cols = [[s[0][i] for s in samples] for i in idxs]
    mus = [mean(c) for c in cols]
    sds = [std(c) or 1.0 for c in cols]
    return mus, sds


def predict_case(prices):
    last = prices[-1]
    samples = build_samples(prices)
    if len(samples) < 30:
        return last
    ckpt = features_at(prices, len(prices) - 1)
    if ckpt is None:
        return last
    members = [
        (0.1, ALL12),
        (1.0, ALL12),
        (10.0, ALL12),
        (1.0, MOMENTUM),
        (1.0, REVERSION),
    ]
    preds = []
    for lam, idxs in members:
        mus, sds = standardize(samples, idxs)
        xs = [[1.0] + [(s[0][i] - mus[j]) / sds[j] for j, i in enumerate(idxs)] for s in samples]
        ys = [s[1] for s in samples]
        beta = fit_lad_ridge(xs, ys, lam)
        xf = [1.0] + [(ckpt[i] - mus[j]) / sds[j] for j, i in enumerate(idxs)]
        y_pred = sum(xf[j] * beta[j] for j in range(len(beta)))
        preds.append(last * (1.0 + y_pred))
    med = median(preds)
    guess = last + SHRINK * (med - last)
    out = clamp_price(guess)
    return out if out is not None else last


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.in"))
    for f in files:
        case_id = f.stem
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            prices = data["markets"][0]["prices"]
            prices = [float(x) for x in prices]
            if len(prices) < 1500:
                raise ValueError("too few prices")
            price = predict_case(prices)
        except Exception:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                price = clamp_price(float(data["markets"][0]["prices"][-1]))
            except Exception:
                continue
            if price is None:
                continue
        out = {"forecasts": [{"case": case_id, "price": price}]}
        (out_dir / (case_id + ".out")).write_text(json.dumps(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
