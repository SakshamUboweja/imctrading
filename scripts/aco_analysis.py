"""
ACO (ASH_COATED_OSMIUM) Deep Statistical Analysis
Exhaustive analysis following the CLAUDE.md pipeline.
Uses numpy/pandas (fine for analysis, not for submission).
"""

import os
import csv
import math
import json
import statistics
from collections import defaultdict

DATA_DIR = "/Users/saksham/Desktop/imctrading/data/round2"
DATA_100K = "/Users/saksham/Desktop/imctrading/data_100k/round2"
PRODUCT = "ASH_COATED_OSMIUM"
IPR = "INTARIAN_PEPPER_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(data_dir, days=(-1, 0, 1)):
    rows = []
    for day in days:
        path = os.path.join(data_dir, f"prices_round_2_day_{day}.csv")
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if row['product'] == PRODUCT:
                    row['day'] = int(row.get('day', day))
                    rows.append(row)
    return rows

def load_prices_all(data_dir, days=(-1, 0, 1)):
    """Load all products."""
    rows = []
    for day in days:
        path = os.path.join(data_dir, f"prices_round_2_day_{day}.csv")
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                row['_day'] = int(day)
                rows.append(row)
    return rows

def load_trades(data_dir, days=(-1, 0, 1)):
    rows = []
    for day in days:
        path = os.path.join(data_dir, f"trades_round_2_day_{day}.csv")
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if row['symbol'] == PRODUCT:
                    row['_day'] = int(day)
                    rows.append(row)
    return rows

def load_trades_all(data_dir, days=(-1, 0, 1)):
    rows = []
    for day in days:
        path = os.path.join(data_dir, f"trades_round_2_day_{day}.csv")
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                row['_day'] = int(day)
                rows.append(row)
    return rows

def safe_float(v, default=None):
    try: return float(v)
    except: return default

def safe_int(v, default=None):
    try: return int(v)
    except: return default

print("=" * 70)
print("ACO DEEP STATISTICAL ANALYSIS")
print("=" * 70)

prices_rows = load_prices(DATA_DIR)
trades_rows = load_trades(DATA_DIR)
all_prices_rows = load_prices_all(DATA_DIR)

print(f"\n[DATA] ACO price rows: {len(prices_rows)} across 3 days")
print(f"[DATA] ACO trade rows: {len(trades_rows)} across 3 days")

# Parse into per-day lists
days_data = {}
for day in (-1, 0, 1):
    day_rows = [r for r in prices_rows if int(r['day']) == day]
    day_rows_sorted = sorted(day_rows, key=lambda r: int(r['timestamp']))

    ts = [int(r['timestamp']) for r in day_rows_sorted]
    mid = [safe_float(r['mid_price']) for r in day_rows_sorted]

    bp1 = [safe_float(r['bid_price_1']) for r in day_rows_sorted]
    bv1 = [safe_float(r['bid_volume_1'], 0) for r in day_rows_sorted]
    bp2 = [safe_float(r['bid_price_2']) for r in day_rows_sorted]
    bv2 = [safe_float(r['bid_volume_2'], 0) for r in day_rows_sorted]
    bp3 = [safe_float(r['bid_price_3']) for r in day_rows_sorted]
    bv3 = [safe_float(r['bid_volume_3'], 0) for r in day_rows_sorted]

    ap1 = [safe_float(r['ask_price_1']) for r in day_rows_sorted]
    av1 = [safe_float(r['ask_volume_1'], 0) for r in day_rows_sorted]
    ap2 = [safe_float(r['ask_price_2']) for r in day_rows_sorted]
    av2 = [safe_float(r['ask_volume_2'], 0) for r in day_rows_sorted]
    ap3 = [safe_float(r['ask_price_3']) for r in day_rows_sorted]
    av3 = [safe_float(r['ask_volume_3'], 0) for r in day_rows_sorted]

    days_data[day] = {
        'ts': ts, 'mid': mid,
        'bp1': bp1, 'bv1': bv1, 'bp2': bp2, 'bv2': bv2, 'bp3': bp3, 'bv3': bv3,
        'ap1': ap1, 'av1': av1, 'ap2': ap2, 'av2': av2, 'ap3': ap3, 'av3': av3,
        'rows': day_rows_sorted
    }

# All-day combined
all_mid = []
all_ts_abs = []
all_ts = []
for day in (-1, 0, 1):
    d = days_data[day]
    all_mid.extend(d['mid'])
    # absolute timestamp across days
    all_ts_abs.extend([t + (day + 1) * 1_000_000 for t in d['ts']])
    all_ts.extend(d['ts'])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Price Dynamics
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 1: PRICE DYNAMICS")
print("=" * 70)

def mean(lst): return sum(lst) / len(lst) if lst else 0
def variance(lst):
    m = mean(lst)
    return sum((x - m) ** 2 for x in lst) / len(lst) if lst else 0
def std(lst): return math.sqrt(variance(lst))
def skewness(lst):
    m = mean(lst)
    s = std(lst)
    if s == 0: return 0
    return sum(((x - m)/s)**3 for x in lst) / len(lst)
def kurtosis(lst):
    m = mean(lst)
    s = std(lst)
    if s == 0: return 0
    return sum(((x - m)/s)**4 for x in lst) / len(lst) - 3

def linreg(xs, ys):
    n = len(xs)
    if n < 2: return 0, 0, 0
    xm, ym = mean(xs), mean(ys)
    sxx = sum((x - xm)**2 for x in xs)
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    if sxx == 0: return 0, ym, 0
    slope = sxy / sxx
    intercept = ym - slope * xm
    y_pred = [slope * x + intercept for x in xs]
    ss_res = sum((y - yp)**2 for y, yp in zip(ys, y_pred))
    ss_tot = sum((y - ym)**2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return slope, intercept, r2

for day in (-1, 0, 1):
    d = days_data[day]
    m = [x for x in d['mid'] if x is not None]
    ts = d['ts']

    slope, intercept, r2 = linreg(ts, m)
    returns = [m[i] - m[i-1] for i in range(1, len(m))]

    print(f"\n  Day {day}:")
    print(f"    N={len(m)}, mean={mean(m):.3f}, std={std(m):.3f}")
    print(f"    min={min(m):.1f}, max={max(m):.1f}, range={max(m)-min(m):.1f}")
    print(f"    start={m[0]:.1f}, end={m[-1]:.1f}, drift={m[-1]-m[0]:.1f}")
    print(f"    skew={skewness(m):.4f}, kurt={kurtosis(m):.4f}")
    print(f"    linreg: slope={slope:.8f}, intercept={intercept:.2f}, R²={r2:.6f}")
    print(f"    returns: mean={mean(returns):.5f}, std={std(returns):.4f}, skew={skewness(returns):.4f}")

# Check polynomial fit (quadratic) across full data
mid_valid = [(t, m) for t, m in zip(all_ts_abs, all_mid) if m is not None]
ts_q = [x[0] for x in mid_valid]
ys_q = [x[1] for x in mid_valid]
slope_all, intercept_all, r2_all = linreg(ts_q, ys_q)
print(f"\n  All-days linear: slope={slope_all:.10f}, intercept={intercept_all:.3f}, R²={r2_all:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Stationarity & Memory
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 2: STATIONARITY & MEMORY")
print("=" * 70)

def autocorr(lst, lag):
    if lag >= len(lst): return 0
    m = mean(lst)
    var = variance(lst)
    if var == 0: return 0
    n = len(lst)
    cov = sum((lst[i] - m) * (lst[i - lag] - m) for i in range(lag, n)) / n
    return cov / var

def hurst_exponent(ts, max_lag=100):
    """Estimate Hurst exponent via R/S analysis."""
    lags = [2, 4, 8, 16, 32, 64, 100]
    rs_vals = []
    valid_lags = []
    for lag in lags:
        if lag >= len(ts) // 2: continue
        n_chunks = len(ts) // lag
        rs_list = []
        for c in range(n_chunks):
            chunk = ts[c*lag:(c+1)*lag]
            m = mean(chunk)
            mean_adj = [x - m for x in chunk]
            cum = []
            running = 0
            for v in mean_adj:
                running += v
                cum.append(running)
            R = max(cum) - min(cum)
            S = std(chunk)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_vals.append(math.log(mean(rs_list)))
            valid_lags.append(math.log(lag))
    if len(valid_lags) < 2: return 0.5
    slope, _, _ = linreg(valid_lags, rs_vals)
    return slope

def adf_approx(ts, maxlag=10):
    """Simplified ADF test (regression-based approximation)."""
    if len(ts) < maxlag + 3: return 0, 1
    y = ts[maxlag:]
    diff = [ts[i] - ts[i-1] for i in range(1, len(ts))]
    # ADF: delta_y = gamma * y_{t-1} + sum of lagged deltas
    # We do a simple version: regress diff[i] on ts[i-1]
    y_lag = ts[1:len(diff)+1]
    d_y = diff
    # just get the slope (gamma) and its t-stat approximation
    n = len(d_y)
    m1, m2 = mean(y_lag), mean(d_y)
    sxx = sum((x - m1)**2 for x in y_lag)
    sxy = sum((x - m1) * (y - m2) for x, y in zip(y_lag, d_y))
    if sxx == 0: return 0, 1
    gamma = sxy / sxx
    y_pred = [gamma * x + (m2 - gamma * m1) for x in y_lag]
    se2 = sum((d - p)**2 for d, p in zip(d_y, y_pred)) / (n - 2)
    se_gamma = math.sqrt(se2 / sxx)
    t_stat = gamma / se_gamma if se_gamma > 0 else 0
    # Rough p-value: t < -3.5 is essentially 0; t > -1.9 is ~0.05+
    return t_stat, gamma

def half_life(ts):
    """OU half-life via regression of delta on level."""
    m = mean(ts)
    diffs = [ts[i] - ts[i-1] for i in range(1, len(ts))]
    lags = [ts[i-1] - m for i in range(1, len(ts))]
    sxx = sum(x**2 for x in lags)
    sxy = sum(x * y for x, y in zip(lags, diffs))
    if sxx == 0: return float('inf')
    phi = sxy / sxx
    if phi >= 0: return float('inf')
    return -math.log(2) / phi

for day in (-1, 0, 1):
    d = days_data[day]
    m = [x for x in d['mid'] if x is not None]

    # Autocorrelations
    lags = [1, 2, 3, 5, 10, 20, 50, 100]
    acf_vals = [autocorr(m, lag) for lag in lags]

    # Returns autocorr
    ret = [m[i] - m[i-1] for i in range(1, len(m))]
    ret_acf = [autocorr(ret, lag) for lag in lags]

    H = hurst_exponent(m)
    t_stat, gamma = adf_approx(m)
    hl = half_life(m)

    print(f"\n  Day {day}:")
    print(f"    Hurst exponent: {H:.4f}  ({'TRENDING' if H > 0.55 else 'MEAN-REVERTING' if H < 0.45 else 'RANDOM WALK'})")
    print(f"    ADF t-stat: {t_stat:.4f}, gamma: {gamma:.6f}  ({'STATIONARY' if t_stat < -3.5 else 'likely stationary' if t_stat < -2.86 else 'NON-STATIONARY'})")
    print(f"    Half-life of mean reversion: {hl:.1f} ticks")
    print(f"    Price ACF at lags {lags}:")
    print(f"      {[round(a, 4) for a in acf_vals]}")
    print(f"    Return ACF at lags {lags}:")
    print(f"      {[round(a, 4) for a in ret_acf]}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Order Book Microstructure & Wall Detection
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 3: ORDER BOOK MICROSTRUCTURE & WALL DETECTION")
print("=" * 70)

def compute_wall_stats(day_data, wall_min_vol=8):
    d = day_data
    n = len(d['ts'])

    spreads_l1 = []
    wall_bids = []
    wall_asks = []
    wall_mids = []
    simple_mids = []
    wall_spreads = []

    bv1_vals = []
    av1_vals = []

    for i in range(n):
        bp1, bv1 = d['bp1'][i], d['bv1'][i]
        bp2, bv2 = d['bp2'][i], d['bv2'][i]
        bp3, bv3 = d['bp3'][i], d['bv3'][i]
        ap1, av1 = d['ap1'][i], d['av1'][i]
        ap2, av2 = d['ap2'][i], d['av2'][i]
        ap3, av3 = d['ap3'][i], d['av3'][i]
        mid = d['mid'][i]

        if bp1 and ap1:
            spreads_l1.append(ap1 - bp1)
            simple_mids.append((bp1 + ap1) / 2.0)

        if bv1: bv1_vals.append(bv1)
        if av1: av1_vals.append(av1)

        # Wall bid: highest volume bid
        best_bid_vol, best_bid_p = 0, None
        for p, v in [(bp1, bv1), (bp2, bv2), (bp3, bv3)]:
            if p and v and v >= wall_min_vol and v > best_bid_vol:
                best_bid_vol, best_bid_p = v, p

        # Wall ask: highest volume ask
        best_ask_vol, best_ask_p = 0, None
        for p, v in [(ap1, av1), (ap2, av2), (ap3, av3)]:
            if p and v and v >= wall_min_vol and v > best_ask_vol:
                best_ask_vol, best_ask_p = v, p

        if best_bid_p and best_ask_p:
            wall_bids.append(best_bid_p)
            wall_asks.append(best_ask_p)
            wall_mids.append((best_bid_p + best_ask_p) / 2.0)
            wall_spreads.append(best_ask_p - best_bid_p)

    return {
        'spreads_l1': spreads_l1,
        'wall_bids': wall_bids,
        'wall_asks': wall_asks,
        'wall_mids': wall_mids,
        'wall_spreads': wall_spreads,
        'simple_mids': simple_mids,
        'bv1_vals': bv1_vals,
        'av1_vals': av1_vals,
    }

for day in (-1, 0, 1):
    d = days_data[day]
    ws = compute_wall_stats(d)
    ts = d['ts']
    n = len(ts)

    spreads = ws['spreads_l1']
    wm = ws['wall_mids']
    sm = ws['simple_mids']
    wb = ws['wall_bids']
    wa = ws['wall_asks']
    wsp = ws['wall_spreads']

    # Residual stds: wall_mid vs time
    ts_short = ts[:len(wm)]
    wm_slope, wm_int, wm_r2 = linreg(ts_short, wm)
    sm_slope, sm_int, sm_r2 = linreg(ts_short[:len(sm)], sm[:len(sm)])

    wm_resid = [m - (wm_slope * t + wm_int) for m, t in zip(wm, ts_short)]
    sm_resid = [m - (sm_slope * t + sm_int) for m, t in zip(sm[:len(sm)], ts_short[:len(sm)])]

    # Check wall_bid + wall_ask constant?
    wb_plus_wa = [b + a for b, a in zip(wb, wa)]

    # Volume distribution analysis
    all_bid_vols = []
    all_ask_vols = []
    for i in range(n):
        for v_key in ['bv1', 'bv2', 'bv3']:
            v = d[v_key][i]
            if v: all_bid_vols.append(v)
        for v_key in ['av1', 'av2', 'av3']:
            v = d[v_key][i]
            if v: all_ask_vols.append(v)

    print(f"\n  Day {day}:")
    print(f"    L1 spread: mean={mean(spreads):.3f}, std={std(spreads):.3f}, "
          f"min={min(spreads):.0f}, max={max(spreads):.0f}")

    # Spread distribution
    spread_counts = defaultdict(int)
    for s in spreads: spread_counts[int(s)] += 1
    print(f"    L1 spread distribution: {dict(sorted(spread_counts.items()))}")

    print(f"    Wall coverage: {len(wm)}/{n} ({100*len(wm)/n:.1f}%)")
    print(f"    Wall mid resid std: {std(wm_resid):.4f} ticks")
    print(f"    Simple mid resid std: {std(sm_resid):.4f} ticks")
    print(f"    Ratio (wall/simple): {std(wm_resid)/std(sm_resid):.4f} "
          f"({'wall more precise' if std(wm_resid) < std(sm_resid) else 'simple more precise'})")

    print(f"    Wall spread: mean={mean(wsp):.3f}, std={std(wsp):.4f}, "
          f"is_constant={std(wsp) < 0.5}")

    # Wall bid+ask sum
    print(f"    Wall_bid+wall_ask: mean={mean(wb_plus_wa):.3f}, std={std(wb_plus_wa):.4f}")

    # Unique wall spreads
    unique_wsp = sorted(set(wsp))
    print(f"    Unique wall spreads seen: {unique_wsp[:20]}")

    if all_bid_vols:
        print(f"    Bid volume distribution: mean={mean(all_bid_vols):.1f}, "
              f"std={std(all_bid_vols):.1f}, max={max(all_bid_vols)}")
        vol_counts = sorted(set(all_bid_vols))
        print(f"    Common bid vol sizes: {vol_counts[:20]}")

    # Imbalance
    imb_vals = []
    for i in range(n):
        bv = d['bv1'][i]
        av = d['av1'][i]
        if bv and av and bv + av > 0:
            imb_vals.append((bv - av) / (bv + av))
    if imb_vals:
        print(f"    L1 imbalance: mean={mean(imb_vals):.4f}, std={std(imb_vals):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Return Predictability (Signal Analysis)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 4: RETURN PREDICTABILITY")
print("=" * 70)

def corr(xs, ys):
    """Pearson correlation."""
    n = len(xs)
    if n < 2: return 0
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx)**2 for x in xs)
    syy = sum((y - my)**2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else 0

def compute_signals_and_preds(day_data, wall_min_vol=8):
    d = day_data
    n = len(d['ts'])

    mids = d['mid']

    signals = {
        'imbalance_l1': [],
        'imbalance_all': [],
        'wall_dev': [],          # mid - wall_mid
        'bid_vol_l1': [],
        'ask_vol_l1': [],
        'spread_l1': [],
        'momentum_5': [],
        'momentum_10': [],
        'momentum_20': [],
        'momentum_50': [],
        'wall_bid_l1_diff': [],  # wall_bid vs l1_bid
        'wall_ask_l1_diff': [],  # wall_ask vs l1_ask
        'total_bid_vol': [],
        'total_ask_vol': [],
        'vol_imbalance_all': [],
        'price_level': [],       # mid - 10000
    }

    wall_mids = []
    simple_mids = []

    for i in range(n):
        bp1, bv1 = d['bp1'][i], d['bv1'][i]
        bp2, bv2 = d['bp2'][i], d['bv2'][i]
        bp3, bv3 = d['bp3'][i], d['bv3'][i]
        ap1, av1 = d['ap1'][i], d['av1'][i]
        ap2, av2 = d['ap2'][i], d['av2'][i]
        ap3, av3 = d['ap3'][i], d['av3'][i]
        mid = mids[i]

        # L1 imbalance
        imb_l1 = None
        if bv1 and av1 and bv1 + av1 > 0:
            imb_l1 = (bv1 - av1) / (bv1 + av1)

        # All-levels imbalance
        total_bv = (bv1 or 0) + (bv2 or 0) + (bv3 or 0)
        total_av = (av1 or 0) + (av2 or 0) + (av3 or 0)
        imb_all = (total_bv - total_av) / (total_bv + total_av) if (total_bv + total_av) > 0 else None

        # Wall detection
        best_bid_vol, best_bid_p = 0, None
        for p, v in [(bp1, bv1), (bp2, bv2), (bp3, bv3)]:
            if p and v and v >= wall_min_vol and v > best_bid_vol:
                best_bid_vol, best_bid_p = v, p
        best_ask_vol, best_ask_p = 0, None
        for p, v in [(ap1, av1), (ap2, av2), (ap3, av3)]:
            if p and v and v >= wall_min_vol and v > best_ask_vol:
                best_ask_vol, best_ask_p = v, p

        wm = None
        if best_bid_p and best_ask_p:
            wm = (best_bid_p + best_ask_p) / 2.0
        wall_mids.append(wm)

        sm = None
        if bp1 and ap1:
            sm = (bp1 + ap1) / 2.0
        simple_mids.append(sm)

        signals['imbalance_l1'].append(imb_l1)
        signals['imbalance_all'].append(imb_all)
        signals['wall_dev'].append(mid - wm if mid and wm else None)
        signals['bid_vol_l1'].append(bv1)
        signals['ask_vol_l1'].append(av1)
        signals['spread_l1'].append(ap1 - bp1 if ap1 and bp1 else None)
        signals['total_bid_vol'].append(total_bv)
        signals['total_ask_vol'].append(total_av)
        signals['vol_imbalance_all'].append(imb_all)
        signals['price_level'].append(mid - 10000 if mid else None)

        # Wall vs L1 diff
        wb_l1_diff = best_bid_p - bp1 if best_bid_p and bp1 else None
        wa_l1_diff = ap1 - best_ask_p if best_ask_p and ap1 else None
        signals['wall_bid_l1_diff'].append(wb_l1_diff)
        signals['wall_ask_l1_diff'].append(wa_l1_diff)

        # Momentum
        for key, lag in [('momentum_5', 5), ('momentum_10', 10), ('momentum_20', 20), ('momentum_50', 50)]:
            if i >= lag and mids[i] and mids[i-lag]:
                signals[key].append(mids[i] - mids[i-lag])
            else:
                signals[key].append(None)

    # Forward returns at multiple horizons
    horizons = [1, 3, 5, 10, 20, 50]

    # For each signal, compute correlation with forward returns at each horizon
    results = {}
    for sig_name, sig_vals in signals.items():
        results[sig_name] = {}
        for h in horizons:
            # Align signal with forward return at horizon h
            pairs = []
            for i in range(n - h):
                s = sig_vals[i]
                m_curr = mids[i]
                m_fut = mids[i + h]
                if s is not None and m_curr is not None and m_fut is not None:
                    pairs.append((s, m_fut - m_curr))
            if len(pairs) < 50:
                results[sig_name][h] = None
                continue
            xs, ys = zip(*pairs)
            results[sig_name][h] = corr(list(xs), list(ys))

    return results, wall_mids, simple_mids

print("\n  Running signal analysis across all days...")

for day in (-1, 0, 1):
    d = days_data[day]
    results, wm, sm = compute_signals_and_preds(d)

    print(f"\n  Day {day}: Correlations with forward returns (horizons 1,3,5,10,20,50)")
    print(f"  {'Signal':<22} {'h=1':>7} {'h=3':>7} {'h=5':>7} {'h=10':>7} {'h=20':>7} {'h=50':>7}")
    print("  " + "-" * 72)

    for sig_name in sorted(results.keys()):
        corrs = results[sig_name]
        vals = [corrs.get(h) for h in [1, 3, 5, 10, 20, 50]]
        # Highlight if any |r| > 0.1
        has_signal = any(v is not None and abs(v) > 0.1 for v in vals)
        marker = " <<" if any(v is not None and abs(v) > 0.3 for v in vals) else (" *" if has_signal else "")
        row = f"  {sig_name:<22}"
        for v in vals:
            row += f" {v:>7.4f}" if v is not None else f" {'N/A':>7}"
        print(row + marker)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Trade Flow Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 5: TRADE FLOW ANALYSIS")
print("=" * 70)

all_trades_rows = load_trades_all(DATA_DIR)
aco_trades = [r for r in all_trades_rows if r['symbol'] == PRODUCT]

print(f"\n  ACO market trades: {len(aco_trades)} total")

# Per-day trade stats
for day in (-1, 0, 1):
    day_trades = [r for r in aco_trades if int(r['_day']) == day]
    if not day_trades:
        continue
    prices = [safe_float(r['price']) for r in day_trades if safe_float(r['price'])]
    qtys = [safe_float(r['quantity']) for r in day_trades if safe_float(r['quantity'])]

    print(f"\n  Day {day}: {len(day_trades)} trades")
    if prices:
        print(f"    Trade price: mean={mean(prices):.3f}, std={std(prices):.4f}")
        print(f"    Trade qty: mean={mean(qtys):.2f}, std={std(qtys):.2f}, "
              f"min={min(qtys):.0f}, max={max(qtys):.0f}")

    # Check buyer/seller
    buyers = [r['buyer'].strip() for r in day_trades if r.get('buyer','').strip()]
    sellers = [r['seller'].strip() for r in day_trades if r.get('seller','').strip()]
    print(f"    Unique buyers: {set(buyers)}")
    print(f"    Unique sellers: {set(sellers)}")

    # Qty distribution
    qty_counts = defaultdict(int)
    for q in qtys: qty_counts[int(q)] += 1
    print(f"    Qty distribution: {dict(sorted(qty_counts.items()))}")

    # Trade price vs mid analysis
    d = days_data[day]
    mid_by_ts = {t: m for t, m in zip(d['ts'], d['mid'])}

    deviations = []
    for r in day_trades:
        ts_t = safe_int(r['timestamp'])
        price_t = safe_float(r['price'])
        # Round to nearest 100 to match mid lookup
        ts_rounded = (ts_t // 100) * 100
        mid_t = mid_by_ts.get(ts_rounded)
        if mid_t and price_t:
            deviations.append(price_t - mid_t)

    if deviations:
        print(f"    Trade price vs mid: mean dev={mean(deviations):.4f}, "
              f"std={std(deviations):.4f}")
        pos_trades = sum(1 for d in deviations if d > 0)
        neg_trades = sum(1 for d in deviations if d < 0)
        print(f"    Trades ABOVE mid: {pos_trades}, BELOW mid: {neg_trades}")

    # Check timing regularity
    timestamps = sorted([safe_int(r['timestamp']) for r in day_trades])
    gaps = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    if gaps:
        gap_counts = defaultdict(int)
        for g in gaps: gap_counts[g] += 1
        top_gaps = sorted(gap_counts.items(), key=lambda x: -x[1])[:10]
        print(f"    Trade timing gaps (top 10): {top_gaps}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Regime Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 6: REGIME ANALYSIS (time-of-day effects)")
print("=" * 70)

for day in (-1, 0, 1):
    d = days_data[day]
    ts = d['ts']
    mid = d['mid']
    n = len(ts)

    # Split into thirds
    third = n // 3
    thirds = {
        'early': (0, third),
        'mid':   (third, 2*third),
        'late':  (2*third, n)
    }

    print(f"\n  Day {day}:")
    for name, (start, end) in thirds.items():
        seg_mid = [m for m in mid[start:end] if m is not None]
        seg_ts = ts[start:end]

        if not seg_mid: continue
        ret = [seg_mid[i] - seg_mid[i-1] for i in range(1, len(seg_mid))]
        slope, intercept, r2 = linreg(list(range(len(seg_mid))), seg_mid)

        print(f"    {name} (ts {seg_ts[0]}-{seg_ts[-1]}):")
        print(f"      mean={mean(seg_mid):.3f}, std={std(seg_mid):.3f}, "
              f"drift={seg_mid[-1]-seg_mid[0]:.2f}, "
              f"ret_std={std(ret):.4f}, slope={slope:.6f}")

    # Intraday volatility pattern (rolling std per 1000-tick block)
    block_size = 1000  # in index units (100 timestamps = 10k ticks → ~10% of day)
    block_vols = []
    for start in range(0, n - block_size, block_size):
        seg = [m for m in mid[start:start+block_size] if m is not None]
        if len(seg) > 10:
            block_vols.append(std(seg))
    print(f"    Intraday vol by block: {[round(v, 3) for v in block_vols]}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Cross-Product Analysis (ACO vs IPR)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 7: CROSS-PRODUCT ANALYSIS (ACO vs IPR)")
print("=" * 70)

# Load IPR data
ipr_days_data = {}
for day in (-1, 0, 1):
    day_rows = [r for r in all_prices_rows if r['product'] == IPR and int(r['_day']) == day]
    day_rows_sorted = sorted(day_rows, key=lambda r: int(r['timestamp']))
    ts = [int(r['timestamp']) for r in day_rows_sorted]
    mid = [safe_float(r['mid_price']) for r in day_rows_sorted]
    bp1 = [safe_float(r['bid_price_1']) for r in day_rows_sorted]
    bv1 = [safe_float(r['bid_volume_1'], 0) for r in day_rows_sorted]
    ap1 = [safe_float(r['ask_price_1']) for r in day_rows_sorted]
    av1 = [safe_float(r['ask_volume_1'], 0) for r in day_rows_sorted]
    ipr_days_data[day] = {'ts': ts, 'mid': mid, 'bp1': bp1, 'bv1': bv1, 'ap1': ap1, 'av1': av1}

for day in (-1, 0, 1):
    aco = days_data[day]
    ipr = ipr_days_data[day]

    # Align by timestamp
    aco_ts_set = {t: m for t, m in zip(aco['ts'], aco['mid']) if m}
    ipr_ts_set = {t: m for t, m in zip(ipr['ts'], ipr['mid']) if m}

    common_ts = sorted(set(aco_ts_set.keys()) & set(ipr_ts_set.keys()))
    if len(common_ts) < 100:
        print(f"  Day {day}: insufficient common timestamps")
        continue

    aco_m = [aco_ts_set[t] for t in common_ts]
    ipr_m = [ipr_ts_set[t] for t in common_ts]

    aco_ret = [aco_m[i] - aco_m[i-1] for i in range(1, len(aco_m))]
    ipr_ret = [ipr_m[i] - ipr_m[i-1] for i in range(1, len(ipr_m))]

    # Contemporaneous correlation
    r_contemp = corr(aco_ret, ipr_ret)

    # Lead-lag: does IPR lead ACO or vice versa?
    lag_corrs = {}
    for lag in [-5, -3, -2, -1, 0, 1, 2, 3, 5, 10]:
        # lag > 0: IPR leads ACO (IPR ret at t predicts ACO ret at t+lag)
        if lag >= 0:
            xs = ipr_ret[:len(ipr_ret)-lag] if lag > 0 else ipr_ret
            ys = aco_ret[lag:] if lag > 0 else aco_ret
        else:
            xs = ipr_ret[-lag:]
            ys = aco_ret[:len(aco_ret)+lag]
        if len(xs) > 100 and len(ys) > 100:
            min_len = min(len(xs), len(ys))
            lag_corrs[lag] = corr(xs[:min_len], ys[:min_len])
        else:
            lag_corrs[lag] = None

    print(f"\n  Day {day}:")
    print(f"    Contemporaneous return correlation: {r_contemp:.4f}")
    print(f"    Lead-lag correlations (lag = IPR leads ACO by N ticks):")
    for lag, c in sorted(lag_corrs.items()):
        marker = " <<" if c and abs(c) > 0.1 else ""
        print(f"      lag={lag:>3}: {c:.4f}{marker}" if c else f"      lag={lag:>3}: N/A")

    # Cointegration check: is aco - k*ipr stationary?
    # Find k via regression
    slope_k, intercept_k, r2_k = linreg(ipr_m, aco_m)
    spread_k = [a - (slope_k * b + intercept_k) for a, b in zip(aco_m, ipr_m)]
    t_stat_sp, gamma_sp = adf_approx(spread_k)
    hl_sp = half_life(spread_k)
    print(f"    Cointegration: ACO = {slope_k:.6f}*IPR + {intercept_k:.2f}, R²={r2_k:.4f}")
    print(f"    Spread stationarity: ADF t={t_stat_sp:.3f}, half-life={hl_sp:.1f} ticks")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Wall Structure Deep Dive
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 8: WALL STRUCTURE DEEP DIVE")
print("=" * 70)

for day in (-1, 0, 1):
    d = days_data[day]
    n = len(d['ts'])

    # Catalog every volume level and its price position
    bid_vol_price = defaultdict(list)  # vol → list of prices
    ask_vol_price = defaultdict(list)

    for i in range(n):
        for p_key, v_key in [('bp1', 'bv1'), ('bp2', 'bv2'), ('bp3', 'bv3')]:
            p, v = d[p_key][i], d[v_key][i]
            if p and v:
                bid_vol_price[int(v)].append(p)
        for p_key, v_key in [('ap1', 'av1'), ('ap2', 'av2'), ('ap3', 'av3')]:
            p, v = d[p_key][i], d[v_key][i]
            if p and v:
                ask_vol_price[int(v)].append(p)

    print(f"\n  Day {day}: Bid volume → price statistics")
    for vol in sorted(bid_vol_price.keys()):
        ps = bid_vol_price[vol]
        if len(ps) > 100:
            print(f"    vol={vol:>3}: count={len(ps):>5}, "
                  f"mean_price={mean(ps):.2f}, std={std(ps):.3f}")

    print(f"  Day {day}: Ask volume → price statistics")
    for vol in sorted(ask_vol_price.keys()):
        ps = ask_vol_price[vol]
        if len(ps) > 100:
            print(f"    vol={vol:>3}: count={len(ps):>5}, "
                  f"mean_price={mean(ps):.2f}, std={std(ps):.3f}")

    # Look for fixed-volume levels (potential bots quoting at same size always)
    # Compute how often a specific volume appears at L1 vs L2 vs L3
    l1_bid_vols = defaultdict(int)
    l2_bid_vols = defaultdict(int)
    l1_ask_vols = defaultdict(int)
    l2_ask_vols = defaultdict(int)

    for i in range(n):
        if d['bv1'][i]: l1_bid_vols[int(d['bv1'][i])] += 1
        if d['bv2'][i]: l2_bid_vols[int(d['bv2'][i])] += 1
        if d['av1'][i]: l1_ask_vols[int(d['av1'][i])] += 1
        if d['av2'][i]: l2_ask_vols[int(d['av2'][i])] += 1

    print(f"  Day {day}: L1 bid vol distribution: {dict(sorted(l1_bid_vols.items()))}")
    print(f"  Day {day}: L1 ask vol distribution: {dict(sorted(l1_ask_vols.items()))}")
    print(f"  Day {day}: L2 bid vol distribution: {dict(sorted(l2_bid_vols.items()))}")
    print(f"  Day {day}: L2 ask vol distribution: {dict(sorted(l2_ask_vols.items()))}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: Fair Value Deep Analysis — mean-reversion signal
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 9: MEAN REVERSION SIGNAL — THRESHOLDS & PREDICTABILITY")
print("=" * 70)

for day in (-1, 0, 1):
    d = days_data[day]
    ts = d['ts']
    mid = [m for m in d['mid'] if m is not None]
    n = len(mid)

    # Overall mean
    overall_mean = mean(mid)
    overall_std = std(mid)

    # z-score (price level above/below mean)
    zscores = [(m - overall_mean) / overall_std for m in mid]

    # Conditional returns given z-score sign
    horizons = [1, 5, 10, 20]
    print(f"\n  Day {day}: overall mean={overall_mean:.4f}, std={overall_std:.4f}")

    for h in horizons:
        positive_z_ret = []
        negative_z_ret = []
        high_z_ret = []   # |z| > 1

        for i in range(n - h):
            z = zscores[i]
            fwd_ret = mid[i + h] - mid[i]
            if z > 0.5:
                positive_z_ret.append(fwd_ret)
            elif z < -0.5:
                negative_z_ret.append(fwd_ret)
            if z > 1.0:
                high_z_ret.append(-fwd_ret)  # expect to revert down
            elif z < -1.0:
                high_z_ret.append(fwd_ret)   # expect to revert up

        pos_mean = mean(positive_z_ret) if positive_z_ret else 0
        neg_mean = mean(negative_z_ret) if negative_z_ret else 0
        high_mean = mean(high_z_ret) if high_z_ret else 0

        print(f"    h={h}: pos_z avg_ret={pos_mean:.5f}, neg_z avg_ret={neg_mean:.5f}, "
              f"|z|>1 reversion={high_mean:.5f} (n={len(high_z_ret)})")

    # Threshold analysis: entry threshold that maximizes edge
    print(f"\n  Day {day}: Entry threshold analysis (take when mid deviates from 10001 by X)")
    for thresh in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        above_thresh = [(i, mid[i]) for i in range(n) if mid[i] > 10001 + thresh]
        below_thresh = [(i, mid[i]) for i in range(n) if mid[i] < 10001 - thresh]

        # Expected reversion at h=5
        h = 5
        above_rets = [mid[i + h] - mid[i] for i, _ in above_thresh if i + h < n]
        below_rets = [mid[i + h] - mid[i] for i, _ in below_thresh if i + h < n]

        above_edge = -mean(above_rets) if above_rets else 0  # short when above
        below_edge = mean(below_rets) if below_rets else 0   # long when below

        print(f"    thresh={thresh}: above_n={len(above_rets)}, short_edge={above_edge:.4f}; "
              f"below_n={len(below_rets)}, long_edge={below_edge:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: Spread structure & fair value anchor
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 10: SPREAD STRUCTURE & FAIR VALUE ANCHOR")
print("=" * 70)

for day in (-1, 0, 1):
    d = days_data[day]
    n = len(d['ts'])

    # Check if bid_price_2 - bid_price_1 is constant (tier structure)
    l1_l2_bid_gaps = []
    l1_l2_ask_gaps = []

    for i in range(n):
        bp1, bp2 = d['bp1'][i], d['bp2'][i]
        ap1, ap2 = d['ap1'][i], d['ap2'][i]
        if bp1 and bp2: l1_l2_bid_gaps.append(bp1 - bp2)
        if ap1 and ap2: l1_l2_ask_gaps.append(ap2 - ap1)

    print(f"\n  Day {day}:")
    if l1_l2_bid_gaps:
        gap_counts = defaultdict(int)
        for g in l1_l2_bid_gaps: gap_counts[round(g)] += 1
        print(f"    L1-L2 bid gap: mean={mean(l1_l2_bid_gaps):.3f}, "
              f"std={std(l1_l2_bid_gaps):.3f}, dist={dict(sorted(gap_counts.items()))}")
    if l1_l2_ask_gaps:
        gap_counts = defaultdict(int)
        for g in l1_l2_ask_gaps: gap_counts[round(g)] += 1
        print(f"    L1-L2 ask gap: mean={mean(l1_l2_ask_gaps):.3f}, "
              f"std={std(l1_l2_ask_gaps):.3f}, dist={dict(sorted(gap_counts.items()))}")

    # Check if wall levels are essentially fixed
    all_wall_bids = []
    all_wall_asks = []
    for i in range(n):
        best_bid_vol, best_bid_p = 0, None
        for p_key, v_key in [('bp1', 'bv1'), ('bp2', 'bv2'), ('bp3', 'bv3')]:
            p, v = d[p_key][i], d[v_key][i]
            if p and v and v >= 8 and v > best_bid_vol:
                best_bid_vol, best_bid_p = v, p
        best_ask_vol, best_ask_p = 0, None
        for p_key, v_key in [('ap1', 'av1'), ('ap2', 'av2'), ('ap3', 'av3')]:
            p, v = d[p_key][i], d[v_key][i]
            if p and v and v >= 8 and v > best_ask_vol:
                best_ask_vol, best_ask_p = v, p
        if best_bid_p: all_wall_bids.append(best_bid_p)
        if best_ask_p: all_wall_asks.append(best_ask_p)

    if all_wall_bids and all_wall_asks:
        wb_counts = defaultdict(int)
        wa_counts = defaultdict(int)
        for p in all_wall_bids: wb_counts[p] += 1
        for p in all_wall_asks: wa_counts[p] += 1

        top_wb = sorted(wb_counts.items(), key=lambda x: -x[1])[:10]
        top_wa = sorted(wa_counts.items(), key=lambda x: -x[1])[:10]
        print(f"    Top wall bid prices (price: count): {top_wb}")
        print(f"    Top wall ask prices (price: count): {top_wa}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: Specific exploitable pattern check
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 11: SPECIFIC EXPLOIT CHECKS")
print("=" * 70)

# Check 1: Is wall_bid + wall_ask always 20002?
print("\n  Check 1: Does wall_bid + wall_ask = 20002?")
for day in (-1, 0, 1):
    d = days_data[day]
    n = len(d['ts'])
    sums = []
    for i in range(n):
        best_bid_vol, best_bid_p = 0, None
        for p_key, v_key in [('bp1', 'bv1'), ('bp2', 'bv2'), ('bp3', 'bv3')]:
            p, v = d[p_key][i], d[v_key][i]
            if p and v and v >= 8 and v > best_bid_vol:
                best_bid_vol, best_bid_p = v, p
        best_ask_vol, best_ask_p = 0, None
        for p_key, v_key in [('ap1', 'av1'), ('ap2', 'av2'), ('ap3', 'av3')]:
            p, v = d[p_key][i], d[v_key][i]
            if p and v and v >= 8 and v > best_ask_vol:
                best_ask_vol, best_ask_p = v, p
        if best_bid_p and best_ask_p:
            sums.append(best_bid_p + best_ask_p)
    if sums:
        sum_counts = defaultdict(int)
        for s in sums: sum_counts[s] += 1
        top_sums = sorted(sum_counts.items(), key=lambda x: -x[1])[:5]
        print(f"    Day {day}: wall_bid+wall_ask top values: {top_sums}")
        print(f"    Day {day}: mean={mean(sums):.3f}, std={std(sums):.4f}")

# Check 2: Does the L1 ask always = 10000?
print("\n  Check 2: Distribution of best ask prices")
for day in (-1, 0, 1):
    d = days_data[day]
    ask_counts = defaultdict(int)
    bid_counts = defaultdict(int)
    for p in d['ap1']:
        if p: ask_counts[p] += 1
    for p in d['bp1']:
        if p: bid_counts[p] += 1
    top_asks = sorted(ask_counts.items(), key=lambda x: -x[1])[:8]
    top_bids = sorted(bid_counts.items(), key=lambda x: -x[1])[:8]
    print(f"  Day {day}: Top L1 ask prices: {top_asks}")
    print(f"  Day {day}: Top L1 bid prices: {top_bids}")

# Check 3: Can we arbitrage between L1 and L2?
print("\n  Check 3: L2 quote quality vs L1")
for day in (-1, 0, 1):
    d = days_data[day]
    n = len(d['ts'])
    count_l2_better_bid = 0
    count_l2_better_ask = 0
    for i in range(n):
        if d['bp2'][i] and d['bp1'][i] and d['bp2'][i] > d['bp1'][i]:
            count_l2_better_bid += 1
        if d['ap2'][i] and d['ap1'][i] and d['ap2'][i] < d['ap1'][i]:
            count_l2_better_ask += 1
    print(f"  Day {day}: L2 bid > L1 bid: {count_l2_better_bid}, "
          f"L2 ask < L1 ask: {count_l2_better_ask} (should be 0; if >0, crossing quotes!)")

# Check 4: imbalance signal — detailed threshold analysis
print("\n  Check 4: Imbalance signal threshold analysis")
for day in (-1, 0, 1):
    d = days_data[day]
    n = len(d['ts'])
    mid = d['mid']

    for h in [1, 5]:
        pairs = []
        for i in range(n - h):
            bv1, av1 = d['bv1'][i], d['av1'][i]
            if bv1 and av1 and bv1 + av1 > 0 and mid[i] and mid[i+h]:
                imb = (bv1 - av1) / (bv1 + av1)
                fwd = mid[i+h] - mid[i]
                pairs.append((imb, fwd))

        if not pairs: continue

        # Bucket by imbalance
        buckets = defaultdict(list)
        for imb, fwd in pairs:
            b = round(imb * 5) / 5  # round to nearest 0.2
            buckets[b].append(fwd)

        print(f"  Day {day}, h={h}: imbalance → mean forward return")
        for b in sorted(buckets.keys()):
            if len(buckets[b]) > 50:
                print(f"    imb={b:>5.2f}: n={len(buckets[b]):>4}, "
                      f"mean_ret={mean(buckets[b]):.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: 100k data analysis (live-equivalent)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 12: 100k SESSION ANALYSIS (LIVE-EQUIVALENT)")
print("=" * 70)

prices_100k = load_prices(DATA_100K)
print(f"\n  100k session ACO rows: {len(prices_100k)}")

for day in (-1, 0, 1):
    day_rows = sorted([r for r in prices_100k if int(r['day']) == day],
                      key=lambda r: int(r['timestamp']))
    if not day_rows: continue
    ts = [int(r['timestamp']) for r in day_rows]
    mid = [safe_float(r['mid_price']) for r in day_rows if safe_float(r['mid_price'])]
    if not mid: continue

    bv1 = [safe_float(r['bid_volume_1'], 0) for r in day_rows]
    av1 = [safe_float(r['ask_volume_1'], 0) for r in day_rows]
    bp1 = [safe_float(r['bid_price_1']) for r in day_rows]
    ap1 = [safe_float(r['ask_price_1']) for r in day_rows]

    spreads = [a - b for a, b in zip(ap1, bp1) if a and b]

    slope, intercept, r2 = linreg(ts[:len(mid)], mid)
    print(f"\n  Day {day} (100k):")
    print(f"    N={len(mid)}, mean={mean(mid):.4f}, std={std(mid):.4f}")
    print(f"    range: [{min(mid):.1f}, {max(mid):.1f}]")
    print(f"    drift: {mid[-1]-mid[0]:.2f} over {ts[-1]-ts[0]} ticks")
    print(f"    linreg: slope={slope:.8f}, R²={r2:.6f}")
    if spreads:
        spread_counts = defaultdict(int)
        for s in spreads: spread_counts[int(s)] += 1
        print(f"    spread dist: {dict(sorted(spread_counts.items()))}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: Volume consistency check (bot detection)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 13: BOT PATTERN DETECTION IN TRADE FLOW")
print("=" * 70)

for day in (-1, 0, 1):
    day_trades = sorted([r for r in aco_trades if int(r['_day']) == day],
                        key=lambda r: int(r['timestamp']))
    if not day_trades: continue

    qtys = [safe_float(r['quantity']) for r in day_trades if safe_float(r['quantity'])]
    prices = [safe_float(r['price']) for r in day_trades if safe_float(r['price'])]
    timestamps = [safe_int(r['timestamp']) for r in day_trades]

    print(f"\n  Day {day}: {len(day_trades)} trades")

    # Quantity consistency: most common quantity
    qty_counts = defaultdict(int)
    for q in qtys: qty_counts[q] += 1
    sorted_qtys = sorted(qty_counts.items(), key=lambda x: -x[1])
    mode_qty, mode_count = sorted_qtys[0] if sorted_qtys else (None, 0)

    print(f"    Most common qty: {mode_qty} ({mode_count}/{len(qtys)} = "
          f"{100*mode_count/len(qtys):.1f}%)")
    print(f"    All qty counts: {sorted_qtys}")

    # Trade timing — look for regular intervals
    gaps = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    if gaps:
        gap_counts = defaultdict(int)
        for g in gaps: gap_counts[g] += 1
        top_gaps = sorted(gap_counts.items(), key=lambda x: -x[1])[:15]
        print(f"    Most common inter-trade gaps: {top_gaps}")

    # Does trade price predict next trade direction?
    if len(prices) > 10:
        d_prices = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        sign_changes = sum(1 for i in range(1, len(d_prices))
                          if d_prices[i] * d_prices[i-1] < 0)
        print(f"    Price direction reversals: {sign_changes}/{len(d_prices)-1}")

    # Compare trade prices to nearest mid
    d_day = days_data[day]
    mid_by_ts = {}
    for t, m in zip(d_day['ts'], d_day['mid']):
        mid_by_ts[t] = m

    trade_vs_mid = []
    for r in day_trades:
        t = (safe_int(r['timestamp']) // 100) * 100
        p = safe_float(r['price'])
        m = mid_by_ts.get(t)
        if p and m:
            trade_vs_mid.append(p - m)

    if trade_vs_mid:
        print(f"    Trade price - mid: mean={mean(trade_vs_mid):.4f}, "
              f"std={std(trade_vs_mid):.4f}")
        pos = sum(1 for x in trade_vs_mid if x > 0)
        neg = sum(1 for x in trade_vs_mid if x < 0)
        zero = sum(1 for x in trade_vs_mid if x == 0)
        print(f"    Above mid: {pos}, at mid: {zero}, below mid: {neg}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
