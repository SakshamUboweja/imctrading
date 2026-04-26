"""
IMC Prosperity 4 - Core Calculation Utilities
Battle-tested implementations from top-performing teams.
"""

import math
import json
from typing import Optional

# ============================================================
# MEAN REVERSION TESTING
# ============================================================

def augmented_dickey_fuller_simple(prices: list, max_lag: int = 5) -> dict:
    """
    Simplified ADF test for mean reversion.
    Tests if a price series is stationary (mean-reverting) vs random walk.

    Returns dict with:
        - statistic: ADF test statistic (more negative = more mean-reverting)
        - is_mean_reverting: True if statistic < -2.86 (5% significance)
        - half_life: estimated half-life of mean reversion in periods
    """
    n = len(prices)
    if n < max_lag + 10:
        return {"statistic": 0, "is_mean_reverting": False, "half_life": float('inf')}

    # First differences
    diffs = [prices[i] - prices[i-1] for i in range(1, n)]

    # Lagged levels
    lagged = prices[:-1]

    # Simple OLS: diff_t = alpha + beta * price_{t-1} + epsilon
    n_obs = len(diffs)
    x_mean = sum(lagged) / n_obs
    y_mean = sum(diffs) / n_obs

    ss_xx = sum((x - x_mean)**2 for x in lagged)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(lagged, diffs))

    if ss_xx == 0:
        return {"statistic": 0, "is_mean_reverting": False, "half_life": float('inf')}

    beta = ss_xy / ss_xx
    alpha = y_mean - beta * x_mean

    # Residuals and standard error
    residuals = [d - (alpha + beta * x) for d, x in zip(diffs, lagged)]
    sse = sum(r**2 for r in residuals)
    se_beta = math.sqrt(sse / (n_obs - 2) / ss_xx) if n_obs > 2 else float('inf')

    # ADF statistic
    adf_stat = beta / se_beta if se_beta > 0 else 0

    # Half-life of mean reversion
    half_life = -math.log(2) / beta if beta < 0 else float('inf')

    return {
        "statistic": adf_stat,
        "is_mean_reverting": adf_stat < -2.86,  # 5% critical value
        "half_life": half_life,
        "beta": beta
    }


def hurst_exponent(prices: list, max_lag: int = 20) -> float:
    """
    Estimate Hurst exponent.
    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending
    """
    n = len(prices)
    if n < max_lag * 2:
        return 0.5

    lags = range(2, max_lag + 1)
    variances = []

    for lag in lags:
        diffs = [prices[i] - prices[i - lag] for i in range(lag, n)]
        if diffs:
            var = sum(d**2 for d in diffs) / len(diffs)
            variances.append(var)
        else:
            variances.append(0)

    # Log-log regression
    valid = [(l, v) for l, v in zip(lags, variances) if v > 0]
    if len(valid) < 3:
        return 0.5

    log_lags = [math.log(l) for l, _ in valid]
    log_vars = [math.log(v) for _, v in valid]

    n_v = len(log_lags)
    x_mean = sum(log_lags) / n_v
    y_mean = sum(log_vars) / n_v

    ss_xx = sum((x - x_mean)**2 for x in log_lags)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(log_lags, log_vars))

    if ss_xx == 0:
        return 0.5

    slope = ss_xy / ss_xx
    return slope / 2  # Hurst exponent = slope/2


# ============================================================
# BLACK-SCHOLES
# ============================================================

def norm_cdf(x: float) -> float:
    """Standard normal CDF (no scipy dependency)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call option price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put option price."""
    if T <= 0 or sigma <= 0:
        return max(0, K - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Delta of a call option."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma of a call option."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega of a call option."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * norm_pdf(d1) * math.sqrt(T)

def implied_volatility(market_price: float, S: float, K: float, T: float,
                       r: float = 0.0, max_iter: int = 100, tol: float = 1e-6) -> float:
    """Newton-Raphson implied volatility solver."""
    sigma = 0.2  # Initial guess

    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)

        if abs(vega) < 1e-12:
            break

        diff = price - market_price
        if abs(diff) < tol:
            break

        sigma -= diff / vega
        sigma = max(0.01, min(sigma, 5.0))

    return sigma


# ============================================================
# Z-SCORE CALCULATIONS
# ============================================================

def rolling_z_score(value: float, history: list, window: int = 20) -> float:
    """Calculate z-score of current value vs rolling window."""
    if len(history) < window:
        return 0.0

    recent = history[-window:]
    mean = sum(recent) / len(recent)
    variance = sum((x - mean)**2 for x in recent) / len(recent)
    std = math.sqrt(variance) if variance > 0 else 1e-10

    return (value - mean) / std

def ema(values: list, span: int = 20) -> float:
    """Exponential moving average of last value."""
    if not values:
        return 0.0

    alpha = 2 / (span + 1)
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


# ============================================================
# VOLATILITY SMILE FITTING
# ============================================================

def fit_quadratic_smile(strikes: list, ivs: list, spot: float) -> tuple:
    """
    Fit IV smile: sigma(K) = a + b*(K/S - 1) + c*(K/S - 1)^2
    Returns (a, b, c) coefficients.
    Pure Python implementation (no numpy needed in submission).
    """
    n = len(strikes)
    if n < 3:
        return (sum(ivs)/len(ivs) if ivs else 0.2, 0, 0)

    # Moneyness
    m = [(K / spot - 1) for K in strikes]

    # Build normal equations for quadratic fit
    # X = [1, m, m^2], solve X'X * beta = X'y
    s0 = n
    s1 = sum(m)
    s2 = sum(x**2 for x in m)
    s3 = sum(x**3 for x in m)
    s4 = sum(x**4 for x in m)

    ty0 = sum(ivs)
    ty1 = sum(x * y for x, y in zip(m, ivs))
    ty2 = sum(x**2 * y for x, y in zip(m, ivs))

    # Solve 3x3 system using Cramer's rule
    A = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]]
    b_vec = [ty0, ty1, ty2]

    def det3(M):
        return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
               -M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
               +M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))

    D = det3(A)
    if abs(D) < 1e-12:
        return (sum(ivs)/n, 0, 0)

    def replace_col(M, col, vec):
        result = [row[:] for row in M]
        for i in range(3):
            result[i][col] = vec[i]
        return result

    a = det3(replace_col(A, 0, b_vec)) / D
    b = det3(replace_col(A, 1, b_vec)) / D
    c = det3(replace_col(A, 2, b_vec)) / D

    return (a, b, c)

def smile_iv(K: float, S: float, coeffs: tuple) -> float:
    """Get IV from fitted smile at strike K."""
    a, b, c = coeffs
    m = K / S - 1
    return a + b * m + c * m**2


# ============================================================
# CORRELATION & REGRESSION
# ============================================================

def correlation(x: list, y: list) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n != len(y) or n < 3:
        return 0.0

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    var_x = sum((xi - x_mean)**2 for xi in x)
    var_y = sum((yi - y_mean)**2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0

    return cov / denom

def linear_regression(x: list, y: list) -> tuple:
    """
    Simple OLS: y = alpha + beta * x
    Returns (alpha, beta, r_squared)
    """
    n = len(x)
    if n != len(y) or n < 3:
        return (0, 0, 0)

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    ss_xx = sum((xi - x_mean)**2 for xi in x)
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    ss_yy = sum((yi - y_mean)**2 for yi in y)

    if ss_xx == 0:
        return (y_mean, 0, 0)

    beta = ss_xy / ss_xx
    alpha = y_mean - beta * x_mean

    r_squared = (ss_xy**2 / (ss_xx * ss_yy)) if ss_yy > 0 else 0

    return (alpha, beta, r_squared)

def multi_linear_regression(X: list, y: list) -> list:
    """
    Multiple linear regression: y = b0 + b1*x1 + b2*x2 + ...
    X is a list of lists (each inner list is one feature's values).
    Returns list of coefficients [b0, b1, b2, ...]

    Pure Python implementation for use in Prosperity submissions.
    """
    n = len(y)
    k = len(X)  # number of features

    # Add intercept column
    Xmat = [[1.0] + [X[j][i] for j in range(k)] for i in range(n)]
    p = k + 1  # columns including intercept

    # X'X
    XtX = [[0.0]*p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            XtX[i][j] = sum(Xmat[r][i] * Xmat[r][j] for r in range(n))

    # X'y
    Xty = [sum(Xmat[r][i] * y[r] for r in range(n)) for i in range(p)]

    # Solve via Gaussian elimination
    # Augmented matrix
    aug = [XtX[i][:] + [Xty[i]] for i in range(p)]

    for col in range(p):
        # Partial pivoting
        max_row = max(range(col, p), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]

        if abs(aug[col][col]) < 1e-12:
            continue

        # Eliminate below
        for row in range(col + 1, p):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, p + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    coeffs = [0.0] * p
    for i in range(p - 1, -1, -1):
        if abs(aug[i][i]) < 1e-12:
            continue
        coeffs[i] = aug[i][p]
        for j in range(i + 1, p):
            coeffs[i] -= aug[i][j] * coeffs[j]
        coeffs[i] /= aug[i][i]

    return coeffs


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def sharpe_ratio(returns: list) -> float:
    """Annualized Sharpe ratio (assuming ~1000 timestamps/day)."""
    if len(returns) < 2:
        return 0.0

    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r)**2 for r in returns) / len(returns)
    std_r = math.sqrt(var_r) if var_r > 0 else 1e-10

    return mean_r / std_r * math.sqrt(1000)  # Annualize by ~timestamps per day

def max_drawdown(pnl_series: list) -> float:
    """Maximum drawdown from peak."""
    if not pnl_series:
        return 0.0

    peak = pnl_series[0]
    max_dd = 0.0

    for pnl in pnl_series:
        peak = max(peak, pnl)
        dd = peak - pnl
        max_dd = max(max_dd, dd)

    return max_dd
