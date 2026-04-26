# Options Pricing & Volatility Trading Reference

## When to Use
- Products described as vouchers, coupons, or options on an underlying
- Multiple strike prices available
- Time-to-expiry mechanics mentioned
- Examples: Coconut Coupons (P2), Volcanic Rock Vouchers (P3)

## Black-Scholes Implementation

```python
import math
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes call option price.
    S: spot price of underlying
    K: strike price
    T: time to expiry (in years, e.g., 5/252 for 5 trading days)
    r: risk-free rate (USE 0.0 for Prosperity)
    sigma: implied volatility
    """
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def black_scholes_delta(S, K, T, r, sigma):
    """Delta of a call option."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)

def implied_volatility(market_price, S, K, T, r, max_iter=100, tol=1e-6):
    """
    Newton-Raphson to find implied volatility.
    """
    sigma = 0.2  # Initial guess
    
    for _ in range(max_iter):
        price = black_scholes_call(S, K, T, r, sigma)
        vega = S * norm.pdf(
            (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        ) * math.sqrt(T)
        
        if abs(vega) < 1e-12:
            break
        
        sigma -= (price - market_price) / vega
        
        if abs(price - market_price) < tol:
            break
        
        sigma = max(0.01, min(sigma, 5.0))  # Clamp
    
    return sigma
```

## Volatility Smile Fitting

Top teams fit a quadratic to implied volatilities across strikes:

```python
def fit_volatility_smile(strikes, ivs, underlying_price):
    """
    Fit IV smile: sigma(K) = a + b*(K/S - 1) + c*(K/S - 1)^2
    Returns coefficients (a, b, c)
    """
    import numpy as np
    
    moneyness = [(K / underlying_price - 1) for K in strikes]
    X = np.column_stack([
        np.ones(len(moneyness)),
        moneyness,
        [m**2 for m in moneyness]
    ])
    
    coeffs = np.linalg.lstsq(X, ivs, rcond=None)[0]
    return coeffs

def smile_iv(K, S, coeffs):
    """Get IV from smile at strike K."""
    m = K / S - 1
    return coeffs[0] + coeffs[1] * m + coeffs[2] * m**2
```

## Mean Reversion on Options (Primary Profit Source)

The Frankfurt Hedgehogs' most profitable options strategy:

```python
def options_mean_reversion(self, state, option_product, underlying_product, 
                           strike, days_remaining, data, limit=200):
    """
    Trade mean reversion on option price vs Black-Scholes fair value.
    Frankfurt Hedgehogs: ~15,000-25,000 SeaShells/round
    """
    orders = []
    
    # Get underlying price
    underlying_depth = state.order_depths.get(underlying_product)
    if not underlying_depth or not underlying_depth.buy_orders or not underlying_depth.sell_orders:
        return orders
    
    S = (max(underlying_depth.buy_orders.keys()) + min(underlying_depth.sell_orders.keys())) / 2
    
    # Get option mid price
    option_depth = state.order_depths.get(option_product)
    if not option_depth or not option_depth.buy_orders or not option_depth.sell_orders:
        return orders
    
    option_mid = (max(option_depth.buy_orders.keys()) + min(option_depth.sell_orders.keys())) / 2
    
    # Calculate fair value
    T = days_remaining / 252  # Convert to years
    
    # Use rolling IV estimate
    iv_history = data.get(f"{option_product}_iv", [])
    current_iv = implied_volatility(option_mid, S, strike, T, 0.0)
    iv_history.append(current_iv)
    if len(iv_history) > 50:
        iv_history = iv_history[-50:]
    data[f"{option_product}_iv"] = iv_history
    
    # Use smoothed IV for fair value
    if len(iv_history) >= 10:
        smooth_iv = sum(iv_history[-30:]) / len(iv_history[-30:])
    else:
        smooth_iv = current_iv
    
    fair_value = black_scholes_call(S, strike, T, 0.0, smooth_iv)
    
    # Trade deviations
    pos = state.position.get(option_product, 0)
    threshold = max(1, fair_value * 0.02)  # 2% deviation threshold
    
    if option_mid < fair_value - threshold and pos < limit:
        # Option underpriced — buy
        best_ask = min(option_depth.sell_orders.keys())
        qty = min(-option_depth.sell_orders[best_ask], limit - pos, 20)
        if qty > 0:
            orders.append(Order(option_product, best_ask, qty))
    
    elif option_mid > fair_value + threshold and pos > -limit:
        # Option overpriced — sell
        best_bid = max(option_depth.buy_orders.keys())
        qty = min(option_depth.buy_orders[best_bid], limit + pos, 20)
        if qty > 0:
            orders.append(Order(option_product, best_bid, -qty))
    
    return orders
```

## Delta Hedging

```python
def delta_hedge(self, state, option_products, underlying_product, 
                strikes, days_remaining, data, underlying_limit=400):
    """
    Hedge aggregate delta exposure with the underlying.
    """
    S = self.get_mid(state.order_depths[underlying_product])
    T = days_remaining / 252
    
    # Calculate aggregate delta
    total_delta = 0
    for option, strike in zip(option_products, strikes):
        pos = state.position.get(option, 0)
        if pos == 0:
            continue
        
        iv_history = data.get(f"{option}_iv", [0.2])
        sigma = sum(iv_history[-10:]) / len(iv_history[-10:])
        
        delta = black_scholes_delta(S, strike, T, 0.0, sigma)
        total_delta += pos * delta
    
    # Current underlying position
    underlying_pos = state.position.get(underlying_product, 0)
    
    # Target: delta-neutral
    hedge_needed = -int(total_delta) - underlying_pos
    
    orders = []
    depth = state.order_depths[underlying_product]
    
    if hedge_needed > 0 and underlying_pos + hedge_needed <= underlying_limit:
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if best_ask:
            orders.append(Order(underlying_product, best_ask, min(hedge_needed, 20)))
    elif hedge_needed < 0 and underlying_pos + hedge_needed >= -underlying_limit:
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        if best_bid:
            orders.append(Order(underlying_product, best_bid, max(hedge_needed, -20)))
    
    return orders
```

## Parameters from Top Teams

| Parameter | P2 Value | P3 Value |
|-----------|----------|----------|
| Risk-free rate | 0.0 | 0.0 |
| IV initial guess | 0.16 | 0.20 |
| IV rolling window | 20 | 30 |
| Execution threshold | 1.8% | 2.0% |
| Position limit per strike | 100 | 200 |
| Delta hedge frequency | Every timestamp | Every timestamp |
| Days in year (for T) | 252 | 252 |
