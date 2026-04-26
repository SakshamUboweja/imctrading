# Statistical Arbitrage Reference

## When to Use
- Composite products (baskets, ETFs) with tradeable components
- Pairs of highly correlated products
- Examples: Picnic Baskets (P3), Gift Baskets (P2), Coconut-Pina Colada (P1)

## Core Concept

When a basket/ETF is composed of known components with fixed weights, its theoretical price is determined by the components. Any deviation from this theoretical price is an arbitrage opportunity that should mean-revert.

## Step 1: Find the Relationship

```python
import numpy as np

def find_basket_weights(basket_prices, component_prices_dict):
    """
    Use linear regression to find weights.
    basket_price ≈ intercept + w1*comp1 + w2*comp2 + ...
    
    In P3: PICNIC_BASKET1 = -57.71 + 6*CROISSANTS + 3*JAMS + 1*DJEMBES
    """
    # Build X matrix from component prices
    components = list(component_prices_dict.keys())
    X = np.column_stack([component_prices_dict[c] for c in components])
    X = np.column_stack([np.ones(len(basket_prices)), X])  # Add intercept
    
    # OLS regression
    coeffs = np.linalg.lstsq(X, basket_prices, rcond=None)[0]
    
    intercept = coeffs[0]
    weights = {c: coeffs[i+1] for i, c in enumerate(components)}
    
    return intercept, weights
```

## Step 2: Calculate Spread and Z-Score

```python
def calculate_spread(self, state, basket_product, components, weights, intercept):
    """Calculate spread between actual basket price and synthetic value."""
    depth = state.order_depths
    
    # Basket mid
    basket_mid = self.get_mid(depth[basket_product])
    
    # Synthetic value
    synthetic = intercept
    for comp, weight in components.items():
        comp_mid = self.get_mid(depth[comp])
        synthetic += weight * comp_mid
    
    return basket_mid - synthetic

def z_score(self, spread, spread_history, window=20):
    """
    Z-score for mean reversion signal.
    Linear Utility (2nd P2) used: (premium - 380) / 76 for gift baskets
    Frankfurt Hedgehogs used: fixed thresholds ±50 with Olivia adjustment
    """
    if len(spread_history) < window:
        return 0
    
    recent = spread_history[-window:]
    mean = sum(recent) / len(recent)
    std = (sum((x - mean)**2 for x in recent) / len(recent)) ** 0.5
    
    if std < 1e-6:
        return 0
    return (spread - mean) / std
```

## Step 3: Trading Logic

```python
def stat_arb_orders(self, state, basket, components, weights, intercept, data, limit):
    """
    Complete stat arb implementation.
    Target: 40,000-60,000 SeaShells/round (P3 baskets)
    """
    orders = {}
    
    # Calculate current spread
    spread = self.calculate_spread(state, basket, components, weights, intercept)
    
    # Update spread history
    history = data.get(f"{basket}_spread_history", [])
    history.append(spread)
    if len(history) > 100:
        history = history[-100:]
    data[f"{basket}_spread_history"] = history
    
    # Calculate z-score
    z = self.z_score(spread, history, window=20)
    
    # Trading thresholds
    ENTRY_THRESHOLD = 2.0   # Enter when |z| > 2
    EXIT_THRESHOLD = 0.5    # Exit when |z| < 0.5
    
    pos = state.position.get(basket, 0)
    depth = state.order_depths[basket]
    
    basket_orders = []
    
    if z > ENTRY_THRESHOLD and pos > -limit:
        # Basket overpriced → sell basket
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        if best_bid:
            qty = min(depth.buy_orders[best_bid], limit + pos)
            basket_orders.append(Order(basket, best_bid, -qty))
    
    elif z < -ENTRY_THRESHOLD and pos < limit:
        # Basket underpriced → buy basket
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if best_ask:
            qty = min(-depth.sell_orders[best_ask], limit - pos)
            basket_orders.append(Order(basket, best_ask, qty))
    
    elif abs(z) < EXIT_THRESHOLD and pos != 0:
        # Mean reverted → close position
        if pos > 0:
            best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
            if best_bid:
                basket_orders.append(Order(basket, best_bid, -pos))
        else:
            best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
            if best_ask:
                basket_orders.append(Order(basket, best_ask, -pos))
    
    if basket_orders:
        orders[basket] = basket_orders
    
    # Optional: Hedge with components (50% hedge ratio from Frankfurt Hedgehogs)
    # Only if position limits allow
    
    return orders
```

## Advanced: Cross-Product Signal Integration

The 2nd place P3 team's biggest edge: using Olivia's position on Croissants to adjust basket thresholds.

```python
def adjusted_thresholds(self, base_threshold, olivia_signal):
    """
    Adjust entry thresholds based on informed trader signals.
    olivia_signal: -1 (short), 0 (flat), +1 (long)
    """
    ADJUSTMENT = 30  # ticks
    
    if olivia_signal > 0:  # Olivia long on component
        long_threshold = -(base_threshold + ADJUSTMENT)   # Harder to go long
        short_threshold = base_threshold - ADJUSTMENT       # Easier to go short
    elif olivia_signal < 0:  # Olivia short
        long_threshold = -(base_threshold - ADJUSTMENT)    # Easier to go long
        short_threshold = base_threshold + ADJUSTMENT       # Harder to go short
    else:
        long_threshold = -base_threshold
        short_threshold = base_threshold
    
    return long_threshold, short_threshold
```

## Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Z-score entry threshold | 1.5-2.5 | Higher = fewer trades but higher conviction |
| Z-score exit threshold | 0.0-0.5 | 0 = close at mean, 0.5 = close early |
| Rolling window | 20-50 | Frankfurt Hedgehogs used exponential smoothing |
| Hedge ratio | 0.5 | Hedge 50% of basket with components |
| Olivia adjustment | ±30 ticks | Only when informed signals available |
