# Conversion / Location Arbitrage Reference

## When to Use
- Products tradeable at multiple locations/exchanges
- ConversionObservation data available (bidPrice, askPrice, fees, tariffs)
- Examples: Orchids (P2), Magnificent Macarons (P3)

## Core Concept

When a product can be bought in one market and sold in another, profits come from the spread minus all transaction costs (transport, tariffs, exchange fees). The key is accounting for ALL costs accurately and detecting favorable regimes.

## Implementation

```python
def conversion_arb(self, state, product, data, local_limit=80, max_conversions=10):
    """
    Cross-market arbitrage via conversions.
    Conversions are limited to max_conversions per timestamp.
    """
    orders = []
    conversions = 0
    
    conv_obs = state.observations.conversionObservations.get(product)
    if not conv_obs:
        return orders, conversions
    
    local_depth = state.order_depths.get(product)
    if not local_depth:
        return orders, conversions
    
    pos = state.position.get(product, 0)
    
    # External market prices (from ConversionObservation)
    ext_bid = conv_obs.bidPrice       # Price we can sell at externally
    ext_ask = conv_obs.askPrice       # Price we can buy at externally
    transport = conv_obs.transportFees
    export_tariff = conv_obs.exportTariff
    import_tariff = conv_obs.importTariff
    
    # Cost to buy externally and sell locally
    import_cost = ext_ask + transport + import_tariff
    # Revenue from selling locally
    if local_depth.buy_orders:
        local_bid = max(local_depth.buy_orders.keys())
        import_profit = local_bid - import_cost
    else:
        import_profit = -float('inf')
    
    # Cost to buy locally and sell externally
    if local_depth.sell_orders:
        local_ask = min(local_depth.sell_orders.keys())
    else:
        local_ask = float('inf')
    export_revenue = ext_bid - transport - export_tariff
    export_profit = export_revenue - local_ask
    
    # Execute profitable conversions
    if import_profit > 0.5:  # Threshold to cover slippage
        # Import: buy externally, sell locally
        conv_qty = min(max_conversions, local_limit + pos)
        if conv_qty > 0:
            conversions = conv_qty
            # Also place sell orders locally to offload
            orders.append(Order(product, local_bid, -conv_qty))
    
    elif export_profit > 0.5:
        # Export: buy locally, sell externally
        conv_qty = min(max_conversions, local_limit - pos)
        if conv_qty > 0:
            conversions = -conv_qty  # Negative = export
            orders.append(Order(product, local_ask, conv_qty))
    
    return orders, conversions
```

## Regime Detection (Sunlight Index)

In P3, the sunlight index created different trading regimes for Macarons:

```python
def regime_based_arb(self, state, product, data):
    """
    Two-regime strategy based on sunlight index.
    Low sun → aggressive accumulation
    Normal sun → two-way arbitrage
    """
    conv_obs = state.observations.conversionObservations.get(product)
    if not conv_obs:
        return [], 0
    
    sunlight = conv_obs.sunlightIndex
    CSI_THRESHOLD = 50  # Calibrate based on data analysis
    
    if sunlight < CSI_THRESHOLD:
        # Low sunlight regime: prices tend to rise
        # Accumulate long position aggressively
        return self.accumulate_long(state, product, data)
    else:
        # Normal regime: standard two-way arbitrage
        return self.conversion_arb(state, product, data)

def accumulate_long(self, state, product, data, limit=80):
    """Aggressive long accumulation in favorable regime."""
    orders = []
    depth = state.order_depths.get(product, OrderDepth())
    pos = state.position.get(product, 0)
    
    remaining = limit - pos
    if remaining <= 0:
        return orders, 0
    
    # Buy at best available prices
    for ask in sorted(depth.sell_orders.keys()):
        qty = min(-depth.sell_orders[ask], remaining)
        if qty > 0:
            orders.append(Order(product, ask, qty))
            remaining -= qty
        if remaining <= 0:
            break
    
    # Also use conversions to import
    conversions = min(10, remaining)
    
    return orders, conversions
```

## Feature Engineering for ML Approach

The Frankfurt Hedgehogs (2nd P3) used gradient boosting for Macarons. Key features:

```python
def extract_features(self, state, product, data):
    """Features for ML-based arbitrage prediction."""
    conv = state.observations.conversionObservations.get(product)
    depth = state.order_depths.get(product)
    
    features = {}
    
    # Spread features
    if conv and depth and depth.buy_orders and depth.sell_orders:
        local_mid = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
        features["local_mid"] = local_mid
        features["ext_mid"] = (conv.bidPrice + conv.askPrice) / 2
        features["spread"] = features["ext_mid"] - local_mid
        features["total_cost"] = conv.transportFees + conv.importTariff
    
    # External indicators
    if conv:
        features["sugar_price"] = conv.sugarPrice
        features["sunlight"] = conv.sunlightIndex
    
    # Historical features
    spreads = data.get(f"{product}_spreads", [])
    if len(spreads) > 5:
        features["spread_ma5"] = sum(spreads[-5:]) / 5
        features["spread_ma20"] = sum(spreads[-20:]) / min(20, len(spreads))
    
    return features
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max conversions/timestamp | 10 | Hard limit from platform |
| Profit threshold | 0.5 ticks | Minimum to execute (covers slippage) |
| Sunlight threshold | 50 (varies) | Calibrate from data analysis |
| Position limit | 80-300 | Depends on product |
