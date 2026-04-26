# Bot Detection & Informed Trading Reference

## When to Use
- ALWAYS. Every single round, from Round 1 onward.
- Especially when market_trades contain identifiable trader IDs
- Critical in final rounds when trader identities are revealed

## Why This Matters

Bot detection was worth 10,000-50,000 SeaShells per round for top P3 teams. The "Olivia" pattern alone accounted for a significant portion of the Frankfurt Hedgehogs' winning edge.

## The Olivia Pattern (Documented from P3)

### Characteristics
- Buys exactly 15 lots at the daily low price
- Sells exactly 15 lots at the daily high price
- Active on: SQUID_INK, CROISSANTS
- Predictable: once she buys at a price, that price is likely the daily low

### Detection Code

```python
def detect_informed_traders(self, state, product, data):
    """
    Analyze market trades to identify informed traders.
    Track each trader's trade accuracy over time.
    """
    stats = data.get("trader_stats", {})
    prices = data.get(f"{product}_all_prices", [])
    
    # Get current mid
    depth = state.order_depths.get(product)
    if not depth or not depth.buy_orders or not depth.sell_orders:
        return stats
    
    current_mid = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
    prices.append(current_mid)
    data[f"{product}_all_prices"] = prices[-1000:]
    
    # Analyze each trade
    for trade in state.market_trades.get(product, []):
        for role, trader_id in [("buyer", trade.buyer), ("seller", trade.seller)]:
            if trader_id == "SUBMISSION":
                continue
            
            if trader_id not in stats:
                stats[trader_id] = {
                    "trades": 0,
                    "good_trades": 0,
                    "total_quantity": 0,
                    "products": {},
                    "quantities": {}
                }
            
            s = stats[trader_id]
            s["trades"] += 1
            s["total_quantity"] += trade.quantity
            
            # Track per-product activity
            if product not in s["products"]:
                s["products"][product] = 0
            s["products"][product] += 1
            
            # Track quantity patterns
            qty_key = str(trade.quantity)
            if qty_key not in s["quantities"]:
                s["quantities"][qty_key] = 0
            s["quantities"][qty_key] += 1
    
    # Retrospective accuracy check (did their buys precede price increases?)
    # This requires tracking trade prices and comparing to future prices
    # Simplified version: compare trade price to 10-period future average
    
    data["trader_stats"] = stats
    return stats

def identify_olivia(self, stats):
    """
    Find traders with Olivia-like characteristics:
    1. Consistent quantity (always same size)
    2. High accuracy (buys low, sells high)
    3. Active on specific products
    """
    candidates = []
    
    for trader_id, s in stats.items():
        if s["trades"] < 5:
            continue
        
        # Check for consistent quantity
        if s["quantities"]:
            most_common_qty = max(s["quantities"], key=s["quantities"].get)
            qty_consistency = s["quantities"][most_common_qty] / s["trades"]
            
            if qty_consistency > 0.7:  # >70% same quantity
                candidates.append({
                    "id": trader_id,
                    "trades": s["trades"],
                    "typical_qty": int(most_common_qty),
                    "consistency": qty_consistency,
                    "products": s["products"]
                })
    
    # Sort by consistency
    candidates.sort(key=lambda x: x["consistency"], reverse=True)
    return candidates
```

## Copy Trading Implementation

```python
def copy_informed_trader(self, state, product, informed_trader_id, data, limit):
    """
    Follow an informed trader's positions.
    When they buy → we buy. When they sell → we sell.
    """
    orders = []
    
    # Check if informed trader traded this timestamp
    for trade in state.market_trades.get(product, []):
        if trade.buyer == informed_trader_id:
            # Informed trader bought → bullish signal
            signal = "BUY"
            signal_price = trade.price
            signal_qty = trade.quantity
        elif trade.seller == informed_trader_id:
            # Informed trader sold → bearish signal
            signal = "SELL"
            signal_price = trade.price
            signal_qty = trade.quantity
        else:
            continue
        
        pos = state.position.get(product, 0)
        depth = state.order_depths[product]
        
        if signal == "BUY" and pos < limit:
            # Buy aggressively
            for ask in sorted(depth.sell_orders.keys()):
                qty = min(-depth.sell_orders[ask], limit - pos)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty
                if pos >= limit:
                    break
        
        elif signal == "SELL" and pos > -limit:
            # Sell aggressively
            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                qty = min(depth.buy_orders[bid], limit + pos)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty
                if pos <= -limit:
                    break
    
    return orders
```

## Daily Extreme Detection

```python
def track_daily_extremes(self, state, product, data):
    """
    Track daily min/max prices and flag trades at extremes.
    This is how the Olivia pattern was first discovered.
    """
    key = f"{product}_daily"
    daily = data.get(key, {"min": float('inf'), "max": float('-inf'), 
                           "trades_at_min": [], "trades_at_max": []})
    
    depth = state.order_depths.get(product)
    if not depth or not depth.buy_orders or not depth.sell_orders:
        return daily
    
    mid = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
    
    # Update daily extremes
    if mid < daily["min"]:
        daily["min"] = mid
        daily["trades_at_min"] = []  # Reset — new low
    if mid > daily["max"]:
        daily["max"] = mid
        daily["trades_at_max"] = []  # Reset — new high
    
    # Check trades at extremes
    for trade in state.market_trades.get(product, []):
        if abs(trade.price - daily["min"]) < 2:
            daily["trades_at_min"].append({
                "trader": trade.buyer,
                "qty": trade.quantity,
                "ts": state.timestamp
            })
        if abs(trade.price - daily["max"]) < 2:
            daily["trades_at_max"].append({
                "trader": trade.seller,
                "qty": trade.quantity,
                "ts": state.timestamp
            })
    
    data[key] = daily
    return daily
```

## Cross-Product Signal Integration

The highest-alpha technique from P3: using bot signals from one product to adjust strategies on correlated products.

```python
def get_informed_signal(self, data, informed_trader_id, component_product):
    """
    Determine the informed trader's net position on a component.
    Returns: -1 (short), 0 (flat), +1 (long)
    """
    stats = data.get("trader_stats", {})
    trader = stats.get(informed_trader_id, {})
    
    # Estimate net position from trade history
    # (This is approximate — we only see trades, not full position)
    buys = trader.get("buy_qty", {}).get(component_product, 0)
    sells = trader.get("sell_qty", {}).get(component_product, 0)
    net = buys - sells
    
    if net > 5:
        return 1   # Likely long
    elif net < -5:
        return -1  # Likely short
    return 0        # Flat/unknown
```

## Timeline for Bot Analysis

| Round | Action | Priority |
|-------|--------|----------|
| R1 | Start tracking all trader patterns | High |
| R2 | Identify consistent traders, begin copy trading | Critical |
| R3 | Expand to cross-product signals | Critical |
| R4 | Integrate signals into all strategies | Critical |
| R5 | Full exploitation with confirmed identities | Maximum |
