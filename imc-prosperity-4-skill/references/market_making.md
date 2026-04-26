# Market Making Strategy Reference

## When to Use
- Products with identifiable fair values (stable or slowly drifting)
- Products with active order books and reasonable spreads
- Examples from past competitions: Pearls, Amethysts, Rainforest Resin, Kelp, Starfruit, Emeralds, Tomatoes

## Core Concept

Market making profits by continuously quoting buy and sell prices around a fair value, capturing the spread. The key challenge is managing inventory — you want to stay flat (position near zero) while maximizing the number of profitable round-trips.

## Implementation

### Stable Product Market Maker (e.g., Resin at 10,000)

```python
def market_make_stable(self, state, product, fair_value, limit):
    """
    For products with known, constant fair value.
    Frankfurt Hedgehogs approach: ~39,000 SeaShells/round on Resin.
    """
    orders = []
    depth = state.order_depths[product]
    pos = state.position.get(product, 0)
    
    # Phase 1: Take any mispriced orders
    for ask in sorted(depth.sell_orders.keys()):
        if ask < fair_value:
            qty = min(-depth.sell_orders[ask], limit - pos)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                pos += qty
    
    for bid in sorted(depth.buy_orders.keys(), reverse=True):
        if bid > fair_value:
            qty = min(depth.buy_orders[bid], limit + pos)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                pos -= qty
    
    # Phase 2: Clear inventory at fair value
    if pos > 0:
        orders.append(Order(product, int(fair_value), -pos))
    elif pos < 0:
        orders.append(Order(product, int(fair_value), -pos))
    
    # Phase 3: Passive quotes
    edge = 2  # ticks from fair value
    remaining_buy = limit - pos
    remaining_sell = limit + pos
    
    if remaining_buy > 0:
        orders.append(Order(product, int(fair_value) - edge, remaining_buy))
    if remaining_sell > 0:
        orders.append(Order(product, int(fair_value) + edge, -remaining_sell))
    
    return orders
```

### Dynamic Product Market Maker (e.g., Kelp, Starfruit)

```python
def market_make_dynamic(self, state, product, limit, data):
    """
    For products with slowly drifting fair value.
    Uses wall-mid pricing for best results.
    """
    orders = []
    depth = state.order_depths[product]
    pos = state.position.get(product, 0)
    
    # Calculate fair value using wall mid
    if not depth.buy_orders or not depth.sell_orders:
        return orders
    
    # Wall mid: average of highest-volume bid and ask
    best_bid_price = max(
        depth.buy_orders.keys(),
        key=lambda p: depth.buy_orders[p]
    )
    best_ask_price = min(
        depth.sell_orders.keys(),
        key=lambda p: -depth.sell_orders[p]  # volumes are negative
    )
    fair_value = (best_bid_price + best_ask_price) / 2
    
    # Track price history for volatility estimation
    prices = data.get(f"{product}_prices", [])
    prices.append(fair_value)
    if len(prices) > 50:
        prices = prices[-50:]
    data[f"{product}_prices"] = prices
    
    # Adjust edge based on recent volatility
    if len(prices) > 10:
        import statistics
        vol = statistics.stdev(prices[-10:])
        edge = max(1, int(vol * 0.5))
    else:
        edge = 2
    
    # Three-phase execution
    # Phase 1: Take mispricings
    for ask in sorted(depth.sell_orders.keys()):
        if ask < fair_value - 0.5 and pos < limit:
            qty = min(-depth.sell_orders[ask], limit - pos)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                pos += qty
    
    for bid in sorted(depth.buy_orders.keys(), reverse=True):
        if bid > fair_value + 0.5 and pos > -limit:
            qty = min(depth.buy_orders[bid], limit + pos)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                pos -= qty
    
    # Phase 2: Inventory management
    if abs(pos) > limit * 0.5:
        # Aggressive flattening when inventory is high
        if pos > 0:
            orders.append(Order(product, int(fair_value), -min(pos, 10)))
        else:
            orders.append(Order(product, int(fair_value), min(-pos, 10)))
    
    # Phase 3: Passive quotes with inventory skew
    inventory_skew = int(pos / limit * edge)  # Shift quotes away from inventory
    
    buy_price = int(fair_value) - edge - inventory_skew
    sell_price = int(fair_value) + edge - inventory_skew
    
    remaining_buy = limit - pos
    remaining_sell = limit + pos
    
    if remaining_buy > 0:
        orders.append(Order(product, buy_price, remaining_buy))
    if remaining_sell > 0:
        orders.append(Order(product, sell_price, -remaining_sell))
    
    return orders
```

## Inventory Management

The key insight from top teams: **inventory is the enemy of market making.** A large position reduces your ability to take profitable trades on one side.

### Inventory-Aware Spread Adjustment

```python
def adjusted_spread(self, base_edge, position, limit):
    """
    Widen spread on the side where you already have inventory.
    Tighten on the side where you want more.
    """
    inventory_ratio = position / limit  # -1 to +1
    
    buy_edge = base_edge + int(inventory_ratio * base_edge)   # Wider when long
    sell_edge = base_edge - int(inventory_ratio * base_edge)  # Tighter when long
    
    return max(1, buy_edge), max(1, sell_edge)
```

## Parameters to Optimize

| Parameter | Range | How to Tune |
|-----------|-------|-------------|
| Edge (spread) | 1-8 ticks | Start at 2, increase if getting adversely selected |
| Position soft limit | 30-60% of hard limit | Start at 50% |
| Rolling window | 10-50 | Shorter = more responsive, longer = more stable |
| Inventory skew factor | 0.3-1.0 | Higher = more aggressive inventory management |

## Common Mistakes

1. **Setting edge too tight** — You get adversely selected (buy right before drops)
2. **Setting edge too wide** — You never get filled
3. **Ignoring inventory** — Position hits limit, all orders cancelled
4. **Using simple mid instead of wall mid** — Leaves 10-30% of alpha on the table
