---
name: imc-prosperity-4
description: |
  Complete IMC Prosperity 4 algorithmic trading development assistant. Use this skill whenever the user mentions IMC Prosperity, Prosperity 4, trading competition, SeaShells, algorithmic trading challenge, market making algorithm, statistical arbitrage, options pricing for competition, backtesting trading strategies, or any task related to developing, testing, debugging, or optimizing a Python trading algorithm for the IMC Prosperity competition. Also trigger when the user mentions specific Prosperity products (EMERALDS, TOMATOES, baskets, vouchers, macarons), the Trader class, TradingState, OrderDepth, or prosperity4bt. This skill contains proven strategies from analysis of 30+ top-performing teams across 3 years of competition.
---

# IMC Prosperity 4 — Trading Algorithm Development Skill

You are helping a competitor develop, test, and optimize their Python trading algorithm for IMC Prosperity 4. This skill contains battle-tested strategies distilled from the top-performing teams across all 3 previous competitions.

## How This Skill Works

This skill has three layers:
1. **This file (SKILL.md)** — Core strategy knowledge, code patterns, and workflow guidance
2. **references/** — Detailed strategy implementations and the complete API reference
3. **scripts/** — Ready-to-use Python modules for common calculations (Black-Scholes, z-scores, mean reversion tests, etc.)

When the user asks you to implement a strategy, read the relevant reference file first, then use the scripts as building blocks.

## Core API (Prosperity 4 Datamodel)

```python
from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade, Listing, Observation, ConversionObservation

class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        """
        Called every timestamp. Returns (orders, conversions, traderData).
        - orders: Dict mapping product symbol to list of Orders
        - conversions: int (for cross-market conversion products)
        - traderData: JSON string persisted to next timestamp
        """
        pass
```

### Key Objects

| Object | Key Attributes | Notes |
|--------|---------------|-------|
| `TradingState` | `.timestamp`, `.listings`, `.order_depths`, `.own_trades`, `.market_trades`, `.position`, `.observations`, `.traderData` | Main input every tick |
| `OrderDepth` | `.buy_orders: Dict[int,int]`, `.sell_orders: Dict[int,int]` | sell volumes are NEGATIVE |
| `Order` | `(symbol, price, quantity)` | quantity>0 = buy, <0 = sell |
| `Trade` | `(symbol, price, quantity, buyer, seller, timestamp)` | buyer/seller="SUBMISSION" for your trades |
| `ConversionObservation` | `.bidPrice`, `.askPrice`, `.transportFees`, `.exportTariff`, `.importTariff`, `.sugarPrice`, `.sunlightIndex` | For conversion products |

### Critical Rules
- Orders do NOT carry over between timestamps — resubmit every tick
- Positions persist across all timestamps
- If any order would push position beyond limit, ALL orders for that product are cancelled
- `traderData` (string) is your only state persistence mechanism
- sell_orders volumes are stored as negative numbers

### Known Position Limits
- EMERALDS: 80
- TOMATOES: 80
- (Future products will have limits announced per round — typically 50-400)

## Strategy Decision Framework

When the user asks "how should I trade [product]?", follow this decision tree:

### Step 1: Classify the Product
- **Stable price** (constant fair value like ~10,000) → Market Making
- **Trending/random walk** (slow drift) → Market Making with rolling fair value
- **Volatile/spiking** (sharp moves, mean-reverting) → Spike Detection + Mean Reversion
- **Composite/basket** (weighted sum of components) → Statistical Arbitrage
- **Option/derivative** (on an underlying) → Black-Scholes + Delta Hedging
- **Cross-market** (traded at multiple locations) → Conversion Arbitrage

### Step 2: Implement the Strategy

For each strategy type, read the corresponding reference file:
- `references/market_making.md` — For stable and trending products
- `references/stat_arb.md` — For baskets, ETFs, and pairs
- `references/options.md` — For options, vouchers, and derivatives
- `references/conversion_arb.md` — For cross-market/conversion products
- `references/bot_detection.md` — For identifying and copying profitable bots

### Step 3: Test and Validate
Run `scripts/backtest_analyzer.py` to evaluate strategy performance. Always check:
1. Total PnL
2. Sharpe ratio (PnL / std of PnL per timestamp)
3. Maximum drawdown
4. Position utilization (are you using your limits effectively?)
5. Win rate by trade

## The Three-Phase Order Execution Pattern

Every top team uses this pattern. Implement it for EVERY product:

```python
def generate_orders(self, state, product, fair_value, position, limit):
    orders = []
    depth = state.order_depths[product]
    
    # PHASE 1: Market-Take (capture mispricings)
    for ask_price in sorted(depth.sell_orders.keys()):
        if ask_price < fair_value and position < limit:
            vol = min(-depth.sell_orders[ask_price], limit - position)
            orders.append(Order(product, ask_price, vol))
            position += vol
    
    for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
        if bid_price > fair_value and position > -limit:
            vol = min(depth.buy_orders[bid_price], limit + position)
            orders.append(Order(product, bid_price, -vol))
            position -= vol
    
    # PHASE 2: Position-Clear (flatten inventory)
    if position > 0:
        orders.append(Order(product, int(fair_value), -position))
    elif position < 0:
        orders.append(Order(product, int(fair_value), -position))
    
    # PHASE 3: Market-Make (passive quotes)
    edge = 2  # Adjust per product volatility
    buy_qty = limit - position
    sell_qty = limit + position
    if buy_qty > 0:
        orders.append(Order(product, int(fair_value) - edge, buy_qty))
    if sell_qty > 0:
        orders.append(Order(product, int(fair_value) + edge, -sell_qty))
    
    return orders
```

## Fair Value Estimation Methods

Use in order of preference:

### 1. Wall Mid (Best — used by 2nd place P3)
```python
def wall_mid(self, depth: OrderDepth) -> float:
    """Average of highest-volume bid and ask levels."""
    best_bid_vol, best_bid = max((v, p) for p, v in depth.buy_orders.items())
    best_ask_vol, best_ask = max((-v, p) for p, v in depth.sell_orders.items())
    return (best_bid + best_ask) / 2
```

### 2. Volume-Weighted Mid
```python
def vwap_mid(self, depth: OrderDepth) -> float:
    total_bid = sum(p * v for p, v in depth.buy_orders.items())
    total_bid_vol = sum(depth.buy_orders.values())
    total_ask = sum(p * (-v) for p, v in depth.sell_orders.items())
    total_ask_vol = sum(-v for v in depth.sell_orders.values())
    if total_bid_vol and total_ask_vol:
        return (total_bid/total_bid_vol + total_ask/total_ask_vol) / 2
    return 0
```

### 3. Rolling EMA
```python
def ema_fair_value(self, prices: list, span: int = 20) -> float:
    alpha = 2 / (span + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema
```

## Bot Detection Framework

This is one of the highest-alpha strategies. Every year has exploitable bots.

```python
def analyze_trader_profitability(self, market_trades, product, data):
    """Track each trader's cumulative PnL to find informed traders."""
    trader_stats = data.get("trader_stats", {})
    
    for trade in market_trades.get(product, []):
        for trader_id in [trade.buyer, trade.seller]:
            if trader_id == "SUBMISSION":
                continue
            if trader_id not in trader_stats:
                trader_stats[trader_id] = {"buys": 0, "sells": 0, "pnl_proxy": 0}
            
            stats = trader_stats[trader_id]
            if trader_id == trade.buyer:
                stats["buys"] += trade.quantity
            else:
                stats["sells"] += trade.quantity
    
    data["trader_stats"] = trader_stats
    return trader_stats

def detect_olivia_pattern(self, market_trades, product, data):
    """Detect traders that consistently buy at lows and sell at highs."""
    daily_prices = data.get("daily_prices", [])
    # Track running min/max per day
    # Flag trades at extreme prices with consistent quantities
    # The 'Olivia' pattern: always trades exactly 15 lots at daily extremes
    pass
```

## When the User Asks You To...

### "Help me analyze this data"
1. Read the CSV data files (prices, trades, observations)
2. Calculate basic statistics: mean, std, autocorrelation
3. Test for mean reversion: run `scripts/mean_reversion_test.py`
4. Identify correlations between products
5. Plot price series and highlight patterns
6. Check for bot patterns in trade data

### "Implement a market maker for [product]"
1. Read `references/market_making.md`
2. Determine fair value method based on product characteristics
3. Implement three-phase execution
4. Add inventory-aware spread adjustment
5. Backtest and show results

### "Build a stat arb strategy for baskets"
1. Read `references/stat_arb.md`
2. Run linear regression on basket vs components
3. Calculate spread and z-score
4. Implement mean-reversion trading on the spread
5. Add hedging logic for components

### "Price these options"
1. Read `references/options.md`
2. Implement Black-Scholes with r=0
3. Fit volatility smile across strikes
4. Calculate Greeks (delta, gamma, vega)
5. Implement delta hedging with underlying

### "Why is my strategy losing money?"
1. Analyze position history — are you hitting limits?
2. Check for adverse selection — are you getting picked off?
3. Verify fair value — is it actually fair?
4. Review parameter sensitivity — are you overfitting?
5. Check execution — are orders being cancelled due to limit violations?

## Testing Checklist

Before submitting any algorithm, verify:

- [ ] Position limits never exceeded (test with extreme scenarios)
- [ ] Orders re-generated every timestamp (no stale order bugs)
- [ ] traderData serialization/deserialization works correctly
- [ ] sell_order volumes handled as negative correctly
- [ ] Edge cases: empty order book, no trades, missing products
- [ ] Strategy works across all provided data days
- [ ] No import errors (only standard library + prosperity4bt.datamodel allowed)
- [ ] run() always returns (dict, int, str) — never None or wrong types
