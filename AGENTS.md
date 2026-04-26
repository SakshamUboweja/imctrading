# IMC Prosperity 4 — Codex Context

## Prime Directive

**There are guaranteed exploitable loopholes in every round's data.** The competition is designed so that careful quantitative analysis reveals predictable structure that can be traded profitably. Your job is to find ALL of it and extract maximum value.

The predictability can take virtually any form. It is NOT limited to a fixed set of techniques. The right approach for any given product might be market making, trend following, mean reversion, statistical arbitrage, options pricing, conversion arbitrage, bot detection, regime switching, lead-lag relationships, order flow prediction, volatility trading, calendar effects, cross-product hedging, or some combination of these that has no standard name. The data tells you what works. You do not choose a strategy and fit data to it — you analyze data exhaustively and let the structure dictate the strategy.

**The analytical pipeline is: data → patterns → model → algorithm → backtest → optimize.**

Never skip from data to algorithm. The pattern discovery step is where all the alpha lives.

## Analytical Pipeline (Execute for Every New Round)

### Phase 1: Raw Data Inventory

Load every CSV file for the round. Map the full landscape:

```
- What products exist? (prices CSV `product` column, trades CSV `symbol` column)
- Which products are NEW vs carried over from previous rounds?
- How many timestamps per day? (typically 1M at 100-tick intervals in training)
- Does observations data exist? (signals conversion/cross-market products)
- What are the position limits? (check backtester data.py, patch if needed)
- What fields are populated in the trades CSV? (buyer/seller IDs? currency?)
```

### Phase 2: Exhaustive Statistical Profiling

For EACH product, compute everything. Do not pre-filter what you think matters — compute it all, then look for anomalies. The exploit is often in the metric you wouldn't have thought to check.

**Price dynamics:**
```
- Mean, std, min, max, skewness, kurtosis of mid prices
- Start/end price, total drift, drift rate per timestamp
- Drift consistency: is the rate constant across days? Does it accelerate/decelerate?
- Linear regression of price vs time: slope, intercept, R², residual std
- Quadratic/polynomial fit: is there curvature?
- Rolling mean and rolling std (windows: 10, 50, 200, 1000)
- Price change distribution: is it normal? Fat tails? Bimodal?
```

**Stationarity and memory:**
```
- Hurst exponent (H < 0.5 mean-reverting, H = 0.5 random walk, H > 0.5 trending)
- Autocorrelation function at lags 1, 2, 3, 5, 10, 20, 50, 100
- Partial autocorrelation function (identifies AR order)
- ADF test statistic and p-value (is the series stationary?)
- Half-life of mean reversion (if stationary)
- Variance ratio test (random walk vs mean-reverting at multiple horizons)
```

**Volatility structure:**
```
- Rolling volatility (windows 10, 50, 200)
- Volatility clustering: autocorrelation of absolute returns
- Intraday volatility pattern: is vol higher at open/close vs midday?
- Volatility regime detection: are there distinct high/low vol periods?
- GARCH-like effects: does past vol predict future vol?
```

**Order book microstructure:**
```
- Bid-ask spread: mean, median, percentiles, distribution shape
- Spread dynamics: is spread constant, cyclical, or regime-dependent?
- Number of price levels visible (L1 only? L2? L3?)
- Volume distribution across levels (where does the big liquidity sit?)
- One-sided quote frequency (how often is only bid or only ask present?)
- Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol) at each level
- Imbalance predictive power: correlation with forward returns at multiple horizons
```

**Wall detection and fair value discovery (proven high-alpha in Round 1):**
```
- For each timestamp, find the price level with maximum volume on bid and ask sides
- Compute wall_mid = (wall_bid + wall_ask) / 2
- Fit wall_mid to time (linear, polynomial) — compute residual std
- Compare wall_mid residual std vs simple_mid residual std
- If wall_mid is more precise, it's a better fair value indicator
- Check: is wall_spread (wall_ask - wall_bid) constant? What's its distribution?
- Check: does wall_bid + wall_ask = constant? (implies symmetric structure)
```

**Return predictability (the core alpha search):**
```
- For EVERY computed signal (imbalance, wall deviation, momentum, vol, spread width, etc.):
  compute correlation with forward returns at horizons 1, 3, 5, 10, 20, 50
- Any correlation |r| > 0.1 is potentially tradeable
- Any correlation |r| > 0.3 is definitely tradeable
- Test non-linear relationships: does the signal predict direction but not magnitude?
- Test threshold effects: does signal matter only when |signal| > some value?
- Compute information coefficient (IC) across time: is the signal stable or decaying?
```

### Phase 3: Cross-Product Analysis

This phase is critical when multiple products exist. The highest-alpha strategies in previous competitions came from cross-product relationships.

```
- Correlation matrix of returns across all products
- Cointegration tests between all product pairs
- Lead-lag analysis: does product A's return predict product B's?
  (compute cross-correlation at lags ±1, ±2, ±5, ±10)
- If a basket/ETF exists: regress basket on components
  basket = intercept + w1*comp1 + w2*comp2 + ...
  Then compute spread = actual - synthetic, test for mean reversion
- If options exist: compute implied vol across strikes, fit smile
  Then look for mispriced strikes relative to the smile
- If conversion products exist: compute import/export profitability
  including ALL fees (transport, tariffs, currency)
  Look for regime-dependent profitability (external indicators like sunlight, sugar price)
```

### Phase 4: Trade Flow & Bot Analysis

Market trades (involving other participants) contain exploitable information:

```
- Extract ALL unique trader IDs from buyer/seller fields
- For each trader: track trade count, typical quantity, product preference
- Quantity consistency: does a trader always trade the same size? (>70% = likely a bot)
- Timing consistency: does a trader appear at regular intervals?
- Direction accuracy: do their buys precede price increases? Track over time.
- Extreme detection: do they buy at daily lows and sell at daily highs? ("Olivia pattern")
- Copy-trading potential: if a trader is consistently profitable, follow their trades
- Cross-product signals: does a bot's activity on product A predict moves on product B?
```

**Even if buyer/seller fields are empty in the current round, the MARKET TRADE prices and sizes contain information about what other participants are doing. Analyze the distribution of market trade prices relative to mid — are they systematically above or below? This reveals the net direction of informed flow.**

### Phase 5: Regime and Structural Analysis

Look for non-stationarities and structural breaks:

```
- Split each day into thirds (early/mid/late): do statistics differ?
- Split across days: are there day-of-week effects or progressive changes?
- If external observations exist (sunlight, sugar price, etc.):
  correlate these with product returns, volatility, and spread width
- Detect regime switches: periods where the mean, vol, or drift changes
- Changepoint detection: are there sharp structural breaks in the price series?
- Seasonal patterns within a day: do prices tend to rise/fall at specific times?
```

### Phase 6: Algorithm Design

**Only after completing Phases 2-5** do you design the algorithm. The design should be dictated by the discovered patterns, not by a pre-chosen strategy template.

For each product, determine:

1. **Fair value method**: Whatever the data says is most precise. Candidates include wall_mid, simple mid, EMA, linear formula, external anchor (like 10,000), regression-based synthetic value, or some combination. Pick the method with the lowest residual std.

2. **Directional bias**: If the product trends, compute the optimal bias via parameter sweep. If it mean-reverts, bias = 0. If it's a basket mispricing, bias comes from the spread z-score.

3. **Quote placement**: Test join-queue (offset=0) vs jump-queue (offset=1+). Test various edge widths. The right answer depends on the product's spread width and fill dynamics.

4. **Take aggressiveness**: How many ticks of mispricing justify taking? Sweep take_edge from 0.5 to 5.0.

5. **Inventory management**: Test different clearing thresholds (50% to 100% of limit). For trending products, clearing is often counterproductive. For mean-reverting products, aggressive clearing is fine.

6. **Signal integration**: Incorporate every signal with |correlation| > 0.1 from Phase 2. Use as quote skewing (shift bid/ask by signal × factor) or as take/no-take filter.

7. **Cross-product logic**: If Phase 3 found relationships, implement hedging, stat arb, or lead-lag trading.

8. **Bot following**: If Phase 4 found informed traders, implement copy-trading.

### Phase 7: Backtest & Optimize

**CRITICAL**: The live server runs ~100,000 timestamps per session, not the full 1,000,000 in training data. Always test on BOTH:

```bash
# Full training data
prosperity4bt trader.py N --data ./data/ --merge-pnl --no-out

# Truncated to match live server (100k timestamps)
# First truncate the CSVs: keep only rows where timestamp < 100000
# Then run backtester on truncated data
```

**Parameter sweep protocol:**
```
1. Sweep one parameter at a time, holding others at defaults
2. For each value, record: total PnL, min-day PnL, per-product breakdown
3. Pick the value that maximizes TOTAL PnL subject to min-day > some floor
4. After all single-parameter sweeps, test the top 2-3 promising combinations
5. Verify best params on BOTH full and truncated data
```

**Anti-overfitting rules:**
- If a parameter only helps on one day and hurts others, reject it
- If the optimal value is at an extreme of the search range, extend the range to find the peak
- If removing a parameter (setting to neutral/0) gives >90% of the PnL, it's not worth the complexity
- Every parameter must have a structural justification from the data analysis
- If you can't explain WHY a parameter value works, it's probably overfit

### Phase 8: Final Validation

Before submission:
```
- [ ] PnL is positive on ALL training days
- [ ] No "Orders exceeded limit" warnings in sandbox log
- [ ] Algorithm handles empty order books gracefully (no crashes)
- [ ] traderData serialization works (test with json.dumps/loads)
- [ ] sell_orders negative volumes handled correctly everywhere
- [ ] Import is `from datamodel import ...` (NOT `from prosperity4bt.datamodel import ...`)
- [ ] run() returns (dict, int, str) — never None
- [ ] Only standard library imports (no numpy, pandas, scipy in submission)
- [ ] Position budgeting: total buy + pos <= limit AND pos - total sell >= -limit
```

## Technique Reference Library

The following techniques have been proven in previous Prosperity competitions. Any of them could be relevant to any round. They can also be COMBINED — the best strategies often layer multiple techniques.

### Market Making
- Three-phase execution: take → clear → quote
- Wall-mid fair value (highest-volume level, not simple mid)
- Inventory-aware spread skewing
- Imbalance-based quote adjustment
- Volatility-adaptive edge width
- Reference: `references/market_making.md`

### Statistical Arbitrage
- Basket/ETF spread trading (synthetic value vs actual)
- Pairs trading with cointegrated products
- Z-score entry/exit with rolling windows
- Component hedging (partial or full)
- Reference: `references/stat_arb.md`

### Options & Derivatives
- Black-Scholes pricing (risk-free rate = 0 in Prosperity)
- Implied volatility surface fitting (quadratic smile)
- Delta hedging with underlying
- Mean reversion on IV (if market IV deviates from smile)
- Pure Python implementations (no scipy): `scripts/calculations.py`
- Reference: `references/options.md`

### Conversion Arbitrage
- Cross-market import/export with fee accounting
- Regime detection via external indicators (sunlight, sugar price)
- Max 10 conversions per timestamp
- Reference: `references/conversion_arb.md`

### Bot Detection & Copy Trading
- Trader ID tracking across timestamps
- Quantity consistency analysis (>70% same size = bot)
- Trade accuracy scoring (buys precede price increases)
- "Olivia" pattern: consistent size at daily extremes
- Cross-product signal extraction from informed bots
- Reference: `references/bot_detection.md`

### Trend Following
- Linear drift detection (regression slope)
- Directional bias in fair value (shift FV by drift × lookahead)
- Asymmetric quoting (tight on accumulation side, wide on exit side)
- Dynamic bias based on remaining session time

### Mean Reversion
- ADF test for stationarity
- Half-life estimation for position sizing
- Z-score entry/exit with dynamic thresholds
- Bollinger band-style approaches

### Volatility Trading
- Realized vs implied vol spread
- Straddle/strangle construction from available products
- Vol regime detection and regime-switching strategies

### Order Flow Analysis
- Trade imbalance (net buy vs sell volume over rolling window)
- Large trade detection (unusual size → informed flow)
- VWAP analysis (is VWAP above or below mid?)
- Toxic flow detection (adverse selection measurement)

### Time Series Techniques
- EMA, DEMA, TEMA for smoothing
- Kalman filter for dynamic fair value estimation
- Momentum indicators (rate of change, RSI-like)
- Mean reversion indicators (distance from rolling mean in std units)

### Information-Theoretic
- Mutual information between signals and future returns
- Entropy of price changes (low entropy = predictable = exploitable)
- Transfer entropy between products (directional information flow)

## Competition API Reference

```python
from datamodel import Order, OrderDepth, TradingState, Trade, Listing, Observation, ConversionObservation

class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        # state.timestamp: int (0, 100, 200, ...)
        # state.order_depths: Dict[str, OrderDepth]
        #   OrderDepth.buy_orders: Dict[int, int]  (price → positive volume)
        #   OrderDepth.sell_orders: Dict[int, int]  (price → NEGATIVE volume)
        # state.position: Dict[str, int]  (product → current position)
        # state.own_trades: Dict[str, List[Trade]]  (your fills from LAST tick)
        # state.market_trades: Dict[str, List[Trade]]  (other people's trades)
        # state.traderData: str  (your persisted state from last tick)
        # state.observations: Observation  (for conversion products)
        #   .conversionObservations: Dict[str, ConversionObservation]
        #     ConversionObservation: .bidPrice, .askPrice, .transportFees,
        #       .exportTariff, .importTariff, .sugarPrice, .sunlightIndex
        #
        # Order(symbol: str, price: int, quantity: int)
        #   quantity > 0 = buy, quantity < 0 = sell
        #
        # Returns: (orders_dict, conversions_int, traderData_string)
        pass
```

**Critical implementation rules:**
- Orders do NOT carry over — resubmit every timestamp
- Positions persist across timestamps
- `sell_orders` volumes are NEGATIVE — use `-depth.sell_orders[price]` for absolute volume
- Position limit check is ALL-OR-NOTHING: if `pos + sum(all_buy_qtys) > limit` OR `pos - sum(all_sell_qtys) < -limit`, ALL orders for that product are cancelled. Use a budget system.
- `traderData` is a string — use `json.dumps()`/`json.loads()` for state persistence
- Only standard library on submission server — no numpy, pandas, scipy
- Conversions: max 10 per timestamp, for cross-market products only
- `run()` must always return `(dict, int, str)` — never None, never crash

## Local Backtesting

```bash
pip install prosperity4bt

# CRITICAL: Patch position limits for the current round's products
# Edit: /path/to/site-packages/prosperity4bt/data.py
# Add each product and its limit to the LIMITS dict

# Data directory structure:
# data/roundN/prices_round_N_day_{-2,-1,0}.csv
# data/roundN/trades_round_N_day_{-2,-1,0}.csv
# data/roundN/observations_round_N_day_{-2,-1,0}.csv  (if applicable)

# Run:
prosperity4bt trader.py N --data ./data/ --merge-pnl --no-out

# For submission: `from datamodel import ...`
# For local testing: `from prosperity4bt.datamodel import ...`
```

## Lessons from Round 1 (Structural, Likely Generalizable)

These are patterns about HOW the competition works, not about specific products:

1. **The live server runs ~100k timestamps, not 1M.** Always optimize for short sessions too.
2. **Wall_mid (highest-volume level) was 5x more precise than simple mid for fair value.** Check this for every new product.
3. **Quote placement dominates strategy selection.** Being at `best_bid` vs `best_bid - 4` was a 3-4x PnL difference with identical strategy logic.
4. **Position limit enforcement is all-or-nothing.** Budget buy/sell capacity across execution phases.
5. **Joining the queue (offset=0) beat jumping it (offset=1).** More spread per fill outweighed fewer fills.
6. **Drift rate was exactly 0.001/tick for the trending product, consistent to 6 decimal places across all data sources.** When the competition builds in a pattern, it's precise and deterministic.
7. **The order book has a two-tier structure: small L1 "skimmer" flow and large L2 "wall" flow.** The wall is the real market maker. This may or may not persist in future rounds, but the principle of looking deeper than L1 always applies.

## File Layout

```
context.md                          — This file
trader.py                           — Current submission algorithm
data/roundN/                        — Training data CSVs
references/                         — Strategy reference documents
  market_making.md
  stat_arb.md
  options.md
  conversion_arb.md
  bot_detection.md
scripts/
  calculations.py                   — Pure Python quant utilities (BS, regression, etc.)
  backtest_analyzer.py              — Log parser and performance reporter
  complete_trader_template.py       — Template with all strategy skeletons
```