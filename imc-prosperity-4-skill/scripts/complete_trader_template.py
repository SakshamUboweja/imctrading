"""
IMC Prosperity 4 — Complete Trader Template
Based on strategies from 2nd-place teams across Prosperity 1, 2, and 3.

This template includes:
1. Three-phase order execution (market-take, position-clear, market-make)
2. Wall-mid fair value estimation
3. Mean reversion spike detection
4. Statistical arbitrage framework
5. Bot detection and copy trading
6. State persistence across timestamps
7. Position limit safety

Usage:
  1. Copy this file as your trader.py
  2. Configure PRODUCTS dict with your round's products
  3. Adjust parameters based on backtesting
  4. Run: prosperity4bt trader.py <round>-<day>
"""

from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade
import json
import math


class Trader:
    """
    Main trading algorithm. The backtester calls run() every timestamp.
    """

    # ============================================================
    # CONFIGURATION — Update per round
    # ============================================================

    PRODUCTS = {
        "EMERALDS": {
            "strategy": "market_make_stable",
            "fair_value": None,       # None = calculate dynamically
            "limit": 80,
            "edge": 2,               # Spread from fair value
            "take_edge": 0.5,        # Threshold to take vs fair value
        },
        "TOMATOES": {
            "strategy": "market_make_dynamic",
            "fair_value": None,
            "limit": 80,
            "edge": 3,
            "take_edge": 1.0,
        },
    }

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    def run(self, state: TradingState) -> tuple[dict, int, str]:
        # Parse persistent state
        data = json.loads(state.traderData) if state.traderData else {}

        all_orders = {}
        conversions = 0

        # Run bot detection on ALL products every timestamp
        self.update_bot_detection(state, data)

        # Generate orders per product
        for product, config in self.PRODUCTS.items():
            if product not in state.listings:
                continue

            strategy_name = config["strategy"]
            strategy_fn = getattr(self, strategy_name, None)
            if strategy_fn:
                orders = strategy_fn(state, product, config, data)
                if orders:
                    all_orders[product] = orders

        # Handle conversions (for products like MAGNIFICENT_MACARONS)
        conv_orders, conv_qty = self.handle_conversions(state, data)
        for product, orders in conv_orders.items():
            all_orders.setdefault(product, []).extend(orders)
        conversions = conv_qty

        # Serialize state (keep it compact)
        trader_data = json.dumps(data, default=str)

        return all_orders, conversions, trader_data

    # ============================================================
    # FAIR VALUE ESTIMATION
    # ============================================================

    def get_mid(self, depth: OrderDepth) -> float:
        """Simple mid price."""
        if not depth.buy_orders or not depth.sell_orders:
            return 0
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2

    def wall_mid(self, depth: OrderDepth) -> float:
        """
        Wall mid: average of highest-volume bid and ask levels.
        Used by Frankfurt Hedgehogs (2nd place P3).
        More robust than simple mid — captures where real liquidity sits.
        """
        if not depth.buy_orders or not depth.sell_orders:
            return self.get_mid(depth)

        # Find price level with maximum volume on each side
        best_bid_price = max(depth.buy_orders.keys(), key=lambda p: depth.buy_orders[p])
        best_ask_price = min(depth.sell_orders.keys(), key=lambda p: -depth.sell_orders[p])

        return (best_bid_price + best_ask_price) / 2

    def ema_fair_value(self, prices: list, span: int = 20) -> float:
        """Exponential moving average fair value."""
        if not prices:
            return 0
        alpha = 2 / (span + 1)
        result = prices[0]
        for p in prices[1:]:
            result = alpha * p + (1 - alpha) * result
        return result

    # ============================================================
    # STRATEGY: STABLE MARKET MAKING
    # ============================================================

    def market_make_stable(self, state, product, config, data):
        """
        For products with known/stable fair value.
        Three-phase execution: take → clear → make.
        Target: ~35,000-40,000 SeaShells/round (based on P3 Resin)
        """
        orders = []
        depth = state.order_depths.get(product)
        if not depth:
            return orders

        limit = config["limit"]
        edge = config["edge"]
        take_edge = config["take_edge"]
        pos = state.position.get(product, 0)

        # Determine fair value
        if config["fair_value"] is not None:
            fv = config["fair_value"]
        else:
            fv = self.wall_mid(depth)

        if fv == 0:
            return orders

        # PHASE 1: Take mispricings
        for ask in sorted(depth.sell_orders.keys()):
            if ask < fv - take_edge and pos < limit:
                qty = min(-depth.sell_orders[ask], limit - pos)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid > fv + take_edge and pos > -limit:
                qty = min(depth.buy_orders[bid], limit + pos)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty

        # PHASE 2: Clear inventory at fair value
        if pos > 0:
            orders.append(Order(product, int(round(fv)), -pos))
        elif pos < 0:
            orders.append(Order(product, int(round(fv)), -pos))

        # PHASE 3: Passive market making
        remaining_buy = limit - pos
        remaining_sell = limit + pos

        if remaining_buy > 0:
            orders.append(Order(product, int(round(fv)) - edge, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, int(round(fv)) + edge, -remaining_sell))

        return orders

    # ============================================================
    # STRATEGY: DYNAMIC MARKET MAKING
    # ============================================================

    def market_make_dynamic(self, state, product, config, data):
        """
        For products with drifting fair value.
        Uses wall-mid + inventory-aware spread adjustment.
        """
        orders = []
        depth = state.order_depths.get(product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return orders

        limit = config["limit"]
        base_edge = config["edge"]
        take_edge = config["take_edge"]
        pos = state.position.get(product, 0)

        # Calculate fair value
        fv = self.wall_mid(depth)

        # Track price history for volatility
        key = f"{product}_prices"
        prices = data.get(key, [])
        prices.append(fv)
        if len(prices) > 100:
            prices = prices[-100:]
        data[key] = prices

        # Adjust edge based on volatility
        if len(prices) > 10:
            recent = prices[-10:]
            mean_p = sum(recent) / len(recent)
            vol = math.sqrt(sum((p - mean_p)**2 for p in recent) / len(recent))
            edge = max(1, int(vol * 0.5 + base_edge * 0.5))
        else:
            edge = base_edge

        # Inventory skew: shift quotes away from current position
        inventory_ratio = pos / limit if limit > 0 else 0
        skew = int(inventory_ratio * edge)

        # PHASE 1: Take mispricings
        for ask in sorted(depth.sell_orders.keys()):
            if ask < fv - take_edge and pos < limit:
                qty = min(-depth.sell_orders[ask], limit - pos)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    pos += qty

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid > fv + take_edge and pos > -limit:
                qty = min(depth.buy_orders[bid], limit + pos)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    pos -= qty

        # PHASE 2: Aggressive clear when inventory high
        soft_limit = int(limit * 0.6)
        if abs(pos) > soft_limit:
            clear_qty = min(abs(pos) - soft_limit, 10)
            if pos > 0:
                orders.append(Order(product, int(round(fv)), -clear_qty))
            else:
                orders.append(Order(product, int(round(fv)), clear_qty))

        # PHASE 3: Passive quotes with skew
        buy_price = int(round(fv)) - edge - skew
        sell_price = int(round(fv)) + edge - skew

        remaining_buy = limit - pos
        remaining_sell = limit + pos

        if remaining_buy > 0:
            orders.append(Order(product, buy_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, sell_price, -remaining_sell))

        return orders

    # ============================================================
    # STRATEGY: MEAN REVERSION (Volatile Products)
    # ============================================================

    def mean_reversion(self, state, product, config, data):
        """
        For volatile products with sharp spikes and mean reversion.
        Based on Squid Ink strategy from P3 top teams.
        """
        orders = []
        depth = state.order_depths.get(product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return orders

        limit = config["limit"]
        pos = state.position.get(product, 0)

        mid = self.get_mid(depth)

        # Track price history
        key = f"{product}_prices"
        prices = data.get(key, [])
        prices.append(mid)
        if len(prices) > 200:
            prices = prices[-200:]
        data[key] = prices

        if len(prices) < 20:
            return orders  # Not enough data yet

        # Z-score based mean reversion
        window = 20
        recent = prices[-window:]
        mean = sum(recent) / len(recent)
        std = math.sqrt(sum((p - mean)**2 for p in recent) / len(recent))

        if std < 1:
            return orders

        z = (mid - mean) / std

        # Entry thresholds
        ENTRY_Z = config.get("entry_z", 2.5)
        EXIT_Z = config.get("exit_z", 0.5)

        if z < -ENTRY_Z and pos < limit:
            # Price crashed — buy expecting reversion
            qty = min(10, limit - pos)  # Conservative sizing
            best_ask = min(depth.sell_orders.keys())
            orders.append(Order(product, best_ask, qty))

        elif z > ENTRY_Z and pos > -limit:
            # Price spiked — sell expecting reversion
            qty = min(10, limit + pos)
            best_bid = max(depth.buy_orders.keys())
            orders.append(Order(product, best_bid, -qty))

        elif abs(z) < EXIT_Z and pos != 0:
            # Mean reverted — close position
            if pos > 0:
                best_bid = max(depth.buy_orders.keys())
                orders.append(Order(product, best_bid, -pos))
            else:
                best_ask = min(depth.sell_orders.keys())
                orders.append(Order(product, best_ask, -pos))

        return orders

    # ============================================================
    # STRATEGY: STATISTICAL ARBITRAGE
    # ============================================================

    def stat_arb(self, state, product, config, data):
        """
        For basket/ETF products. Trade spread between basket and synthetic value.
        Configure with: components dict, weights, intercept
        """
        orders = []
        depth = state.order_depths.get(product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return orders

        limit = config["limit"]
        pos = state.position.get(product, 0)
        components = config.get("components", {})
        weights = config.get("weights", {})
        intercept = config.get("intercept", 0)

        # Calculate synthetic value
        basket_mid = self.get_mid(depth)
        synthetic = intercept

        for comp, weight in weights.items():
            comp_depth = state.order_depths.get(comp)
            if not comp_depth or not comp_depth.buy_orders or not comp_depth.sell_orders:
                return orders  # Can't calculate synthetic
            synthetic += weight * self.get_mid(comp_depth)

        spread = basket_mid - synthetic

        # Track spread history
        key = f"{product}_spread"
        spreads = data.get(key, [])
        spreads.append(spread)
        if len(spreads) > 100:
            spreads = spreads[-100:]
        data[key] = spreads

        if len(spreads) < 10:
            return orders

        # Z-score on spread
        window = min(20, len(spreads))
        recent = spreads[-window:]
        mean_s = sum(recent) / len(recent)
        std_s = math.sqrt(sum((s - mean_s)**2 for s in recent) / len(recent))

        if std_s < 0.5:
            return orders

        z = (spread - mean_s) / std_s

        ENTRY = config.get("entry_z", 2.0)
        EXIT = config.get("exit_z", 0.5)

        if z > ENTRY and pos > -limit:
            # Basket overpriced — sell
            best_bid = max(depth.buy_orders.keys())
            qty = min(depth.buy_orders[best_bid], limit + pos, 5)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

        elif z < -ENTRY and pos < limit:
            # Basket underpriced — buy
            best_ask = min(depth.sell_orders.keys())
            qty = min(-depth.sell_orders[best_ask], limit - pos, 5)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

        elif abs(z) < EXIT and pos != 0:
            # Close at mean
            if pos > 0:
                best_bid = max(depth.buy_orders.keys())
                orders.append(Order(product, best_bid, -pos))
            else:
                best_ask = min(depth.sell_orders.keys())
                orders.append(Order(product, best_ask, -pos))

        return orders

    # ============================================================
    # BOT DETECTION
    # ============================================================

    def update_bot_detection(self, state, data):
        """Track all trader activity to identify informed bots."""
        stats = data.get("bot_stats", {})

        for product in state.listings:
            for trade in state.market_trades.get(product, []):
                for trader_id in [trade.buyer, trade.seller]:
                    if trader_id == "SUBMISSION" or not trader_id:
                        continue

                    if trader_id not in stats:
                        stats[trader_id] = {"trades": 0, "qty_counts": {}, "products": {}}

                    s = stats[trader_id]
                    s["trades"] += 1

                    # Track quantity patterns
                    q = str(trade.quantity)
                    s["qty_counts"][q] = s["qty_counts"].get(q, 0) + 1

                    # Track product activity
                    s["products"][product] = s["products"].get(product, 0) + 1

        # Keep stats compact — only top traders
        if len(stats) > 50:
            sorted_traders = sorted(stats.items(), key=lambda x: x[1]["trades"], reverse=True)
            stats = dict(sorted_traders[:30])

        data["bot_stats"] = stats

    def get_informed_traders(self, data, min_trades=10):
        """Identify traders with suspiciously consistent patterns."""
        stats = data.get("bot_stats", {})
        candidates = []

        for trader_id, s in stats.items():
            if s["trades"] < min_trades:
                continue

            # Check quantity consistency
            if s["qty_counts"]:
                most_common = max(s["qty_counts"].values())
                consistency = most_common / s["trades"]

                if consistency > 0.6:
                    candidates.append((trader_id, consistency, s["trades"]))

        return sorted(candidates, key=lambda x: x[1], reverse=True)

    # ============================================================
    # CONVERSIONS
    # ============================================================

    def handle_conversions(self, state, data):
        """Handle conversion-based arbitrage (e.g., Macarons)."""
        orders = {}
        conversions = 0

        for product, conv_obs in state.observations.conversionObservations.items():
            if product not in state.listings:
                continue

            local_depth = state.order_depths.get(product)
            if not local_depth or not local_depth.buy_orders:
                continue

            pos = state.position.get(product, 0)
            local_bid = max(local_depth.buy_orders.keys())

            import_cost = conv_obs.askPrice + conv_obs.transportFees + conv_obs.importTariff
            profit = local_bid - import_cost

            if profit > 1.0:
                conv_qty = min(10, 80 - pos)  # Max 10 conversions
                if conv_qty > 0:
                    conversions = conv_qty
                    orders[product] = [Order(product, local_bid, -conv_qty)]

        return orders, conversions
