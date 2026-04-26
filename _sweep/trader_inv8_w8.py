"""
IMC Prosperity 4 — Round 2 Trader (wall-based MM, ACO bias dropped)

CORE INSIGHT (from live server log analysis):
The L2 order book levels (volume >= 15) are the REAL market maker's quotes.
The L1 levels (smaller volume) are "skimmer" noise traders.

For IPR: wall_mid follows 12000 + 0.001*t exactly (residual std ~0.20 ticks on R2)
For ACO: wall_mid ≈ 10001 flat; no drift, no bias needed

STRATEGY (re-verified on Round 2 CSVs):
- Detect the wall (highest-volume level on each side)
- Fair value = (wall_bid + wall_ask) / 2
- For IPR: add bias of 7.5 to fair (extracts the upward drift asymmetrically)
- For ACO: NO bias (Round 2 re-sweep: bias 0 beats bias 0.75 by +10k on 1M, +533 on 100k)
- Take aggressively when prices cross fair (take_edge = 1)
- Passive quotes JOIN the best bid/ask (bid_offset=0, ask_offset=0)
- Never clear inventory (clear_thresh=1.0) — let market reversion handle it
- Imbalance skew: ±3 ticks based on L1 buy/sell pressure
- Inventory skew: ±4 ticks based on current position

VALIDATED ON ROUND 2:
- Full 1M session: 302,129 Zyrex total (100k+ per day, all days positive)
- Truncated 100k session: 28,494 total (9.1k–9.9k per day)

ROUND 2 MAF (Market Access Fee):
The MAF is declared as the class-level constant MAF (also mirrored at module
level). Set to the Zyrex amount you are willing to pay if your bid lands in the
top 50% of all MAF bids submitted. Winning → full order-book visibility; losing
→ you keep a stochastic ~75% subset of quotes but pay nothing. Hard ceiling on
a rational bid is the expected incremental PnL from the extra 25% access (~7k
for our current trader); mid-bid tradeoff target is ~20–30% of that.
"""

from datamodel import Order, OrderDepth, TradingState
import json
import math


# --- Market Access Fee (Round 2) -------------------------------------------
# Set to 0 to opt out (free, keep base volume allocation).
# Any positive integer enters you in the top-50%-wins auction; if you win,
# you pay exactly this number of Zyrex and gain access to the extra 25%.
MAF = 1500
# ---------------------------------------------------------------------------


class Trader:
    LIMITS = {"INTARIAN_PEPPER_ROOT": 80, "ASH_COATED_OSMIUM": 80}

    # Market Access Fee (mirror of module-level MAF — whichever the grader reads)
    MAF = MAF

    # Parameters — tuned on 100k-timestamp sessions (Round 2 re-sweep)
    IPR_BIAS = 7.5
    ACO_BIAS = 0
    BID_OFFSET = 0
    ASK_OFFSET = 0
    TAKE_EDGE = 1.0
    IMB_FACTOR = 3
    INV_FACTOR = 8
    CLEAR_THRESH = 1.0
    WALL_MIN_VOL = 8
    MAX_TAKE_PER_LEVEL = 25

    def run(self, state: TradingState):
        data = json.loads(state.traderData) if state.traderData else {}
        all_orders = {}
        for product in self.LIMITS:
            if product not in state.order_depths:
                continue
            depth = state.order_depths[product]
            if not depth.buy_orders or not depth.sell_orders:
                continue
            pos = state.position.get(product, 0)
            limit = self.LIMITS[product]
            orders = self.trade(state, depth, product, pos, limit, data)
            if orders:
                all_orders[product] = orders
        return all_orders, 0, json.dumps(data)

    def find_wall_bid(self, depth):
        """Find the highest-volume bid price (the market maker wall)."""
        best_p, best_v = None, 0
        for p, v in depth.buy_orders.items():
            if v >= self.WALL_MIN_VOL and v > best_v:
                best_v, best_p = v, p
        return best_p

    def find_wall_ask(self, depth):
        """Find the highest-volume ask price (volumes stored as negative)."""
        best_p, best_v = None, 0
        for p, v in depth.sell_orders.items():
            if -v >= self.WALL_MIN_VOL and -v > best_v:
                best_v, best_p = -v, p
        return best_p

    def get_imbalance(self, depth):
        """L1 order book imbalance: +1 (all buyers) to -1 (all sellers)."""
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bv = depth.buy_orders[best_bid]
        av = -depth.sell_orders[best_ask]
        t = bv + av
        return (bv - av) / t if t > 0 else 0.0

    def trade(self, state, depth, product, pos, limit, data):
        orders = []

        # --- FAIR VALUE via wall detection ---
        wb = self.find_wall_bid(depth)
        wa = self.find_wall_ask(depth)
        if wb and wa:
            fair = (wb + wa) / 2.0
        else:
            # Fallback to simple mid when walls aren't present
            fair = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

        # Product-specific bias
        if product == "INTARIAN_PEPPER_ROOT":
            fair += self.IPR_BIAS
        else:
            fair += self.ACO_BIAS

        imb = self.get_imbalance(depth)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        # Position budget
        max_buy = limit - pos
        max_sell = limit + pos
        if max_buy <= 0 and max_sell <= 0:
            return orders
        used_buy = 0
        used_sell = 0

        # --- PHASE 1: TAKE mispricings ---
        for ask in sorted(depth.sell_orders.keys()):
            if ask > fair - self.TAKE_EDGE:
                break
            avail = -depth.sell_orders[ask]
            qty = min(avail, max_buy - used_buy, self.MAX_TAKE_PER_LEVEL)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                used_buy += qty
                pos += qty

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid < fair + self.TAKE_EDGE:
                break
            avail = depth.buy_orders[bid]
            qty = min(avail, max_sell - used_sell, self.MAX_TAKE_PER_LEVEL)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                used_sell += qty
                pos -= qty

        # --- PHASE 2: Inventory management ---
        # (Disabled by default: CLEAR_THRESH=1.0 means never clear)
        if pos > limit * self.CLEAR_THRESH and max_sell - used_sell > 0:
            q = min(5, max_sell - used_sell)
            orders.append(Order(product, int(math.ceil(fair)), -q))
            used_sell += q
            pos -= q
        elif pos < -limit * self.CLEAR_THRESH and max_buy - used_buy > 0:
            q = min(5, max_buy - used_buy)
            orders.append(Order(product, int(math.floor(fair)), q))
            used_buy += q
            pos += q

        # --- PHASE 3: Passive market making quotes ---
        inv_skew = int(pos / limit * self.INV_FACTOR)
        imb_skew = int(imb * self.IMB_FACTOR)

        # Default tight quotes at fair ± 1 with skews
        default_bid = int(math.floor(fair - 1)) - inv_skew + imb_skew
        default_ask = int(math.ceil(fair + 1)) - inv_skew + imb_skew

        # Join the book: quote = min(default, best_bid+offset) for bids
        # This picks the WORSE price for us (lower bid / higher ask), which
        # captures more spread per fill vs jumping ahead in queue
        quote_bid = min(default_bid, best_bid + self.BID_OFFSET)
        quote_ask = max(default_ask, best_ask - self.ASK_OFFSET)

        # Safety: never cross the spread
        if quote_bid >= best_ask:
            quote_bid = best_ask - 1
        if quote_ask <= best_bid:
            quote_ask = best_bid + 1

        # Safety: never cross fair value (would mean buying above fair or selling below)
        if quote_bid > int(math.floor(fair)):
            quote_bid = int(math.floor(fair)) - 1
        if quote_ask < int(math.ceil(fair)):
            quote_ask = int(math.ceil(fair)) + 1

        bq = max_buy - used_buy
        sq = max_sell - used_sell
        if bq > 0:
            orders.append(Order(product, quote_bid, bq))
        if sq > 0:
            orders.append(Order(product, quote_ask, -sq))

        return orders
