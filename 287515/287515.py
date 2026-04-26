"""
IMC Prosperity 4 — Round 2 Trader (v4)

CORE INSIGHT (from live server log analysis):
The L2 order book levels (volume >= 8) are the REAL market maker's quotes.
The L1 levels (smaller volume) are "skimmer" noise traders.

For IPR: wall_mid follows 12000 + 0.001*t exactly (residual std ~0.20 ticks on R2)
For ACO: wall_mid ≈ 10001 flat; no drift, no bias needed

V4 CHANGES (driven by deep statistical analysis of ACO order book):

1. IMB_FACTOR = 0: exhaustive sweep showed imbalance skew adds noise, not signal.
   ACO book is 82% of ticks perfectly balanced (L1 vols 10-15 on both sides, imb=0).
   The 87% direction accuracy from imbalance signal cannot be monetized — the L1
   spread is 16 ticks and the signal edge is only ~3.5 ticks, below the cost to take.
   Quote suppression was also tested and found to hurt net fills. IMB_FACTOR = 0 wins.

2. MAF dynamic limit detection (run-time, via traderData):
   LIMITS starts at 80. Once confirmed MAF is won, switches to 100 for both products.
   Detection: if any position ever exceeds 80, server limit must be 100 → confirmed.
   Probe: first tick IPR pos reaches 79, try limit=81 (2-unit overshoot). If server
   limit=80: all-or-nothing cancel fires (1 wasted tick). If server limit=100: pos
   reaches 81 next tick, confirmed, full 100-unit capacity for rest of session.

VALIDATED ON ROUND 2 DATA (match-trades worse, i.e. live-equivalent):
- 100k session: 28,285 total across 3 days (9,776 / 9,529 / 8,980)
"""

from datamodel import Order, OrderDepth, TradingState
import json
import math


# --- Market Access Fee (Round 2) -------------------------------------------
# Bumped from 2000 -> 3500 after submission 277550 lost the top-50% cutoff.
# Rationale: winning MAF adds ~1,851/session (IPR limit 80->100 = 7,403->9,254).
# Break-even is ~5,500 over 3 days; 3,500 is ~65% of that — positive EV if win,
# bounded loss if lose. Known-losing ceiling: 2,000. Rational upper bound: 5,500.
MAF = 3500
# ---------------------------------------------------------------------------


class Trader:
    LIMITS = {"INTARIAN_PEPPER_ROOT": 80, "ASH_COATED_OSMIUM": 80}

    # Market Access Fee (mirror of module-level MAF — whichever the grader reads)
    MAF = MAF

    # Parameters — tuned under --match-trades worse (live-equivalent matching)
    IPR_BIAS = 7.5
    ACO_BIAS = 0
    BID_OFFSET = 1        # quote 1 tick inside best_bid (top of queue)
    ASK_OFFSET = 1        # quote 1 tick inside best_ask (top of queue)
    TAKE_EDGE = 1.0
    IMB_FACTOR = 0
    INV_FACTOR = 4
    CLEAR_THRESH = 1.0
    WALL_MIN_VOL = 8      # Round 2 walls are thinner than R1 walls
    MAX_TAKE_PER_LEVEL = 25

    def run(self, state: TradingState):
        data = json.loads(state.traderData) if state.traderData else {}

        # MAF detection: position > 80 proves server limit is 100
        for product in self.LIMITS:
            if abs(state.position.get(product, 0)) > 80:
                data["maf_confirmed"] = True
                break

        maf_confirmed = data.get("maf_confirmed", False)

        all_orders = {}
        for product in self.LIMITS:
            if product not in state.order_depths:
                continue
            depth = state.order_depths[product]
            if not depth.buy_orders or not depth.sell_orders:
                continue
            pos = state.position.get(product, 0)

            # Dynamic limit: 100 if MAF confirmed, else 80 with one probe.
            # Probe fires once when IPR first hits pos=79: sets limit=81 so total
            # submitted buys = 2. If server limit=80 → all cancelled (1 wasted tick).
            # If server limit=100 → pos reaches 81 next tick → maf_confirmed fires.
            if maf_confirmed:
                limit = 100
            elif (not data.get("maf_probed")
                  and product == "INTARIAN_PEPPER_ROOT"
                  and pos >= 79):
                limit = 81
                data["maf_probed"] = True
            else:
                limit = 80

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