"""
IMC Prosperity 4 — Round 2 Trader (v8)

v8 = v3 (proven wall-detection + queue-jump + aggressive IPR take)
   + def bid(self) method (the critical MAF fix; prior submissions bid 0)

WHY v3 LOGIC, NOT v7's EMA-ANCHOR ACO (validated 2026-04-19 on R2 CSVs):
  Direct statistical analysis of /data/round2/prices_round_2_day_{-1,0,1}.csv:

  IPR (27,700 ticks):
    wall_mid vs linear fit: slope = exactly 0.001000/tick, resid std 0.80-0.92
    Simple mid: resid std ~400 (zero-mid artifacts when one side of book empty)
    → wall detection dominates for IPR by ~500x.
    With IPR_BIAS=7.5, take triggers on 42% of ticks — fast 80-unit load.

  ACO (27,638 ticks with walls present, 92% coverage):
    wall_mid std: 3.70 (day -1), 5.10 (day 0), 4.34 (day 1)
    wall_mid MEAN varies by day: 10000.85 / 10001.63 / 10000.19
      → EMA anchored at fixed 10001 is biased 0.8 ticks on 2/3 days
      → takes ~35 ticks to converge, so first 100 ticks quote at wrong level
    ACO drift: slope ~1e-5/tick (effectively zero; truly stationary)
    Take opportunities @ edge>1: 1.2% of ticks (ask side), 1.0% (bid side)
      mean edge when takeable: 2.60 ask / 1.74 bid, max 4/3
      → ACO taking IS profitable despite low frequency. v7's passive-only
         cost ~1,500/session of real edge capture.

  Conclusion: wall-detection dominates EMA-anchor for both products.
  The "smoother curve" in 310976 traded EV for chart prettiness — the user's
  instinct ("maximize EV > maximize smoothness") was data-correct.

MAF BID (method-based, per PDF FAQ):
  Prior submissions declared MAF as a class constant; the grader reads only
  def bid(self): return X. "If no bid(), we consider it 0." We have been
  bidding 0 the whole time. Bid 3000 here:
    - Incremental MAF win: 80% flow → 100% flow ≈ +25% counterparty volume
      ≈ +2,200/simulation in PnL. Break-even single-sim: ~2,200; 3-day ~6,600.
    - Competitors who don't read the PDF bid 0. Many will bid 1,500-2,500.
    - 3000 clears expected median comfortably, stays positive-EV under the
      conservative single-simulation interpretation.

BACKTEST (--match-trades worse, live-equivalent):
  - 100k: 28,285 total (9,776 / 9,529 / 8,980) — all 3 days clear 8.5k target
  - 1M:   295,325 total (98k+ per day)

PARAMETERS (sweep-validated under 'worse' match-mode):
  IPR_BIAS=7.5, WALL_MIN_VOL=8, BID/ASK_OFFSET=1, TAKE_EDGE=1.0
  CLEAR_THRESH=1.0 (disabled - drift strategy doesn't want clearing)

ANTI-OVERFIT GUARDRAILS:
  - Every parameter has a structural justification from Phase-2 stats
  - IPR wall_mid slope 0.001 and ACO wall_mid mean 10001 are derived from
    3-day data; not fit to a single day
  - No per-day tuning; identical behavior across all 3 days
  - All 3 days are positive with comfortable margin (>8.5k floor)
"""

from datamodel import Order, OrderDepth, TradingState
import json
import math


class Trader:
    # Position limits stay 80 — per PDF, MAF adds +25% quote VOLUME, not a
    # higher position cap. Any code that raises this to 100 would trigger
    # all-or-nothing cancellation if MAF is not won.
    LIMITS = {"INTARIAN_PEPPER_ROOT": 80, "ASH_COATED_OSMIUM": 80}

    # Parameters — tuned under --match-trades worse (live-equivalent matching)
    IPR_BIAS = 7.5        # wall_mid + 7.5 → take threshold hits L1_ask
    ACO_BIAS = 0          # ACO is stationary around wall_mid; no bias
    BID_OFFSET = 1        # quote 1 tick inside best_bid (top of queue)
    ASK_OFFSET = 1        # quote 1 tick inside best_ask
    TAKE_EDGE = 1.0       # take if edge >= 1 tick vs (biased) fair
    IMB_FACTOR = 3        # L1 imbalance skew on quotes
    INV_FACTOR = 4        # inventory skew on quotes
    CLEAR_THRESH = 1.0    # disabled (drift doesn't want clearing)
    WALL_MIN_VOL = 8      # R2 walls are thinner than R1 (10-15 units)
    MAX_TAKE_PER_LEVEL = 25

    def bid(self):
        """
        Market Access Fee bid. Read by grader ONLY from this method —
        class-level constants (MAF = X) are IGNORED. Per PDF FAQ: missing
        bid() == 0, negative bid == 0.

        Return value is XIRECs deducted from Round 2 final PnL IF our bid
        is in the top 50% (above median) of all submitted bids. Losers
        pay nothing. One-time fee, applied only in the final simulation.

        Sizing rationale:
          Tested PnL ~8,800 with 80% of quotes visible (testing mode).
          Winning MAF → full 100% flow → +25% more counterparty volume
          → ~+2,200 per simulation incremental.
          Break-even ceiling single-sim: 2,200; 3-day sim: 6,600.
          3000 sits safely above expected median while staying positive-EV
          even under the conservative single-sim interpretation.
        """
        return 3000

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
        # Data-validated: wall_mid residual std = 0.80 (IPR), 3.7-5.1 (ACO).
        # Simple mid has resid std ~400 due to empty-side zero artifacts.
        wb = self.find_wall_bid(depth)
        wa = self.find_wall_ask(depth)
        if wb and wa:
            fair = (wb + wa) / 2.0
        else:
            # Fallback when walls not present (~8% of ACO ticks)
            fair = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

        # Product-specific bias
        if product == "INTARIAN_PEPPER_ROOT":
            fair += self.IPR_BIAS
        else:
            fair += self.ACO_BIAS

        imb = self.get_imbalance(depth)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        # Position budget (all-or-nothing limit enforcement)
        max_buy = limit - pos
        max_sell = limit + pos
        if max_buy <= 0 and max_sell <= 0:
            return orders
        used_buy = 0
        used_sell = 0

        # --- PHASE 1: TAKE mispricings ---
        # For IPR: fires ~42% of ticks (drift captures +0.001/tick → 80 long)
        # For ACO: fires ~1-2% of ticks (real but rare edge, mean 2.60 ticks)
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
        # Disabled by default (CLEAR_THRESH=1.0). The IPR drift strategy
        # WANTS to hold +80 for the full session. Clearing destroys alpha.
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

        # Queue-jump: quote 1 tick inside best_bid/best_ask (top of book)
        # vs sitting behind a 20-unit skimmer queue at best_bid exactly.
        quote_bid = min(default_bid, best_bid + self.BID_OFFSET)
        quote_ask = max(default_ask, best_ask - self.ASK_OFFSET)

        # Safety: never cross the spread
        if quote_bid >= best_ask:
            quote_bid = best_ask - 1
        if quote_ask <= best_bid:
            quote_ask = best_bid + 1

        # Safety: never cross fair value
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
