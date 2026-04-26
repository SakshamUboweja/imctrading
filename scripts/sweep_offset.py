"""Sweep BID_OFFSET / ASK_OFFSET: do we benefit from jumping the queue to
quote inside best_bid (closer to fair)? The current default joins best_bid
(offset=0) which means we sit with 20+ units ahead of us."""
import sys
sys.path.insert(0, "/Users/saksham/Desktop/imctrading/scripts")
from sweep import make_variant, run_backtest

def run(label, edits, data="./data_100k/"):
    path = make_variant(label, edits)
    pnl = run_backtest(path, data, match="worse")
    days = {k: v for k, v in pnl.items() if isinstance(k, int)}
    return pnl, days

# All tests use WALL_MIN_VOL=8 (proven win)
base = {"WALL_MIN_VOL": "8"}

print("=== BID_OFFSET sweep (higher = more aggressive bid) ===")
for v in [0, 1, 2, 3, 5, 10, 20]:
    pnl, days = run(f"bo_{v}", {**base, "BID_OFFSET": str(v)})
    print(f"  BID_OFFSET={v:>2}: total={pnl.get('total'):>6} days={days}")

print("\n=== ASK_OFFSET sweep ===")
for v in [0, 1, 2, 3, 5, 10, 20]:
    pnl, days = run(f"ao_{v}", {**base, "ASK_OFFSET": str(v)})
    print(f"  ASK_OFFSET={v:>2}: total={pnl.get('total'):>6} days={days}")

print("\n=== Both offsets ===")
for b in [0, 3, 10]:
    for a in [0, 3, 10]:
        pnl, days = run(f"boao_{b}_{a}", {**base, "BID_OFFSET": str(b), "ASK_OFFSET": str(a)})
        print(f"  BID={b:>2} ASK={a:>2}: total={pnl.get('total'):>6} days={days}")

print("\n=== 1M validation ===")
for label, edits in [("base_w8", base),
                     ("b3_w8", {**base, "BID_OFFSET": "3"}),
                     ("b10_w8", {**base, "BID_OFFSET": "10"})]:
    path = make_variant(f"v1m_{label}", edits)
    pnl = run_backtest(path, "./data/", match="worse")
    days = {k: v for k, v in pnl.items() if isinstance(k, int)}
    print(f"  {label}: total={pnl.get('total'):>7} days={days}")
