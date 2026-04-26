"""Realistic sweep under --match-trades worse (matches live server)."""
import sys
sys.path.insert(0, "/Users/saksham/Desktop/imctrading/scripts")
from sweep import make_variant, run_backtest

# Since live = day 1 likely, but the live session draws from Round 2 data
# stochastically, test on all 3 days to avoid overfitting to a specific day.

def run_100k(label, edits):
    path = make_variant(label, edits)
    pnl = run_backtest(path, "./data_100k/", match="worse")
    return pnl

print("=== Baseline ===")
base = run_100k("base", {})
print(f"  baseline: {base}")

# Sweep IPR_BIAS
print("\n=== IPR_BIAS sweep (100k worse, ACO=0, WALL=15) ===")
for v in [3, 4, 5, 6, 7.5, 9, 11, 14]:
    pnl = run_100k(f"ipr{v}", {"IPR_BIAS": str(v)})
    days = {k: v2 for k, v2 in pnl.items() if isinstance(k, int)}
    print(f"  IPR_BIAS={v:>4}: total={pnl.get('total'):>6} days={days}")

# Sweep ACO_BIAS
print("\n=== ACO_BIAS sweep (100k worse, IPR=7.5, WALL=15) ===")
for v in [-1, -0.5, 0, 0.5, 1, 2]:
    pnl = run_100k(f"aco{v}", {"ACO_BIAS": str(v)})
    days = {k: v2 for k, v2 in pnl.items() if isinstance(k, int)}
    print(f"  ACO_BIAS={v:>4}: total={pnl.get('total'):>6} days={days}")

# Sweep WALL_MIN_VOL
print("\n=== WALL_MIN_VOL sweep (100k worse, IPR=7.5, ACO=0) ===")
for v in [8, 10, 12, 15, 18, 20, 25, 30]:
    pnl = run_100k(f"wall{v}", {"WALL_MIN_VOL": str(v)})
    days = {k: v2 for k, v2 in pnl.items() if isinstance(k, int)}
    print(f"  WALL_MIN_VOL={v:>3}: total={pnl.get('total'):>6} days={days}")

# Sweep TAKE_EDGE
print("\n=== TAKE_EDGE sweep (100k worse) ===")
for v in [0.5, 1.0, 1.5, 2.0, 3.0]:
    pnl = run_100k(f"take{v}", {"TAKE_EDGE": str(v)})
    days = {k: v2 for k, v2 in pnl.items() if isinstance(k, int)}
    print(f"  TAKE_EDGE={v:>4}: total={pnl.get('total'):>6} days={days}")

# Sweep MAX_TAKE_PER_LEVEL (how fast we load up drift)
print("\n=== MAX_TAKE_PER_LEVEL sweep (100k worse) ===")
for v in [10, 15, 25, 40, 80]:
    pnl = run_100k(f"mt{v}", {"MAX_TAKE_PER_LEVEL": str(v)})
    days = {k: v2 for k, v2 in pnl.items() if isinstance(k, int)}
    print(f"  MAX_TAKE_PER_LEVEL={v:>3}: total={pnl.get('total'):>6} days={days}")
