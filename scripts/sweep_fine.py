"""Fine-tune around BID_OFFSET=3, ASK_OFFSET=3 with WALL_MIN_VOL=8."""
import sys
sys.path.insert(0, "/Users/saksham/Desktop/imctrading/scripts")
from sweep import make_variant, run_backtest

def run(label, edits, data="./data_100k/"):
    path = make_variant(label, edits)
    pnl = run_backtest(path, data, match="worse")
    days = {k: v for k, v in pnl.items() if isinstance(k, int)}
    return pnl, days

base = {"WALL_MIN_VOL": "8"}

print("=== Fine around (BID=3, ASK=3) ===")
for b in [1, 2, 3, 4, 5]:
    for a in [1, 2, 3, 4, 5]:
        pnl, days = run(f"f_{b}_{a}", {**base, "BID_OFFSET": str(b), "ASK_OFFSET": str(a)})
        print(f"  BID={b} ASK={a}: total={pnl.get('total'):>6} days={days}")

print("\n=== Combined with IPR_BIAS variation ===")
for ipr in [6, 7.5, 9]:
    for b, a in [(3, 3), (2, 3), (3, 2), (2, 2)]:
        pnl, days = run(f"c_{ipr}_{b}_{a}", {**base, "IPR_BIAS": str(ipr), "BID_OFFSET": str(b), "ASK_OFFSET": str(a)})
        print(f"  IPR={ipr} BID={b} ASK={a}: total={pnl.get('total'):>6} days={days}")

print("\n=== Top candidates — 1M validation ===")
candidates = [
    ("default", {}),
    ("w8_b3_a3", {"WALL_MIN_VOL": "8", "BID_OFFSET": "3", "ASK_OFFSET": "3"}),
    ("w8_b2_a3", {"WALL_MIN_VOL": "8", "BID_OFFSET": "2", "ASK_OFFSET": "3"}),
    ("w8_b3_a2", {"WALL_MIN_VOL": "8", "BID_OFFSET": "3", "ASK_OFFSET": "2"}),
]
for label, edits in candidates:
    path = make_variant(f"final1m_{label}", edits)
    pnl = run_backtest(path, "./data/", match="worse")
    days = {k: v for k, v in pnl.items() if isinstance(k, int)}
    print(f"  {label}: total={pnl.get('total'):>7} days={days}")
