"""Probe wall threshold + combos, all under --match-trades worse."""
import sys
sys.path.insert(0, "/Users/saksham/Desktop/imctrading/scripts")
from sweep import make_variant, run_backtest

def run(label, edits, data="./data_100k/"):
    path = make_variant(label, edits)
    pnl = run_backtest(path, data, match="worse")
    days = {k: v for k, v in pnl.items() if isinstance(k, int)}
    return pnl, days

print("=== Lower WALL_MIN_VOL (100k worse) ===")
for v in [3, 4, 5, 6, 7, 8]:
    pnl, days = run(f"wall_{v}", {"WALL_MIN_VOL": str(v)})
    print(f"  WALL={v:>2}: total={pnl.get('total'):>6} days={days}")

print("\n=== WALL=8 × IPR_BIAS combos ===")
for ipr in [6, 7.5, 9]:
    for wall in [6, 8, 10]:
        pnl, days = run(f"c_ipr{ipr}_w{wall}", {"IPR_BIAS": str(ipr), "WALL_MIN_VOL": str(wall)})
        print(f"  IPR={ipr:>4} WALL={wall:>2}: total={pnl.get('total'):>6} days={days}")

print("\n=== WALL=8 × TAKE_EDGE combos ===")
for te in [0.5, 1.0, 0.25]:
    pnl, days = run(f"c_te{te}_w8", {"TAKE_EDGE": str(te), "WALL_MIN_VOL": "8"})
    print(f"  TAKE={te:>4} WALL=8: total={pnl.get('total'):>6} days={days}")

print("\n=== INV_FACTOR / IMB_FACTOR sweep (WALL=8) ===")
for inv in [2, 3, 4, 6, 8]:
    pnl, days = run(f"inv{inv}_w8", {"INV_FACTOR": str(inv), "WALL_MIN_VOL": "8"})
    print(f"  INV={inv:>2}: total={pnl.get('total'):>6} days={days}")
for imb in [0, 2, 3, 5]:
    pnl, days = run(f"imb{imb}_w8", {"IMB_FACTOR": str(imb), "WALL_MIN_VOL": "8"})
    print(f"  IMB={imb:>2}: total={pnl.get('total'):>6} days={days}")

print("\n=== 1M validation of top candidates ===")
for label, edits in [("default", {}),
                     ("w8", {"WALL_MIN_VOL": "8"}),
                     ("w6", {"WALL_MIN_VOL": "6"}),
                     ("w8_te05", {"WALL_MIN_VOL": "8", "TAKE_EDGE": "0.5"})]:
    path = make_variant(f"v1m_{label}", edits)
    pnl = run_backtest(path, "./data/", match="worse")
    days = {k: v for k, v in pnl.items() if isinstance(k, int)}
    print(f"  {label}: total={pnl.get('total'):>7} days={days}")
