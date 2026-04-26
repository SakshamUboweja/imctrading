"""Follow-up sweep: IPR extension, combined best, and 100k verification."""
import sys
sys.path.insert(0, "/Users/saksham/Desktop/imctrading/scripts")
from sweep import sweep, make_variant, run_backtest

# 1. Extend IPR_BIAS to lower values
sweep("IPR_BIAS", [0, 1, 2, 3, 4, 5], data_dir="./data/")

# 2. Combine best ACO_BIAS=0 with varying IPR_BIAS
print("\n=== Combined: ACO_BIAS=0, varying IPR_BIAS ===")
for ipr in [0, 2, 4, 5, 7.5]:
    path = make_variant(f"combo_ipr{ipr}", {"IPR_BIAS": str(ipr), "ACO_BIAS": "0"})
    pnl = run_backtest(path, "./data/")
    print(f"  IPR_BIAS={ipr}, ACO_BIAS=0: {pnl}")

# 3. Best-of-best on 100k
print("\n=== 100k-tick verification of top candidates ===")
for (ipr, aco, wall) in [(7.5, 0.75, 15), (0, 0, 15), (5, 0, 15), (5, 0, 25), (0, 0, 25)]:
    path = make_variant(f"v100k_{ipr}_{aco}_{wall}",
                        {"IPR_BIAS": str(ipr), "ACO_BIAS": str(aco), "WALL_MIN_VOL": str(wall)})
    pnl = run_backtest(path, "./data_100k/")
    print(f"  IPR={ipr}, ACO={aco}, WALL={wall}: {pnl}")
