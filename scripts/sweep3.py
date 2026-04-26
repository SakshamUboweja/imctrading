"""Reconcile 1M vs 100k: search for params that win BOTH."""
import sys
sys.path.insert(0, "/Users/saksham/Desktop/imctrading/scripts")
from sweep import make_variant, run_backtest

combos = [
    # Keep IPR at the live-winning 7.5 but drop ACO bias (1M showed ACO=0 adds 10k)
    (7.5, 0, 15),
    (7.5, 0.5, 15),
    (7.5, 0.25, 15),
    # Mid-IPR, low ACO
    (6, 0, 15),
    (6, 0.5, 15),
    # WALL variation with defaults
    (7.5, 0.75, 20),
    (7.5, 0.75, 25),
    (7.5, 0, 25),
]

for ipr, aco, wall in combos:
    path = make_variant(f"recon_{ipr}_{aco}_{wall}",
                        {"IPR_BIAS": str(ipr), "ACO_BIAS": str(aco), "WALL_MIN_VOL": str(wall)})
    pnl_1m = run_backtest(path, "./data/")
    pnl_100k = run_backtest(path, "./data_100k/")
    print(f"IPR={ipr:>4} ACO={aco:>4} WALL={wall:>2} | 1M total={pnl_1m.get('total'):>7} | 100k total={pnl_100k.get('total'):>6} | 100k days={ {k:v for k,v in pnl_100k.items() if isinstance(k,int)} }")
