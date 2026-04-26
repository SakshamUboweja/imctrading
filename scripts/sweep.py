"""Parameter sweep harness for trader.py on Round 2 data.

Clones trader.py, monkey-patches class constants, runs the full 1M-tick
backtest, and reports per-day + total PnL.
"""

import subprocess
import sys
import re
import os
import shutil
from pathlib import Path

ROOT = Path("/Users/saksham/Desktop/imctrading")
BASE = ROOT / "trader.py"
SWEEP_DIR = ROOT / "_sweep"
SWEEP_DIR.mkdir(exist_ok=True)

def make_variant(name, edits: dict[str, str]) -> Path:
    src = BASE.read_text()
    for key, val in edits.items():
        pattern = rf"^(    {re.escape(key)}\s*=\s*)[^\n]+"
        if not re.search(pattern, src, flags=re.MULTILINE):
            raise ValueError(f"Could not patch {key}")
        src = re.sub(pattern, rf"\g<1>{val}", src, count=1, flags=re.MULTILINE)
    safe_name = name.replace(".", "p").replace("-", "m")
    p = SWEEP_DIR / f"trader_{safe_name}.py"
    p.write_text(src)
    # Copy datamodel.py shim so the variant can import it
    dm = SWEEP_DIR / "datamodel.py"
    if not dm.exists():
        shutil.copy(ROOT / "datamodel.py", dm)
    return p

def run_backtest(path: Path, data_dir="./data/", match="worse") -> dict:
    result = subprocess.run(
        ["prosperity4bt", str(path), "2", "--data", data_dir, "--merge-pnl",
         "--no-out", "--match-trades", match],
        capture_output=True, text=True, cwd=ROOT,
    )
    out = result.stdout + result.stderr
    pnl = {}
    for match in re.finditer(r"Round 2 day (-?\d+): ([\d,\-]+)", out):
        pnl[int(match.group(1))] = int(match.group(2).replace(",", ""))
    total_match = re.search(r"Profit summary:.*?Total profit: ([\d,\-]+)", out, re.DOTALL)
    if total_match:
        pnl["total"] = int(total_match.group(1).replace(",", ""))
    if "Orders exceeded limit" in out:
        pnl["limit_warn"] = True
    return pnl

def sweep(param_name, values, fixed=None, data_dir="./data/"):
    fixed = fixed or {}
    print(f"\n=== Sweep {param_name} on {data_dir} (fixed: {fixed}) ===")
    results = []
    for v in values:
        edits = {param_name: str(v), **fixed}
        path = make_variant(f"{param_name}_{v}", edits)
        pnl = run_backtest(path, data_dir)
        results.append((v, pnl))
        print(f"  {param_name}={v}: days={ {k:v for k,v in pnl.items() if isinstance(k,int)} } total={pnl.get('total')}")
    return results

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    data = sys.argv[2] if len(sys.argv) > 2 else "./data/"
    if which in ("all", "ipr"):
        sweep("IPR_BIAS", [5, 6, 7.5, 9, 11], data_dir=data)
    if which in ("all", "aco"):
        sweep("ACO_BIAS", [0, 0.5, 0.75, 1.0], data_dir=data)
    if which in ("all", "wall"):
        sweep("WALL_MIN_VOL", [10, 15, 20, 25], data_dir=data)
