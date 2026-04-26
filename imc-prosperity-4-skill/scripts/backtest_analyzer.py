"""
IMC Prosperity 4 — Backtest Analysis Utility
Parses backtester output logs and generates performance metrics.

Usage:
    python backtest_analyzer.py backtests/<timestamp>.log
"""

import sys
import json
import math
import csv
from io import StringIO


def parse_log_file(filepath: str) -> dict:
    """Parse a prosperity4bt output log into structured data."""
    with open(filepath, 'r') as f:
        content = f.read()

    sections = {}
    current_section = None
    current_lines = []

    for line in content.split('\n'):
        if line.startswith('Sandbox logs:'):
            if current_section:
                sections[current_section] = '\n'.join(current_lines)
            current_section = 'sandbox'
            current_lines = []
        elif line.startswith('Activities log:'):
            if current_section:
                sections[current_section] = '\n'.join(current_lines)
            current_section = 'activities'
            current_lines = []
        elif line.startswith('Trade History:'):
            if current_section:
                sections[current_section] = '\n'.join(current_lines)
            current_section = 'trades'
            current_lines = []
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = '\n'.join(current_lines)

    return sections


def parse_activities(activities_text: str) -> list:
    """Parse activities CSV into list of dicts."""
    rows = []
    reader = csv.reader(StringIO(activities_text), delimiter=';')

    for row in reader:
        if len(row) < 17:
            continue
        try:
            rows.append({
                'day': int(row[0]),
                'timestamp': int(row[1]),
                'product': row[2],
                'bid1': int(row[3]) if row[3] else 0,
                'bid_vol1': int(row[4]) if row[4] else 0,
                'bid2': int(row[5]) if row[5] else 0,
                'bid_vol2': int(row[6]) if row[6] else 0,
                'bid3': int(row[7]) if row[7] else 0,
                'bid_vol3': int(row[8]) if row[8] else 0,
                'ask1': int(row[9]) if row[9] else 0,
                'ask_vol1': int(row[10]) if row[10] else 0,
                'ask2': int(row[11]) if row[11] else 0,
                'ask_vol2': int(row[12]) if row[12] else 0,
                'ask3': int(row[13]) if row[13] else 0,
                'ask_vol3': int(row[14]) if row[14] else 0,
                'mid_price': float(row[15]) if row[15] else 0,
                'pnl': float(row[16]) if row[16] else 0,
            })
        except (ValueError, IndexError):
            continue

    return rows


def parse_trades(trades_text: str) -> list:
    """Parse trade history JSON."""
    try:
        trades = json.loads(trades_text.strip())
        return trades if isinstance(trades, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def analyze_performance(activities: list, trades: list) -> dict:
    """Generate comprehensive performance metrics."""
    if not activities:
        return {"error": "No activity data"}

    # Group by product
    products = {}
    for row in activities:
        prod = row['product']
        if prod not in products:
            products[prod] = []
        products[prod].append(row)

    results = {}

    for product, rows in products.items():
        pnls = [r['pnl'] for r in rows]
        prices = [r['mid_price'] for r in rows if r['mid_price'] > 0]

        # PnL metrics
        final_pnl = pnls[-1] if pnls else 0
        pnl_changes = [pnls[i] - pnls[i-1] for i in range(1, len(pnls))]

        # Calculate metrics
        if pnl_changes:
            mean_change = sum(pnl_changes) / len(pnl_changes)
            var_change = sum((c - mean_change)**2 for c in pnl_changes) / len(pnl_changes)
            std_change = math.sqrt(var_change) if var_change > 0 else 1e-10
            sharpe = mean_change / std_change * math.sqrt(1000)
        else:
            sharpe = 0

        # Max drawdown
        peak = pnls[0] if pnls else 0
        max_dd = 0
        for p in pnls:
            peak = max(peak, p)
            max_dd = max(max_dd, peak - p)

        # Price statistics
        if len(prices) > 1:
            price_mean = sum(prices) / len(prices)
            price_std = math.sqrt(sum((p - price_mean)**2 for p in prices) / len(prices))
        else:
            price_mean = prices[0] if prices else 0
            price_std = 0

        results[product] = {
            'final_pnl': final_pnl,
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_dd, 2),
            'num_timestamps': len(rows),
            'price_mean': round(price_mean, 2),
            'price_std': round(price_std, 2),
            'pnl_per_timestamp': round(final_pnl / len(rows), 2) if rows else 0,
        }

    # Aggregate
    total_pnl = sum(r['final_pnl'] for r in results.values())
    results['TOTAL'] = {
        'final_pnl': total_pnl,
        'products': list(results.keys()),
    }

    return results


def print_report(results: dict):
    """Print formatted performance report."""
    print("\n" + "="*60)
    print("IMC PROSPERITY 4 — BACKTEST PERFORMANCE REPORT")
    print("="*60)

    for product, metrics in results.items():
        if product == 'TOTAL':
            continue

        print(f"\n--- {product} ---")
        for key, value in metrics.items():
            print(f"  {key:.<30} {value}")

    if 'TOTAL' in results:
        print(f"\n{'='*60}")
        print(f"  TOTAL PnL: {results['TOTAL']['final_pnl']:,.2f} SeaShells")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python backtest_analyzer.py <logfile>")
        sys.exit(1)

    sections = parse_log_file(sys.argv[1])
    activities = parse_activities(sections.get('activities', ''))
    trades = parse_trades(sections.get('trades', '[]'))

    results = analyze_performance(activities, trades)
    print_report(results)
