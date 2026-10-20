"""Command line:  python -m src.cli refresh | backtest | serve | rescore"""
from __future__ import annotations

import argparse
import json
import sys


def _progress(stage: str, done: int, total: int):
    msg = f"{stage}: {done}/{total}" if total else stage
    sys.stdout.write("\r" + msg.ljust(70))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(prog="scanner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh", help="download new data and re-score")
    r.add_argument("--skip-news", action="store_true")
    r.add_argument("--force-fundamentals", action="store_true")
    sub.add_parser("backtest", help="walk-forward backtest on stored history")
    sub.add_parser("rescore", help="re-score from stored features with current weights.yaml")
    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()

    if a.cmd == "refresh":
        from .engine import run_refresh
        log = run_refresh(_progress, fundamentals_force=a.force_fundamentals, skip_news=a.skip_news)
        print("\n" + json.dumps(log, indent=2, default=str))
    elif a.cmd == "backtest":
        from .backtest import run_backtest
        res = run_backtest(_progress)
        print()
        for h, v in res["horizons"].items():
            ic = v["ic"]
            print(f"{v['label']:>9}: mean IC {ic['mean']:+.4f}  t {ic['t'] or 0:+.2f}  "
                  f"Q5-Q1 {v['top_minus_bottom'] * 100:+.2f}%/period  hit {v['top_hit_rate']:.0%}")
    elif a.cmd == "rescore":
        from .engine import rescore
        print(rescore())
    elif a.cmd == "serve":
        sys.argv = ["app.py", "--port", str(a.port)]
        import app  # noqa: F401
        app.main()


if __name__ == "__main__":
    main()
