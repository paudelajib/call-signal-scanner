"""S&P 500 universe with GICS sectors."""
from __future__ import annotations

import io
import time
from typing import Dict

import pandas as pd
import requests

from .config import Paths


def yahoo_symbol(sym: str) -> str:
    """Class shares use '-' on Yahoo (BRK.B -> BRK-B)."""
    return sym.strip().upper().replace(".", "-")


def load_universe(cfg: Dict, paths: Paths, force: bool = False) -> pd.DataFrame:
    """Columns: ticker, name, sector, sub_industry, date_added.

    Downloaded from the public datasets/s-and-p-500-companies list and cached; refreshed every
    `universe.refresh_days` days so additions/removals are picked up.
    """
    p = paths.universe
    max_age = cfg["universe"].get("refresh_days", 7) * 86400
    if p.exists() and not force and (time.time() - p.stat().st_mtime) < max_age:
        return pd.read_csv(p)

    try:
        r = requests.get(cfg["universe"]["source_url"], timeout=30)
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        if p.exists():
            print(f"[universe] download failed ({e}); using cached list")
            return pd.read_csv(p)
        raise

    df = pd.DataFrame({
        "ticker": raw["Symbol"].map(yahoo_symbol),
        "name": raw["Security"],
        "sector": raw["GICS Sector"],
        "sub_industry": raw["GICS Sub-Industry"],
        "date_added": raw.get("Date added"),
    }).drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    df.to_csv(p, index=False)
    return df
