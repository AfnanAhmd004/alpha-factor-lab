"""Synthetic cross-sectional equity panel with known factor premia.

Planting known effects lets you check that the research pipeline recovers
them and does not invent effects that are not there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_panel(n_assets: int = 200, n_days: int = 1000, seed: int = 0,
                    mom_premium: float = 0.0004, rev_premium: float = 0.0006, lowvol_premium: float = 0.0002) -> pd.DataFrame:
    """Long-format panel with columns date, asset, close.

    Next-day returns load on three characteristics computed from *past* prices:
    medium-term momentum (+), 5-day reversal (-) and low volatility (+).
    """
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.01, 0.03, n_assets)
    beta = rng.uniform(0.6, 1.4, n_assets)
    rets = np.zeros((n_days, n_assets))
    for t in range(n_days):
        mkt = rng.normal(0.0002, 0.008)
        alpha = np.zeros(n_assets)
        if t > 130:
            mom = rets[t - 126 : t - 21].sum(0)
            rev = rets[t - 5 : t].sum(0)
            lv = -rets[t - 63 : t].std(0)
            alpha = mom_premium * _z(mom) - rev_premium * _z(rev) + lowvol_premium * _z(lv)
        rets[t] = beta * mkt + alpha + vol * rng.standard_normal(n_assets)
    close = 50 * np.exp(np.cumsum(rets, axis=0))
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    wide = pd.DataFrame(close, index=dates, columns=[f"A{i:03d}" for i in range(n_assets)])
    return wide.stack().rename("close").rename_axis(["date", "asset"]).reset_index()


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def to_wide(panel: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    return panel.pivot(index="date", columns="asset", values=col).sort_index()
