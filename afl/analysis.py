"""Factor evaluation: IC, IC decay, quantile returns, turnover and IC-weighted combination."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return from close t to close t+h, aligned to t (the day the signal is known)."""
    return prices.shift(-horizon) / prices - 1


def information_coefficient(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily Spearman rank correlation between scores and forward returns."""
    s, f = scores.rank(axis=1), fwd.rank(axis=1)
    valid = scores.notna() & fwd.notna()
    s, f = s.where(valid), f.where(valid)
    sc, fc = s.sub(s.mean(axis=1), axis=0), f.sub(f.mean(axis=1), axis=0)
    ic = (sc * fc).sum(axis=1) / np.sqrt((sc**2).sum(axis=1) * (fc**2).sum(axis=1))
    return ic[valid.sum(axis=1) > 10].dropna()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    return {"mean_ic": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std() * np.sqrt(252)) if ic.std() > 0 else 0.0,
            "t_stat": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() > 0 else 0.0,
            "pct_positive": float((ic > 0).mean())}


def ic_decay(scores: pd.DataFrame, prices: pd.DataFrame, horizons=(1, 5, 10, 21)) -> pd.Series:
    return pd.Series({h: information_coefficient(scores, forward_returns(prices, h)).mean() for h in horizons})


def quantile_returns(scores: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 5) -> pd.DataFrame:
    """Mean forward return of each score quintile, per day."""
    ranks = scores.rank(axis=1, pct=True)
    buckets = np.ceil(ranks * n_q).clip(1, n_q)
    out = {q: fwd.where(buckets == q).mean(axis=1) for q in range(1, n_q + 1)}
    return pd.DataFrame(out).dropna()


def long_short(scores: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 5) -> pd.Series:
    q = quantile_returns(scores, fwd, n_q)
    return q[n_q] - q[1]


def turnover(scores: pd.DataFrame, n_q: int = 5) -> float:
    """Average daily fraction of the top quintile that changes."""
    top = scores.rank(axis=1, pct=True) > 1 - 1 / n_q
    changes = (top != top.shift()).sum(axis=1) / (2 * top.sum(axis=1).replace(0, np.nan))
    return float(changes.iloc[1:].mean())


def combine_ic_weighted(factors: dict[str, pd.DataFrame], fwd: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Blend factors with weights equal to their trailing mean IC (lagged so weights are known in advance)."""
    weighted = None
    for name, sc in factors.items():
        w = information_coefficient(sc, fwd).reindex(sc.index).rolling(lookback, min_periods=60).mean().shift(2)
        contrib = sc.mul(w, axis=0)
        weighted = contrib if weighted is None else weighted.add(contrib, fill_value=0)
    return weighted
