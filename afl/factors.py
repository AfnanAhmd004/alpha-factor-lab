"""Factor library. Each factor maps a wide price frame to a wide score frame using only past data."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

FactorFn = Callable[[pd.DataFrame], pd.DataFrame]
REGISTRY: dict[str, FactorFn] = {}


def factor(name: str):
    def deco(fn: FactorFn) -> FactorFn:
        REGISTRY[name] = fn
        return fn
    return deco


@factor("momentum_6_1")
def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """Return from t-126 to t-21: skips the most recent month to avoid short-term reversal."""
    return prices.shift(21) / prices.shift(126) - 1


@factor("reversal_5d")
def reversal(prices: pd.DataFrame) -> pd.DataFrame:
    return -(prices / prices.shift(5) - 1)


@factor("low_vol_63d")
def low_vol(prices: pd.DataFrame) -> pd.DataFrame:
    return -np.log(prices).diff().rolling(63).std()


@factor("trend_quality")
def trend_quality(prices: pd.DataFrame) -> pd.DataFrame:
    """60-day return divided by its volatility: a t-stat-like trend score."""
    r = np.log(prices).diff()
    return r.rolling(60).sum() / (r.rolling(60).std() * np.sqrt(60))


def cross_sectional_zscore(scores: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    z = scores.sub(scores.mean(axis=1), axis=0).div(scores.std(axis=1), axis=0)
    return z.clip(-clip, clip)


def compute_all(prices: pd.DataFrame, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    names = names or list(REGISTRY)
    return {n: cross_sectional_zscore(REGISTRY[n](prices)) for n in names}
