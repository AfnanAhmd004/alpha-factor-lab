"""alpha-factor-lab: cross-sectional factor research toolkit."""
from .analysis import (combine_ic_weighted, forward_returns, ic_decay, ic_summary, information_coefficient,
                       long_short, quantile_returns, turnover)
from .factors import REGISTRY, compute_all, cross_sectional_zscore, factor
from .panel import synthetic_panel, to_wide

__all__ = ["REGISTRY", "combine_ic_weighted", "compute_all", "cross_sectional_zscore", "factor", "forward_returns",
           "ic_decay", "ic_summary", "information_coefficient", "long_short", "quantile_returns", "synthetic_panel",
           "to_wide", "turnover"]
