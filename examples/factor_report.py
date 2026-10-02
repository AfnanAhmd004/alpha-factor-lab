"""Evaluate every registered factor on a synthetic 200-stock panel."""
import numpy as np

from afl import (combine_ic_weighted, compute_all, forward_returns, ic_decay, ic_summary, information_coefficient,
                 long_short, synthetic_panel, to_wide, turnover)

prices = to_wide(synthetic_panel(n_assets=200, n_days=1000, seed=0))
fwd = forward_returns(prices, 1)
factors = compute_all(prices)
factors["combined"] = combine_ic_weighted(factors, fwd)

print(f"{'factor':<16}{'meanIC':>8}{'t':>7}{'IC>0':>7}{'LS Sharpe':>11}{'turnover':>10}")
for name, sc in factors.items():
    s = ic_summary(information_coefficient(sc, fwd))
    ls = long_short(sc, fwd)
    sharpe = ls.mean() / ls.std() * np.sqrt(252)
    print(f"{name:<16}{s['mean_ic']:>8.3f}{s['t_stat']:>7.1f}{s['pct_positive']:>7.2f}{sharpe:>11.2f}{turnover(sc):>10.2f}")

print("\nIC decay for momentum_6_1:")
print(ic_decay(factors["momentum_6_1"], prices).round(4).to_string())
