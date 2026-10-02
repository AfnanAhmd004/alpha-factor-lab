import numpy as np
import pandas as pd
import pytest

from afl import compute_all, forward_returns, ic_summary, information_coefficient, long_short, synthetic_panel, to_wide


@pytest.fixture(scope="module")
def prices():
    return to_wide(synthetic_panel(n_assets=120, n_days=700, seed=1))


def test_planted_reversal_is_recovered(prices):
    fwd = forward_returns(prices)
    f = compute_all(prices, ["reversal_5d"])["reversal_5d"]
    s = ic_summary(information_coefficient(f, fwd))
    assert s["mean_ic"] > 0.01 and s["t_stat"] > 3


def test_random_factor_has_no_edge(prices):
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.standard_normal(prices.shape), index=prices.index, columns=prices.columns)
    s = ic_summary(information_coefficient(noise, forward_returns(prices)))
    assert abs(s["t_stat"]) < 3


def test_factors_do_not_use_future_prices(prices):
    cut = 400
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= 2
    a, b = compute_all(prices), compute_all(tampered)
    for name in a:
        pd.testing.assert_frame_equal(a[name].iloc[: cut + 1], b[name].iloc[: cut + 1])


def test_ic_of_perfect_signal_is_one(prices):
    fwd = forward_returns(prices)
    assert information_coefficient(fwd, fwd).mean() == pytest.approx(1.0)


def test_long_short_positive_for_planted_factor(prices):
    fwd = forward_returns(prices)
    ls = long_short(compute_all(prices, ["reversal_5d"])["reversal_5d"], fwd)
    assert ls.mean() > 0
