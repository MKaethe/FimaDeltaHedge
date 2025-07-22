from bs_model import (
    run_random_example,
    monte_carlo_histogram,
    monte_carlo_error_vs_rebalance_period,
    run_stock_example,
)
from utility import get_stock_data


if __name__ == "__main__":
    # seeds for interesting scenarios
    # Close ATM (High delta towards end): 934
    S0 = 100
    K = 100
    r = 0.02
    sigma = 0.2
    m = 0.2
    T = 1.0
    """run_random_example(
        S0,
        K,
        r,
        sigma,
        m,
        T,
        seed=934,
        hedging_strategy="delta",
        option_type="call",
        rebalancing="monthly",
    )"""

    """# We work with minute timesteps
    dt = 1 / (252 * 6.5 * 60)
    time_array = np.arange(0.0, T + dt, dt)
    N = len(time_array)

    sigmas=np.linspace(0.2,0.7,N)"""

    """monte_carlo_histogram(
        S0,
        K,
        r,
        sigma,
        m,
        T,
        150,
        seed=42,
        option_type="call",
        hedging_strategy="delta",
        rebalancing="weekly",
    )"""

    """monte_carlo_error_vs_rebalance_period(
        S0,
        K,
        r,
        sigmas,
        m,
        T,
        150,
        seed=42,
        option_type="call",
        # rebalancing="monthly",
    )"""

    """ticker = "MSFT"
    data = get_stock_data(ticker).sort_index()


    run_stock_example(
        data, 1, hedging_strategy="delta-gamma-vega", rebalancing="monthly"
    )"""
