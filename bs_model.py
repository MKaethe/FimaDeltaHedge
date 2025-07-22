# Here are functions for BS modeling
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import hashlib
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import skew, kurtosis
import pandas as pd

from utility import (
    verify_directory_exists,
    load_from_path,
    save_to_path,
    get_stock_data,
)

MAX_WORKERS = 14


# Default creation for a simulated stock path the entries are assumed to be equally spaced
def euler_black_scholes_path(S0, N, m, sigma, dt, seed=None):
    # Seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Initialize price path and create dW
    S = np.zeros(N)
    S[0] = S0
    dW = np.random.normal(0.0, np.sqrt(dt), size=N - 1)

    # Is sigma constant or does it evolve over tiem?
    scalar_sigma = np.isscalar(sigma)

    # Is sigma as long as N?
    if not scalar_sigma:
        assert len(sigma) == N

    # Iterate
    for n in range(N - 1):
        # Get correct current volatility
        if scalar_sigma:
            vol = sigma
        else:
            vol = sigma[n]

        # Calculate price paths step by step
        dSt = m * S[n] * dt + vol * S[n] * dW[n]

        S[n + 1] = S[n] + dSt

    return S


# Generate n_paths
def monte_carlo_black_scholes(S0, N, m, sigma, dt, n_paths, seed=None, use_cache=True):
    # Is sigma constant or does it evolve over time?
    scalar_sigma = np.isscalar(sigma)

    # For caching we can save the random paths in files to prevent long running times...
    folder_path = "paths"
    verify_directory_exists(folder_path)

    # To also replicate nonconstant sigmas we can hash it!
    if not scalar_sigma:
        sigma_hash = hashlib.md5(np.array(sigma).tobytes()).hexdigest()[:8]
        filename = f"random_paths_{n_paths}_{seed}_vol_{sigma_hash}.pkl"
    else:
        filename = f"random_paths_{n_paths}_{seed}_vol_{sigma:.3f}.pkl"
    filepath = os.path.join(folder_path, filename)

    # If the paths with that size and seed are already cached, then we simply load
    if use_cache and seed != None:
        if os.path.exists(filepath):
            return load_from_path(filepath)

    # Random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # We get a matrix with all the paths
    paths = np.zeros((n_paths, N))
    for i in tqdm.tqdm(range(n_paths), "Generating paths", n_paths):
        paths[i] = euler_black_scholes_path(S0, N, m, sigma, dt)

    # if we want to cache, we simply save to the file
    if use_cache and seed != None:
        save_to_path(filepath, paths)

    return paths


# A fast and vectorized version of black scholes
def black_scholes_vectorized(S, K, tte, r, sigma, option_type="call"):
    # Check for nonconstant sigma again
    scalar_sigma = np.isscalar(sigma)

    # shape checks
    if scalar_sigma:
        assert S.shape == tte.shape
    else:
        assert S.shape == tte.shape == sigma.shape

    # Avoid divide by zero or log(0)
    eps = 1e-10
    S = np.maximum(S, eps)
    tte = np.maximum(tte, eps)

    # Pre-calculate some values to enhance performance
    sqrt_tte = np.sqrt(tte)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tte) / (sigma * sqrt_tte)
    d2 = d1 - sigma * sqrt_tte
    pdf_d1 = norm.pdf(d1)

    # Calculate the greeks based on wether it is call or put
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * tte) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -S * pdf_d1 * sigma / (2 * sqrt_tte) - r * K * np.exp(
            -r * tte
        ) * norm.cdf(d2)
        rho = K * tte * np.exp(-r * tte) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * tte) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = -S * pdf_d1 * sigma / (2 * sqrt_tte) + r * K * np.exp(
            -r * tte
        ) * norm.cdf(-d2)
        rho = -K * tte * np.exp(-r * tte) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # These two greeks are the same for both
    gamma = pdf_d1 / (S * sigma * sqrt_tte)
    vega = S * pdf_d1 * sqrt_tte

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


# This is the forward method of delta hedging, it returns the hedged portfolio value
def delta_hedging_forward(S, greeks, r, time_array, rebalance_period=0):
    # retrieve greeks needed for hedging
    derivate_price = greeks["price"]
    delta = greeks["delta"]

    # shape checks
    assert delta.shape == derivate_price.shape == S.shape == time_array.shape

    # initializing the result arrays
    N = len(delta)
    portfolio_value = np.zeros(N)
    hedging_error = np.zeros(N)

    # the portfolio value is the initial option price of course, we can calculate the cash as well
    portfolio_value[0] = derivate_price[0]

    # Cash account to make the portfolio self‑financing
    cash = derivate_price[0] - delta[0] * S[0]

    # Due to different rebalancing periods we have to hold delta_pos for the rebalancing time
    delta_pos = delta[0]

    # Track the last rebalance time
    t_last_rebalance = time_array[0]

    # forward iterations
    for i in range(1, N):
        # Grow the cash
        dt = time_array[i] - time_array[i - 1]
        cash *= np.exp(r * dt)

        # Get the current timestamp
        t = time_array[i]

        # If we surpassed the rebalance period, we rebalance
        if t - t_last_rebalance > rebalance_period:
            t_last_rebalance = t
            cash -= (delta[i] - delta_pos) * S[i]
            delta_pos = delta[i]

        # Track portfolio value and error
        portfolio_value[i] = delta_pos * S[i] + cash
        hedging_error[i] = portfolio_value[i] - derivate_price[i]

    return {
        "hedging_error": hedging_error,
        "portfolio_value": portfolio_value,
    }


# This is the forward method of delta-gamma hedging, it returns the hedged portfolio value
def delta_gamma_hedging_forward(
    S, greeks_F, greeks_G, r, time_array, rebalance_period=0.0
):
    # retrieve greeks needed for hedging, F is the target derivative
    price_F = greeks_F["price"]
    delta_F = greeks_F["delta"]
    gamma_F = greeks_F["gamma"]

    # greeks of the second derivative
    N = len(S)
    price_G = greeks_G["price"][:N]
    delta_G = greeks_G["delta"][:N]
    gamma_G = greeks_G["gamma"][:N]

    # shape checks
    assert (
        S.shape == price_F.shape == delta_F.shape == gamma_F.shape == time_array.shape
    )

    # initializing the result arrays
    portfolio_value = np.zeros(N)
    hedging_error = np.zeros(N)
    q_path = np.zeros(N)  # position in option G
    x_path = np.zeros(N)  # position in underlying

    # starting values
    q_pos = gamma_F[0] / gamma_G[0]  # keep gamma neutral
    x_pos = delta_F[0] - q_pos * delta_G[0]  # keep delta neutral

    # Cash account to make the portfolio self‑financing
    cash = price_F[0] - q_pos * price_G[0] - x_pos * S[0]

    # Tracking the values
    portfolio_value[0] = price_F[0]
    hedging_error[0] = 0.0
    q_path[0] = q_pos
    x_path[0] = x_pos

    # Track the last rebalance time
    t_last_rebalance = time_array[0]

    # forward iterations
    for i in range(1, N):
        # Grow cash
        dt = time_array[i] - time_array[i - 1]
        cash *= np.exp(r * dt)

        # Get the current timestamp
        t = time_array[i]

        # If we surpassed the rebalance period, we rebalance
        if t - t_last_rebalance > rebalance_period:
            t_last_rebalance = t

            q_new = gamma_F[i] / gamma_G[i]
            x_new = delta_F[i] - q_new * delta_G[i]

            # rebalance both derivative and underlying
            cash -= (q_new - q_pos) * price_G[i] + (x_new - x_pos) * S[i]

            q_pos, x_pos = q_new, x_new

        # Track portfolio value and error
        portfolio_value[i] = q_pos * price_G[i] + x_pos * S[i] + cash
        hedging_error[i] = portfolio_value[i] - price_F[i]
        q_path[i] = q_pos
        x_path[i] = x_pos

    return {
        "portfolio_value": portfolio_value,
        "hedging_error": hedging_error,
        "q_path": q_path,
        "x_path": x_path,
    }


# This is the forward method of delta-gamma-vega hedging, it returns the hedged portfolio value
def delta_gamma_vega_hedging_forward(
    S, greeks_F, greeks_G, greeks_H, r, time_array, rebalance_period=0.0
):
    # retrieve greeks needed for hedging, F is the target derivative
    price_F = greeks_F["price"]
    delta_F = greeks_F["delta"]
    gamma_F = greeks_F["gamma"]
    vega_F = greeks_F["vega"]

    # greeks of the second and third derivative
    N = len(S)
    price_G = greeks_G["price"][:N]
    delta_G = greeks_G["delta"][:N]
    gamma_G = greeks_G["gamma"][:N]
    vega_G = greeks_G["vega"][:N]

    price_H = greeks_H["price"][:N]
    delta_H = greeks_H["delta"][:N]
    gamma_H = greeks_H["gamma"][:N]
    vega_H = greeks_H["vega"][:N]

    # shape checks
    assert (
        S.shape
        == price_F.shape
        == delta_F.shape
        == gamma_F.shape
        == vega_F.shape
        == time_array.shape
    )

    # initializing the result arrays
    portfolio_value = np.zeros(N)
    hedging_error = np.zeros(N)
    q_path = np.zeros(N)  # position in option G
    p_path = np.zeros(N)  # position in option H
    x_path = np.zeros(N)  # position in underlying

    # starting values, these calculations come from solving the linear equation system
    det = gamma_G[0] * vega_H[0] - gamma_H[0] * vega_G[0]
    q_pos = (gamma_F[0] * vega_H[0] - gamma_H[0] * vega_F[0]) / det
    p_pos = (gamma_G[0] * vega_F[0] - gamma_F[0] * vega_G[0]) / det
    x_pos = delta_F[0] - q_pos * delta_G[0] - p_pos * delta_H[0]

    # Cash account to make the portfolio self‑financing
    cash = price_F[0] - q_pos * price_G[0] - p_pos * price_H[0] - x_pos * S[0]

    # Tracking the values
    portfolio_value[0] = price_F[0]
    hedging_error[0] = 0.0
    q_path[0] = q_pos
    p_path[0] = p_pos
    x_path[0] = x_pos

    # Track the last rebalance time
    t_last_rebalance = time_array[0]

    # forward iterations
    for i in range(1, N):
        # Grow cash
        dt = time_array[i] - time_array[i - 1]
        cash *= np.exp(r * dt)

        # Get the current timestamp
        t = time_array[i]

        # If we surpassed the rebalance period, we rebalance
        if t - t_last_rebalance > rebalance_period:
            t_last_rebalance = t

            det_i = gamma_G[i] * vega_H[i] - gamma_H[i] * vega_G[i]
            q_new = (gamma_F[i] * vega_H[i] - gamma_H[i] * vega_F[i]) / det_i
            p_new = (gamma_G[i] * vega_F[i] - gamma_F[i] * vega_G[i]) / det_i
            x_new = delta_F[i] - q_new * delta_G[i] - p_new * delta_H[i]

            # rebalance both derivative and underlying
            cash -= (
                (q_new - q_pos) * price_G[i]
                + (p_new - p_pos) * price_H[i]
                + (x_new - x_pos) * S[i]
            )

            q_pos, p_pos, x_pos = q_new, p_new, x_new

        # Track portfolio value and error
        portfolio_value[i] = (
            q_pos * price_G[i] + p_pos * price_H[i] + x_pos * S[i] + cash
        )
        hedging_error[i] = portfolio_value[i] - price_F[i]
        q_path[i] = q_pos
        p_path[i] = p_pos
        x_path[i] = x_pos

    return {
        "portfolio_value": portfolio_value,
        "hedging_error": hedging_error,
        "q_path": q_path,  # option G position
        "p_path": p_path,  # option H position
        "x_path": x_path,  # underlying position
    }


# Wrapper for the different hedging strategy
def hedge_portfolio_path(
    S,
    K,
    r,
    sigma,
    T,
    time_array,
    rebalance_period,
    option_type="call",
    hedging_strat="delta",
):
    # Is sigma constant or does it evolve over time?
    scalar_sigma = np.isscalar(sigma)

    # shape checks
    if scalar_sigma:
        assert S.shape == time_array.shape
    else:
        assert S.shape == time_array.shape == sigma.shape

    # The derivates are also call options, so we modify their parameters
    K_G = 1.1 * K
    T_G = T + 2.0

    K_H = 0.8 * K
    T_H = T + 1.0

    tte = T - time_array
    tte_G = T_G - time_array
    tte_H = T_H - time_array

    # Depending on the type of hedging, do the following:
    if hedging_strat == "delta":
        greeks_F = black_scholes_vectorized(S, K, tte, r, sigma, option_type)
        result = delta_hedging_forward(S, greeks_F, r, time_array, rebalance_period)

    elif hedging_strat == "delta-gamma":
        greeks_F = black_scholes_vectorized(S, K, tte, r, sigma, option_type)
        greeks_G = black_scholes_vectorized(S, K_G, tte_G, r, sigma, option_type)
        result = delta_gamma_hedging_forward(
            S, greeks_F, greeks_G, r, time_array, rebalance_period
        )

    elif hedging_strat == "delta-gamma-vega":
        greeks_F = black_scholes_vectorized(S, K, tte, r, sigma, option_type)
        greeks_G = black_scholes_vectorized(S, K_G, tte_G, r, sigma, option_type)
        greeks_H = black_scholes_vectorized(S, K_H, tte_H, r, sigma, option_type)
        result = delta_gamma_vega_hedging_forward(
            S, greeks_F, greeks_G, greeks_H, r, time_array, rebalance_period
        )

    else:
        raise ValueError("No valid hedging strategy given!")

    return result, greeks_F


# Overall a plotting function to demonstrate performance of hedging strategy
def plot_results_single(
    S, K, time_array, greeks_F, portfolio_values, hedging_error, option_type="call"
):
    # The figures and stuff...
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12, 8))

    # Calculating the final payoff and retrieving other data
    final_value = S[-1]
    option_prices = greeks_F["price"]
    deltas = greeks_F["delta"]

    if option_type == "call":
        payoff = max(final_value - K, 0)
    elif option_type == "put":
        payoff = max(final_value - K, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Plotting the portfolio value vs the option price
    axs[0, 0].plot(
        time_array,
        portfolio_values,
        color="red",
        label="Portfolio value",
        linewidth=1,
    )
    axs[0, 0].plot(
        time_array, option_prices, color="blue", label="Option price", linewidth=1
    )
    axs[0, 0].axhline(payoff, color="black", label=f"Real Payoff: {payoff:.2f}")
    axs[0, 0].grid()
    axs[0, 0].legend()
    axs[0, 0].set_xlabel("t [yrs]")
    axs[0, 0].set_ylabel("Price [$]")
    axs[0, 0].set_title("Option price and replicated portfolio value")

    # Plotting the underlying price over time
    axs[1, 0].plot(time_array, S, color="black", linewidth=1)
    axs[1, 0].axhline(K, color="black", label="Strike price")
    axs[1, 0].grid()
    axs[1, 0].set_xlabel("t [yrs]")
    axs[1, 0].set_ylabel("Price [$]")
    axs[1, 0].set_title("Stock price")

    # Plotting the hedging error over time
    axs[0, 1].plot(time_array, hedging_error, color="black")
    axs[0, 1].axhline(
        hedging_error[-1],
        color="black",
        label=f"Final error: ${hedging_error[-1]:.2f}",
    )
    axs[0, 1].grid()
    axs[0, 1].set_xlabel("t [yrs]")
    axs[0, 1].set_ylabel("Error [$]")
    axs[0, 1].set_title("Hedging error")
    axs[0, 1].legend()

    # Plotting the Delta over time
    axs[1, 1].plot(time_array, deltas, color="black", linewidth=1)
    axs[1, 1].grid()
    axs[1, 1].set_xlabel("t [yrs]")
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].set_title("Delta")

    plt.tight_layout()
    # plt.savefig("./benchmark_increasing_vol_delta_vega_gamma_monthly.png")
    plt.show()


# A function that can return a dict with the periods
def get_string_to_period_dict(dt):
    periods = {
        "instant": 0,
        "hourly": dt * 60,
        "bihourly": dt * 60 * 2,
        "daily": dt * 60 * 6.5,
        "bidaily": dt * 60 * 6.5 * 2,
        "weekly": dt * 60 * 6.5 * 5,
        "biweekly": dt * 60 * 6.5 * 5 * 2,
        "monthly": dt * 60 * 6.5 * 5 * 4,
        "bimonthly": dt * 60 * 6.5 * 5 * 4 * 2,
    }

    return periods


# Here we simulate one path and apply and evaluate the hedging strategy
def run_random_example(
    S0=100,
    K=100,
    r=0.02,
    sigma=0.2,
    m=0.2,
    T=1.0,
    seed=42,
    hedging_strategy="delta",
    option_type="call",
    rebalancing="weekly",
):
    # We work with minute timesteps
    dt = 1 / (252 * 6.5 * 60)
    time_array = np.arange(0.0, T + dt, dt)
    N = len(time_array)

    # Period dict
    stp_dict = get_string_to_period_dict(dt)
    rebalance_period = stp_dict[rebalancing]

    # Receive the simulated price path
    S = euler_black_scholes_path(S0, N, m, sigma, dt, seed)

    # Now we simply hedge a replicated portfolio and get the results
    result, greeks = hedge_portfolio_path(
        S, K, r, sigma, T, time_array, rebalance_period, option_type, hedging_strategy
    )

    # We now plot the result
    plot_results_single(
        S,
        K,
        time_array,
        greeks,
        result["portfolio_value"],
        result["hedging_error"],
        option_type,
    )


# Here we create n_paths paths and generate a histogram to evaluate the distribution of errors
def monte_carlo_histogram(
    S0=100,
    K=100,
    r=0.02,
    sigma=0.2,
    m=0.2,
    T=1.0,
    n_paths=1000,
    seed=42,
    hedging_strategy="delta",
    option_type="call",
    rebalancing="weekly",
):
    # We work with minute timesteps
    dt = 1 / (252 * 6.5 * 60)
    time_array = np.arange(0.0, T + dt, dt)
    N = len(time_array)

    # Period dict
    stp_dict = get_string_to_period_dict(dt)
    rebalance_period = stp_dict[rebalancing]

    # Generate a number of paths
    paths = monte_carlo_black_scholes(
        S0, N, m, sigma, dt, n_paths, seed, use_cache=True
    )

    n_paths, N = paths.shape

    # Initialize a list to collect the errors
    final_errors = []

    # Use the hedging strategy to get the final error
    for i in tqdm.tqdm(range(n_paths), "Evaluating paths", n_paths):
        current_path = paths[i]

        result, greeks = hedge_portfolio_path(
            current_path,
            K,
            r,
            sigma,
            T,
            time_array,
            rebalance_period,
            option_type,
            hedging_strategy,
        )

        final_errors.append(result["hedging_error"][-1])

    final_errors = np.array(final_errors)

    # Compute 4 moments
    mean = np.mean(final_errors)
    variance = np.var(final_errors)
    skewness = skew(final_errors)
    excess_kurtosis = kurtosis(final_errors)  # default is Fisher kurtosis (subtracts 3)

    # Print the 4 moments
    print(f"Mean:           {mean:.6f}")
    print(f"Variance:       {variance:.6f}")
    print(f"Skewness:       {skewness:.6f}")
    print(f"Excess Kurtosis:{excess_kurtosis:.6f}")

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(
        final_errors,
        bins=50,
        density=True,
        alpha=0.75,
        color="steelblue",
        edgecolor="black",
    )
    plt.title(f"Hedging Error Distribution ({hedging_strategy.capitalize()} Hedge)")
    plt.xlabel("Final Hedging Error")
    plt.ylabel("Density")
    plt.axvline(0, color="black", label="Zero Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Here we compare different hedging strategies with different rebalancing  time
def monte_carlo_error_vs_rebalance_period(
    S0=100,
    K=100,
    r=0.02,
    sigma=0.2,
    m=0.2,
    T=1.0,
    n_paths=1000,
    seed=42,
    option_type="call",
):
    dt = 1 / (252 * 6.5 * 60)
    time_array = np.arange(0.0, T + dt, dt)

    # Simulate/load paths once
    paths = monte_carlo_black_scholes(
        S0, len(time_array), m, sigma, dt, n_paths, seed, use_cache=True
    )

    # Rebalancing
    rebalance_dict = get_string_to_period_dict(dt)
    
    # The strategies to test and colors
    strategies = ["delta", "delta-gamma", "delta-gamma-vega"]
    colours = ["steelblue", "darkorange", "seagreen"]

    # Containers for collecting mean errors
    mean_errors = {s: [] for s in strategies}

    # X-axis, rebalance periods
    x_periods = np.array(list(rebalance_dict.values()))
    labels = list(rebalance_dict.keys())

    # For every rebalances, for every strategy, simulate n_path paths and calculate error
    for label, period in rebalance_dict.items():
        for strat in strategies:
            final_err = np.empty(n_paths)

            for i in tqdm.tqdm(range(n_paths), leave=False, desc=f"{strat} | {label}"):
                S_path = paths[i]
                result, _ = hedge_portfolio_path(
                    S_path, K, r, sigma, T, time_array, period, option_type, strat
                )
                final_err[i] = result["hedging_error"][-1]

            mean_errors[strat].append(np.mean(np.abs(final_err)))

    # plotting results
    plt.figure(figsize=(10, 6))

    for strat, color in zip(strategies, colours):
        y_vals = mean_errors[strat]
        plt.plot(x_periods, y_vals, marker="o", label=strat.capitalize(), color=color)

    y_vals = mean_errors["delta"]
    for x, y, lbl in zip(x_periods, y_vals, labels):
        plt.annotate(
            lbl,
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            ha="left",
            va="bottom",
        )

    plt.xlabel("Rebalance period [yrs]")
    plt.ylabel("Mean absolute error")
    plt.title("Hedging error vs rebalance period\n(three hedging methods)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Here we try to apply the strategies onto real stock data
def run_stock_example(
    data,
    T=1.0,
    K=None,
    r=0.02,
    hedging_strategy="delta",
    option_type="call",
    rebalancing="weekly",
    seed=42,
    rolling_window_days=365,
    start_date=None
):
    # Get close values
    close = data["Close"]



    # Pick random date if no start date given
    rng = np.random.default_rng(seed)
    max_start = close.index[-1] - pd.Timedelta(days=365 * T)
    min_start = close.index[0] + pd.Timedelta(days=rolling_window_days)
    valid_index = close.index[(close.index <= max_start) & (close.index > min_start)]

    if start_date:
        start_time=start_date
    else:  
        start_time = rng.choice(valid_index)
    end_time = start_time + pd.Timedelta(days=365 * T)

    print(f"Timespan: {start_time}-{end_time}")

    # Get the window of the indices
    window = close.loc[start_time:end_time]

    # The window is 1 (real) year long, dt estimation
    deltas_sec = window.index.to_series().diff().dropna().dt.total_seconds()
    avg_dt_seconds = deltas_sec.mean()
    seconds_per_year = 365 * 24 * 3600
    dt_year = avg_dt_seconds / seconds_per_year

    # time grid
    time_array = (window.index - start_time).total_seconds() / seconds_per_year

    # underlying values
    S = window.values

    # Strike ATM
    if K is None:
        K = S[0]

    # Rebalance periods
    periods = {
        "instant": 0.0,
        "hourly": 1 / (365 * 24),
        "daily": 1 / 365,
        "weekly": 7 / 365,
        "monthly": 30 / 365,
    }
    rebalance_period = periods[rebalancing]

    # volatility estimation using rolling window
    log_price_full = np.log(close)
    returns_full = log_price_full.diff()

    # time-based rolling std over returns
    rolling_std = returns_full.rolling(f"{rolling_window_days}D",min_periods=24*60*30).std()

    # annualizing
    annual_factor = np.sqrt(1.0 / dt_year)
    sigma_series = rolling_std * annual_factor

    # Align sigma with truncated expiry window
    sigma_input = sigma_series.loc[start_time:end_time].values

    # run hedging
    result, greeks = hedge_portfolio_path(
        S,
        K,
        r,
        sigma_input,
        T,
        time_array,
        rebalance_period,
        option_type=option_type,
        hedging_strat=hedging_strategy,
    )

    # plotting
    plot_results_single(
        S,
        K,
        time_array,
        greeks,
        result["portfolio_value"],
        result["hedging_error"],
        option_type=option_type,
    )



