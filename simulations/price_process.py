import numpy as np
from numba import jit


@jit(nopython=True)
def generate_correlated_basket_prices_arrays(
    S0_array, mu_array, sigma_array, correlation_matrix, T, dt, n_paths=1000
):
    n_assets = len(S0_array)
    n_steps = int(T / dt) + 1

    L = np.linalg.cholesky(correlation_matrix)

    prices = np.zeros((n_assets, n_paths, n_steps))
    for i in range(n_assets):
        prices[i, :, 0] = S0_array[i]

    for path in range(n_paths):
        for step in range(1, n_steps):
            Z = np.random.randn(n_assets)
            corr_Z = L @ Z

            for i in range(n_assets):
                S_prev = prices[i, path, step - 1]
                dS = (
                    mu_array[i] * S_prev * dt
                    + sigma_array[i] * S_prev * np.sqrt(dt) * corr_Z[i]
                )
                prices[i, path, step] = S_prev + dS

    return prices


def generate_correlated_basket_prices(
    S0_list, mu_list, sigma_list, correlation_matrix, T, dt, n_paths=1000
):
    S0_array = np.array(S0_list)
    mu_array = np.array(mu_list)
    sigma_array = np.array(sigma_list)

    prices_array = generate_correlated_basket_prices_arrays(
        S0_array, mu_array, sigma_array, correlation_matrix, T, dt, n_paths
    )

    prices = {}
    for i in range(len(S0_list)):
        prices[f"St_{i+1}"] = prices_array[i]

    return prices


@jit(nopython=True)
def create_correlation_matrix(n_assets, correlation_ratio):
    correlation_matrix = np.full((n_assets, n_assets), correlation_ratio)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix
