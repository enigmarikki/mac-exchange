import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
np.random.seed(200)
COLLATERAL = np.array([2e3, 0.2])
S0_COLLATERAL = np.array([50, 50e3])
N = 4
S0 = [50, 50e3, 150, 3e3] #HYPE, BTC, SOL, ETH

#Exercise 5: Portfolio-Level Risk with Multiple Positions and Multi-Asset Margin
#Scenario: You deposit 2,000 HYPE as collateral at $50/HYPE and 0.2 BTC at $50,000/BTC. The exchange offers at 90% max LTV (loan to value) on HYPE and BTC. Open two positions simultaneously:
#Long 10 ETH perp at $3,000/ETH (max leverage is 10x leverage).
#Short 100 SOL perp at $150/SOL (max leverage is 5x)
#Tasks:
#At entry: Calculate initial equity.
#Build a table for three price scenarios (assume BTC/HYPE move together as "majors" for simplicity; ETH/SOL as alts).
#Unrealized PnL (ETH long) at different prices
#Unrealized PnL (SOL short) at different prices
#New collateral: BTC value = 0.2 × new BTC price; HYPE value = 2,000 × new HYPE price.
#Total equity = (BTC value + HYPE value) + (ETH PnL + SOL PnL).
#Check if total equity > total MM.
#In which scenario does the portfolio risk liquidation first? Why?


# TODO: run sims to capture liq price of the basket -
# i think the liq price is a convex opt function with min criterion
def new_exchange(mark_prices: dict, L: float, collateral: np.array):
    path = 0
    S_hype = mark_prices["St_1"][path]
    S_btc = mark_prices["St_2"][path]
    S_sol = mark_prices["St_3"][path]
    S_eth = mark_prices["St_4"][path]
    
    for i in range(3600):
        ccv = S_hype * collateral[0] + S_btc * collateral[1]
        



def calculate_loan_value(ltv: float, collateral: np.array, mark_price: np.array):
    return collateral@mark_price*ltv


@jit(nopython=True)
def generate_correlated_basket_prices_arrays(S0_array, mu_array, sigma_array, correlation_matrix, T, dt, n_paths=1000):
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
                S_prev = prices[i, path, step-1]
                dS = mu_array[i] * S_prev * dt + sigma_array[i] * S_prev * np.sqrt(dt) * corr_Z[i]
                prices[i, path, step] = S_prev + dS

    return prices

def generate_correlated_basket_prices(S0_list, mu_list, sigma_list, correlation_matrix, T, dt, n_paths=1000):
    S0_array = np.array(S0_list)
    mu_array = np.array(mu_list)
    sigma_array = np.array(sigma_list)

    prices_array = generate_correlated_basket_prices_arrays(S0_array, mu_array, sigma_array, correlation_matrix, T, dt, n_paths)

    prices = {}
    for i in range(len(S0_list)):
        prices[f'St_{i+1}'] = prices_array[i]

    return prices

@jit(nopython=True)
def create_correlation_matrix(n_assets, correlation_ratio):
    correlation_matrix = np.full((n_assets, n_assets), correlation_ratio)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix

def simulate_basket(n_assets=3, correlation_ratio=0.5, S0_list=None, T=1.0, dt=0.01, n_paths=1000):
    if S0_list is None:
        S0_list = [100.0] * n_assets
    else:
        n_assets = len(S0_list)

    seconds_per_year = 365.25 * 24 * 3600
    mu_list = [0.05 / seconds_per_year] * n_assets
    sigma_list = [1 / np.sqrt(seconds_per_year)] * n_assets

    correlation_matrix = create_correlation_matrix(n_assets, correlation_ratio)

    prices = generate_correlated_basket_prices(
        S0_list, mu_list, sigma_list, correlation_matrix, T, dt, n_paths
    )
    
    time_grid = np.linspace(0, T, int(T/dt) + 1)

    return {
        'prices': prices,
        'time_grid': time_grid,
        'correlation_matrix': correlation_matrix,
        'parameters': {
            'S0': np.array(S0_list),
            'mu': np.array(mu_list),
            'sigma': np.array(sigma_list),
            'T': T,
            'dt': dt,
            'n_paths': n_paths,
            'correlation_ratio': correlation_ratio
        }
    }

if __name__ == "__main__":
    results = simulate_basket(
        n_assets=N,
        correlation_ratio=0.80,
        T=3600.0,
        dt=1.0,
        S0_list=S0,
        n_paths=1000  # Run many simulations for statistical analysis
    )

    print("Simulation complete!")
    print(f"Price matrix dimensions: {N} assets x {len(results['time_grid'])} time steps")
    print(f"Matrix shape for each path: {results['prices']['St_1'].shape}")
    print(f"Correlation matrix:\n{results['correlation_matrix']}")

    np.save('price_simulation_results.npy', results['prices'])
    print("Results saved to 'price_simulation_results.npy'")

    n_plots = min(N, 4)
    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    elif n_plots == 3:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

    time_subset = results['time_grid'][:3600]

    for i in range(n_plots):
        n_sample_paths = min(10, results['parameters']['n_paths'])  # Show up to 10 paths
        for path in range(n_sample_paths):
            price_subset = results['prices'][f'St_{i+1}'][path, :3600]
            axes[i].plot(time_subset, price_subset, alpha=0.7)
        axes[i].set_title(f'St_{i+1} - First 60 Minutes ({n_sample_paths} Sample Paths)')
        axes[i].set_xlabel('Time (seconds)')
        axes[i].set_ylabel('Price')
        axes[i].grid(True)

    if n_plots == 3:
        axes[3].set_visible(False)

    plt.tight_layout()
    plt.savefig('basket_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()