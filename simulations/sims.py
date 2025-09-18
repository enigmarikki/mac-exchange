import numpy as np
from typing import Dict, List, Optional
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from collections import defaultdict
from agent import Agent, HealthStatus
from exchange import BoostExchange
from price_process import generate_correlated_basket_prices


@dataclass
class AgentCategory:
    name: str
    ltv_ratio: float
    leverage_ratio: float
    num_agents: int
    min_usd: float
    max_usd: float


def create_agent_population(
    num_agents: int = 1000,
    seed: int = 42,
    assets: Optional[List[str]] = None,
    initial_prices: Optional[Dict[str, float]] = None,
) -> Dict[int, Agent]:
    """
    Create a realistic population with whales, mids, plebs, and degens.

    Categories:
    - Whales: >$1MM collateral, 70% LTV, 3.5x leverage (conservative institutions)
    - Mids: $100k-1M collateral, 60% LTV, 2.5x leverage (retail traders)
    - Plebs: $1k-100k collateral, 40% LTV, 1.75x leverage (cautious retail)
    - Degens: $5k-500k collateral, 90% LTV, 8x leverage (high-risk traders)
    """
    if assets is None or initial_prices is None:
        raise ValueError("assets and initial_prices must be provided")

    np.random.seed(seed)

    categories = [
        AgentCategory(
            "whale", 0.70, 3.5, num_agents // 10, 1_000_000.0, 10_000_000.0
        ),  # 10% conservative whales
        AgentCategory(
            "mid", 0.60, 2.5, num_agents // 4, 100_000.0, 1_000_000.0
        ),  # 25% mids
        AgentCategory(
            "pleb", 0.40, 1.75, num_agents // 3, 1_000.0, 100_000.0
        ),  # 33% plebs
        AgentCategory(
            "degen",
            0.90,
            8.0,
            num_agents - num_agents // 10 - num_agents // 4 - num_agents // 3,
            5_000.0,
            500_000.0,
        ),  # 32% degens
    ]

    agents = {}
    agent_counter = 0

    for category in categories:
        for _ in range(category.num_agents):
            agent = Agent(agent_id=agent_counter)

            # Randomly choose collateral asset
            collateral_asset = np.random.choice(assets)

            # Random USD value in category range
            usd_value = np.random.uniform(category.min_usd, category.max_usd)

            # Calculate units
            units = usd_value / initial_prices[collateral_asset]
            agent.collateral_deposited[collateral_asset] = units

            agent.category = category.name  # type: ignore
            agent.ltv_ratio = category.ltv_ratio
            agent.leverage_ratio = category.leverage_ratio

            agents[agent_counter] = agent
            agent_counter += 1

    print(f"Created {len(agents)} agents:")
    for cat in categories:
        count = sum(1 for a in agents.values() if a.category == cat.name)
        print(
            f"  - {cat.name}: {count} agents ({cat.ltv_ratio*100:.0f}% LTV, {cat.leverage_ratio*100:.0f}% leverage, USD {cat.min_usd:,.0f} - {cat.max_usd:,.0f})"
        )

    return agents


def setup_exchange(
    agents: Dict[int, Agent],
    assets: List[str],
    initial_prices: Dict[str, float],
    ltvs: Dict[str, float],
    max_leverage: Dict[str, float],
) -> BoostExchange:
    """Initialize the BoostExchange with 4 markets."""

    _ = assets  # Mark as used
    markets = initial_prices.copy()

    reserve_ratio = 0.1
    usdc_pool = 500_000_000.0
    liquidation_slack = 0.03

    exchange = BoostExchange(
        ltvs=ltvs,
        reserve_ratio=reserve_ratio,
        usdc_pool=usdc_pool,
        max_leverage=max_leverage,
        markets=markets,
        liquidation_slack=liquidation_slack,
        agents=agents,
    )

    # Deposits
    for agent in agents.values():
        exchange.handle_agent_deposit(agent)

    # Loans and positions
    for agent_id, agent in agents.items():
        try:
            # Max loanable
            max_loanable = sum(
                units * initial_prices[asset] * ltvs[asset]
                for asset, units in agent.collateral_deposited.items()
            )
            requested_loan = max_loanable * agent.ltv_ratio

            exchange.handle_loan(agent_id, requested_loan)

            # Open position in the collateral asset (for simplicity)
            position_asset = list(agent.collateral_deposited.keys())[0]
            position_size = (requested_loan * agent.leverage_ratio) / initial_prices[
                position_asset
            ]

            agent.positions[position_asset] = {
                "size": position_size,
                "entry_price": initial_prices[position_asset],
                "direction": 1,  # Long
            }

            # print(f"Agent {agent_id} ({agent.category}): Collateral {position_asset} {units:.4f} (${usd_value:,.0f}), Borrowed ${requested_loan:,.0f}, Position {position_size:.4f}")

        except Exception as e:
            print(f"Failed to setup agent {agent_id}: {e}")

    print(f"\nExchange setup complete:")
    print(f"Total loans issued: ${exchange.total_loans_issued:,.0f}")
    print(f"Available USDC: ${exchange.calculate_available_usdc():,.0f}")

    return exchange


def calculate_liquidation_slippage(
    asset: str,
    liquidation_amount_usd: float,
    current_price: float,  # noqa: ARG001
    market_stress_factor: float = 1.0,
) -> float:
    """
    Calculate slippage during forced liquidation of collateral.

    Args:
        asset: Asset being liquidated
        liquidation_amount_usd: USD value being liquidated
        current_price: Current market price
        market_stress_factor: Multiplier for stress conditions (1.0 = normal, 2.0 = high stress)

    Returns:
        Slippage as decimal (e.g., 0.05 = 5% slippage)
    """
    # Base slippage rates by asset (more liquid = lower slippage)
    base_slippage = {
        "BTC": 0.002,  # 0.2% base slippage
        "ETH": 0.003,  # 0.3% base slippage
        "SOL": 0.008,  # 0.8% base slippage
        "HYPE": 0.025,  # 2.5% base slippage (illiquid)
    }

    # Size impact: larger liquidations have more slippage
    size_buckets = [
        (10_000, 1.0),  # <$10k: no additional slippage
        (100_000, 1.5),  # $10k-100k: 1.5x slippage
        (1_000_000, 2.5),  # $100k-1M: 2.5x slippage
        (10_000_000, 4.0),  # $1M-10M: 4x slippage
        (float("inf"), 6.0),  # >$10M: 6x slippage (whale liquidation)
    ]

    size_multiplier = 1.0
    for threshold, multiplier in size_buckets:
        if liquidation_amount_usd <= threshold:
            size_multiplier = multiplier
            break

    # Calculate total slippage
    total_slippage = (
        base_slippage.get(asset, 0.02)  # Default 2% for unknown assets
        * size_multiplier
        * market_stress_factor
    )

    # Cap slippage at 50% (market completely broken)
    return min(total_slippage, 0.50)


def apply_liquidation_price_impact(
    current_prices: Dict[str, float],
    liquidations_this_tick: Dict[str, float],  # asset -> USD amount liquidated
    correlation_spillover: float = 0.3,
) -> Dict[str, float]:
    """
    Apply price impact from liquidations to market prices.

    Args:
        current_prices: Current market prices
        liquidations_this_tick: USD value liquidated by asset
        correlation_spillover: How much liquidation in one asset affects others

    Returns:
        Updated prices after liquidation impact
    """
    updated_prices = current_prices.copy()

    # Calculate direct price impact for each asset
    for asset, liquidated_usd in liquidations_this_tick.items():
        if liquidated_usd == 0:
            continue

        # Price impact based on liquidation size
        if liquidated_usd < 50_000:
            impact = 0.001  # 0.1%
        elif liquidated_usd < 500_000:
            impact = 0.005  # 0.5%
        elif liquidated_usd < 5_000_000:
            impact = 0.04  # 2%
        else:
            impact = 0.07  # 5% for massive liquidations

        # Apply direct impact
        updated_prices[asset] *= 1 - impact

        # Apply spillover to correlated assets
        for other_asset in updated_prices.keys():
            if other_asset != asset:
                spillover_impact = impact * correlation_spillover
                updated_prices[other_asset] *= 1 - spillover_impact

    return updated_prices


def run_tick_by_tick_simulation(
    exchange: BoostExchange,
    scenario_name: str,
    total_drop: float,
    n_steps: int,
    assets: List[str],
    initial_prices: Dict[str, float],
    sigma_list: List[float],
    corr_matrix: np.ndarray,
):
    """Run tick-by-tick stress test with correlated price paths."""

    print(f"\n{'='*60}")
    print(f"TICK-BY-TICK SIMULATION: {scenario_name}")
    print(f"{'='*60}")

    # Generate realistic price paths - smoother crash with recovery
    S0_list = [initial_prices[a] for a in assets]

    # Create a realistic crash pattern: gradual decline with sharp drops
    crash_multipliers = {
        "BTC": 0.7,  # BTC holds up better (30% drop)
        "ETH": 0.8,  # ETH drops more (35% drop)
        "SOL": 1.2,  # SOL gets hit harder (40% drop)
        "HYPE": 1.8,  # HYPE collapses (54% drop) - reduced from 2.0 to avoid log(negative)
    }

    mu_list = []
    for i, asset in enumerate(assets):
        asset_drop = total_drop * crash_multipliers[asset]
        # Ensure we don't get log of negative number
        asset_drop = min(asset_drop, 0.95)  # Cap at 95% drop
        # Smooth drift to target price
        mu_list.append(np.log(1 - asset_drop) / 1.0)

    T = 1.0
    dt = T / n_steps

    # Use lower volatility for smoother paths
    smooth_sigma = [
        vol * 0.3 for vol in sigma_list
    ]  # Much lower vol for realistic paths

    prices_dict = generate_correlated_basket_prices(
        S0_list, mu_list, smooth_sigma, corr_matrix, T, dt, n_paths=1
    )

    # Track stats over time
    full_liq_over_time = []
    partial_liq_over_time = []
    usdc_pool_over_time = []
    liquidation_stats = {
        step: defaultdict(lambda: {"full": 0, "partial": 0})
        for step in range(n_steps + 1)
    }

    # Enhanced Pool PnL Tracking
    pool_pnl_data = {
        "step": [],
        "usdc_pool": [],
        "liquidation_revenue": [],
        "lending_income": [],
        "bad_debt": [],
        "total_loans_outstanding": [],
        "collateral_value": [],
        "net_pool_pnl": [],
        "pool_utilization": [],
        "effective_yield": [],
    }

    initial_pool_size = exchange.usdc_pool
    cumulative_liquidation_revenue = 0.0
    cumulative_bad_debt = 0.0

    # Initial health check
    exchange.health_check()
    full_liq_over_time.append(0)  # Start with 0 liquidations
    partial_liq_over_time.append(0)
    usdc_pool_over_time.append(exchange.usdc_pool)

    # Initial pool PnL tracking
    total_collateral_value = sum(
        exchange.collateral_pool[asset] * initial_prices[asset] for asset in assets
    )

    pool_pnl_data["step"].append(0)
    pool_pnl_data["usdc_pool"].append(exchange.usdc_pool)
    pool_pnl_data["liquidation_revenue"].append(0.0)
    pool_pnl_data["lending_income"].append(0.0)
    pool_pnl_data["bad_debt"].append(0.0)
    pool_pnl_data["total_loans_outstanding"].append(exchange.total_loans_issued)
    pool_pnl_data["collateral_value"].append(total_collateral_value)
    pool_pnl_data["net_pool_pnl"].append(0.0)
    pool_pnl_data["pool_utilization"].append(
        exchange.total_loans_issued / exchange.usdc_total
        if exchange.usdc_total > 0
        else 0
    )
    pool_pnl_data["effective_yield"].append(0.0)

    cumulative_full = 0
    cumulative_partial = 0

    for step in range(1, n_steps + 1):
        # Get current prices from path
        current_prices = {
            assets[i]: prices_dict[f"St_{i+1}"][0, step - 1] for i in range(len(assets))
        }

        # Track liquidations this tick for price impact
        liquidations_this_tick = {asset: 0.0 for asset in assets}

        # Calculate market stress factor (higher during black swan)
        if "BLACK SWAN" in scenario_name:
            # Stress increases as more liquidations happen
            total_liquidated = cumulative_full + cumulative_partial
            market_stress_factor = 1.0 + min(
                total_liquidated / 1000.0, 3.0
            )  # Cap at 4x stress
        else:
            market_stress_factor = 1.0

        # Update markets first
        exchange.update_mark_price(current_prices)

        # Check for new liquidations
        exchange.health_check()

        # Process partial liquidations with slippage
        partial_liquidated_agents = list(exchange.partial_liquidation_ids)
        new_partial = len(partial_liquidated_agents)
        step_liquidation_revenue = 0.0

        for aid in partial_liquidated_agents:
            agent = exchange.agents[aid]

            # Calculate pre-liquidation position value
            pre_liquidation_value = 0.0
            for asset, position in agent.positions.items():
                pre_liquidation_value += (
                    abs(position.get("size", 0)) * current_prices[asset]
                )

            # Calculate liquidation amount and slippage
            for asset, units in agent.collateral_deposited.items():
                liquidation_usd = (
                    units * current_prices[asset] * 0.5
                )  # Assume 50% liquidated
                slippage = calculate_liquidation_slippage(
                    asset, liquidation_usd, current_prices[asset], market_stress_factor
                )

                # Account for slippage in liquidation proceeds
                actual_proceeds = liquidation_usd * (1 - slippage)
                step_liquidation_revenue += (
                    liquidation_usd - actual_proceeds
                )  # Protocol keeps slippage

                liquidations_this_tick[asset] += liquidation_usd

                # Print slippage info for very large liquidations only
                if liquidation_usd > 1_000_000:
                    print(
                        f"Tick {step}: Partial liquidation {asset} ${liquidation_usd:,.0f} with {slippage:.1%} slippage"
                    )

            exchange.partial_liquidate_user(aid)

        cumulative_partial += new_partial
        cumulative_liquidation_revenue += step_liquidation_revenue

        # Process full liquidations with slippage
        full_liquidated_agents = list(exchange.total_liquidation_ids)
        new_full = len(full_liquidated_agents)
        step_bad_debt = 0.0

        for aid in full_liquidated_agents:
            agent = exchange.agents[aid]

            # Calculate total liquidation value
            total_liquidation_value = 0.0

            # Value of positions
            for asset, position in agent.positions.items():
                position_value = abs(position.get("size", 0)) * current_prices[asset]
                total_liquidation_value += position_value

            # Value of collateral
            for asset, units in agent.collateral_deposited.items():
                liquidation_usd = units * current_prices[asset]
                slippage = calculate_liquidation_slippage(
                    asset, liquidation_usd, current_prices[asset], market_stress_factor
                )

                # Account for slippage in actual proceeds
                actual_proceeds = liquidation_usd * (1 - slippage)
                total_liquidation_value += actual_proceeds

                # Protocol profit from slippage
                step_liquidation_revenue += liquidation_usd - actual_proceeds

                liquidations_this_tick[asset] += liquidation_usd

                # Print slippage info for very large liquidations only
                if liquidation_usd > 5_000_000:
                    print(
                        f"Tick {step}: Full liquidation {asset} ${liquidation_usd:,.0f} with {slippage:.1%} slippage"
                    )

            # Check for bad debt (liquidation value < loan amount)
            if total_liquidation_value < agent.loan:
                bad_debt_amount = agent.loan - total_liquidation_value
                step_bad_debt += bad_debt_amount
                cumulative_bad_debt += bad_debt_amount
            else:
                # Protocol profit from successful liquidation
                liquidation_profit = total_liquidation_value - agent.loan
                step_liquidation_revenue += liquidation_profit

            exchange.liquidate_user(aid)

        cumulative_full += new_full
        cumulative_liquidation_revenue += step_liquidation_revenue

        # Apply price impact from liquidations
        if any(amount > 0 for amount in liquidations_this_tick.values()):
            current_prices = apply_liquidation_price_impact(
                current_prices, liquidations_this_tick, correlation_spillover=0.4
            )
            # Update exchange with impacted prices
            exchange.update_mark_price(current_prices)

        # Health check again after liquidations and price impacts
        exchange.health_check()

        full_liq_over_time.append(cumulative_full)
        partial_liq_over_time.append(cumulative_partial)
        usdc_pool_over_time.append(exchange.usdc_pool)

        # Calculate current collateral value
        current_collateral_value = sum(
            exchange.collateral_pool[asset] * current_prices[asset] for asset in assets
        )

        # Calculate lending income (simplified as interest on loans)
        lending_income_rate = 0.05 / (365 * 24 * 60)  # 5% APY in per-minute rate
        step_lending_income = exchange.total_loans_issued * lending_income_rate

        # Net pool PnL calculation
        pool_change = exchange.usdc_pool - initial_pool_size
        net_pool_pnl = (
            pool_change + cumulative_liquidation_revenue - cumulative_bad_debt
        )

        # Pool utilization
        utilization = (
            exchange.total_loans_issued / exchange.usdc_total
            if exchange.usdc_total > 0
            else 0
        )

        # Effective yield (annualized)
        if initial_pool_size > 0:
            effective_yield = (
                (net_pool_pnl / initial_pool_size) * (365 * 24 * 60 / step)
                if step > 0
                else 0
            )
        else:
            effective_yield = 0

        # Update pool PnL tracking
        pool_pnl_data["step"].append(step)
        pool_pnl_data["usdc_pool"].append(exchange.usdc_pool)
        pool_pnl_data["liquidation_revenue"].append(cumulative_liquidation_revenue)
        pool_pnl_data["lending_income"].append(step_lending_income * step)  # Cumulative
        pool_pnl_data["bad_debt"].append(cumulative_bad_debt)
        pool_pnl_data["total_loans_outstanding"].append(exchange.total_loans_issued)
        pool_pnl_data["collateral_value"].append(current_collateral_value)
        pool_pnl_data["net_pool_pnl"].append(net_pool_pnl)
        pool_pnl_data["pool_utilization"].append(utilization)
        pool_pnl_data["effective_yield"].append(effective_yield)

        # Per category
        for agent in exchange.agents.values():
            if agent.health_status == HealthStatus.FULL_LIQUIDATION:
                liquidation_stats[step][agent.category]["full"] += 1
            elif agent.health_status == HealthStatus.PARTIAL_LIQUIDATION:
                liquidation_stats[step][agent.category]["partial"] += 1

    # Enhanced visualizations with pool PnL
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 15))

    # Create 6 subplots in a 3x2 grid
    ax1 = plt.subplot(3, 2, 1)
    ax2 = plt.subplot(3, 2, 2)
    ax3 = plt.subplot(3, 2, 3)
    ax4 = plt.subplot(3, 2, 4)
    ax5 = plt.subplot(3, 2, 5)
    ax6 = plt.subplot(3, 2, 6)

    steps = list(range(n_steps + 1))

    # Plot 1: Liquidations
    ax1.plot(
        steps,
        full_liq_over_time,
        label="Full Liquidations",
        color="#ff4444",
        linewidth=3,
    )
    ax1.plot(
        steps,
        partial_liq_over_time,
        label="Partial Liquidations",
        color="#ff8800",
        linewidth=3,
    )
    ax1.fill_between(steps, full_liq_over_time, alpha=0.3, color="#ff4444")
    ax1.set_title("Liquidation Cascade", fontsize=14, fontweight="bold", color="white")
    ax1.set_xlabel("Time (minutes)", color="white")
    ax1.set_ylabel("Cumulative Agents Liquidated", color="white")
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.2)

    # Plot 2: Asset Prices (% change)
    for i, asset in enumerate(assets):
        price_path = [initial_prices[asset]] + [
            prices_dict[f"St_{i+1}"][0, s] for s in range(n_steps)
        ]
        # Handle NaN values and calculate percentage change safely
        pct_change = []
        for p in price_path:
            if np.isfinite(p) and initial_prices[asset] > 0:
                pct_change.append((p / initial_prices[asset] - 1) * 100)
            else:
                pct_change.append(
                    0.0 if len(pct_change) == 0 else pct_change[-1]
                )  # Use last valid value

        colors = ["#ff6b35", "#004e89", "#9d4edd", "#e63946"]
        ax2.plot(steps, pct_change, label=asset, color=colors[i], linewidth=3)

    ax2.set_title(
        "Asset Price Performance", fontsize=14, fontweight="bold", color="white"
    )
    ax2.set_xlabel("Time (minutes)", color="white")
    ax2.set_ylabel("Price Change (%)", color="white")
    ax2.axhline(y=0, color="white", linestyle="--", alpha=0.5)
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.2)

    # Plot 3: USDC Pool
    pool_billions = [x / 1e9 for x in usdc_pool_over_time]
    ax3.plot(steps, pool_billions, color="#00d2ff", linewidth=3)
    ax3.fill_between(steps, pool_billions, alpha=0.3, color="#00d2ff")
    ax3.set_title("USDC Pool", fontsize=14, fontweight="bold", color="white")
    ax3.set_xlabel("Time (minutes)", color="white")
    ax3.set_ylabel("Pool Size (Billions $)", color="white")
    ax3.grid(True, alpha=0.2)

    # Plot 4: Liquidation Velocity
    if len(full_liq_over_time) > 5:
        velocity = []
        for i in range(5, len(full_liq_over_time)):
            recent_liq = (
                full_liq_over_time[i]
                + partial_liq_over_time[i]
                - full_liq_over_time[i - 5]
                - partial_liq_over_time[i - 5]
            )
            velocity.append(recent_liq)

        velocity_steps = steps[5:]
        ax4.bar(velocity_steps, velocity, width=1, alpha=0.8, color="#ff073a")
        ax4.set_title(
            "Liquidation Velocity (5-min windows)",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        ax4.set_xlabel("Time (minutes)", color="white")
        ax4.set_ylabel("Liquidations per 5min", color="white")
        ax4.grid(True, alpha=0.2)

    # Plot 5: Pool PnL Breakdown
    ax5.plot(
        pool_pnl_data["step"],
        [x / 1e6 for x in pool_pnl_data["liquidation_revenue"]],
        label="Liquidation Revenue",
        color="#00ff41",
        linewidth=3,
    )
    ax5.plot(
        pool_pnl_data["step"],
        [x / 1e6 for x in pool_pnl_data["bad_debt"]],
        label="Bad Debt",
        color="#ff0000",
        linewidth=3,
    )
    ax5.plot(
        pool_pnl_data["step"],
        [x / 1e6 for x in pool_pnl_data["net_pool_pnl"]],
        label="Net Pool PnL",
        color="#ffff00",
        linewidth=4,
    )
    ax5.set_title("Pool PnL Breakdown", fontsize=14, fontweight="bold", color="white")
    ax5.set_xlabel("Time (minutes)", color="white")
    ax5.set_ylabel("PnL (Millions $)", color="white")
    ax5.legend(frameon=False, fontsize=10)
    ax5.grid(True, alpha=0.2)
    ax5.axhline(y=0, color="white", linestyle="--", alpha=0.5)

    # Plot 6: Pool Utilization & Health
    ax6_twin = ax6.twinx()

    # Pool utilization on left axis
    utilization_pct = [x * 100 for x in pool_pnl_data["pool_utilization"]]
    line1 = ax6.plot(
        pool_pnl_data["step"],
        utilization_pct,
        label="Pool Utilization",
        color="#00d2ff",
        linewidth=3,
    )
    ax6.set_ylabel("Utilization (%)", color="#00d2ff")
    ax6.tick_params(axis="y", labelcolor="#00d2ff")

    # Collateral value on right axis
    collateral_billions = [x / 1e9 for x in pool_pnl_data["collateral_value"]]
    line2 = ax6_twin.plot(
        pool_pnl_data["step"],
        collateral_billions,
        label="Collateral Value",
        color="#ff6b35",
        linewidth=3,
    )
    ax6_twin.set_ylabel("Collateral Value (Billions $)", color="#ff6b35")
    ax6_twin.tick_params(axis="y", labelcolor="#ff6b35")

    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc="upper right", frameon=False, fontsize=10)

    ax6.set_title("Pool Health Metrics", fontsize=14, fontweight="bold", color="white")
    ax6.set_xlabel("Time (minutes)", color="white")
    ax6.grid(True, alpha=0.2)

    # Ensure all plots have proper labels and formatting
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.tick_params(colors="white", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.5)

    # Adjust layout and save
    plt.tight_layout(pad=2.0)

    # Save high-quality plot
    try:
        plt.savefig(
            "simulation_results.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="black",
            edgecolor="none",
        )
        print(f"\nPlot saved as 'simulation_results.png'")
    except Exception as e:
        print(f"\nWarning: Could not save plot - {e}")

    # Show plot
    try:
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display plot - {e}")
        plt.close(fig)

    # Enhanced final summary with pool PnL analysis
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: {scenario_name}")
    print("=" * 60)
    print(
        f"Final Cumulative Liquidations: Full {full_liq_over_time[-1]}, Partial {partial_liq_over_time[-1]}"
    )
    print(f"Final USDC Pool: ${usdc_pool_over_time[-1]:,.0f}")

    # Pool PnL Summary
    print("\n" + "=" * 40)
    print("POOL PnL ANALYSIS")
    print("=" * 40)
    print(f"Initial Pool Size: ${initial_pool_size:,.0f}")
    print(f"Final Pool Size: ${exchange.usdc_pool:,.0f}")
    print(f"Pool Size Change: ${exchange.usdc_pool - initial_pool_size:,.0f}")
    print(f"\nRevenue Breakdown:")
    print(f"  Liquidation Revenue: ${cumulative_liquidation_revenue:,.0f}")
    print(f"  Bad Debt Losses: ${cumulative_bad_debt:,.0f}")
    print(
        f"  Net Liquidation PnL: ${cumulative_liquidation_revenue - cumulative_bad_debt:,.0f}"
    )

    if initial_pool_size > 0:
        pool_return = (exchange.usdc_pool - initial_pool_size) / initial_pool_size * 100
        print("\nPool Performance:")
        print(f"  Total Return: {pool_return:.2f}%")
        print(
            f"  Final Utilization: {exchange.total_loans_issued / exchange.usdc_total * 100:.1f}%"
        )

        # Risk metrics
        bad_debt_ratio = (
            cumulative_bad_debt / exchange.total_loans_issued * 100
            if exchange.total_loans_issued > 0
            else 0
        )
        print(f"  Bad Debt Ratio: {bad_debt_ratio:.2f}%")

        # Calculate maximum drawdown
        peak_pool = max(pool_pnl_data["usdc_pool"])
        max_drawdown = (
            (peak_pool - min(pool_pnl_data["usdc_pool"])) / peak_pool * 100
            if peak_pool > 0
            else 0
        )
        print(f"  Maximum Drawdown: {max_drawdown:.2f}%")

    # Calculate final asset prices and drops
    final_prices = {
        assets[i]: prices_dict[f"St_{i+1}"][0, -1] for i in range(len(assets))
    }

    print("\nAsset Performance:")
    for asset in assets:
        initial_price = initial_prices[asset]
        final_price = final_prices[asset]

        # Handle NaN values in final price
        if not np.isfinite(final_price):
            final_price = 0.0
            drop_pct = 100.0
        else:
            drop_pct = (initial_price - final_price) / initial_price * 100

        print(
            f"  {asset}: ${initial_price:,.0f} → ${final_price:,.0f} ({drop_pct:.1f}% drop)"
        )

    # Liquidation breakdown by category
    print("\nLiquidation Breakdown by Category:")
    category_liq_counts = {"whale": 0, "mid": 0, "pleb": 0, "degen": 0}
    for agent in exchange.agents.values():
        if hasattr(agent, "category") and str(agent.health_status) in [
            "HealthStatus.FULL_LIQUIDATION",
            "HealthStatus.PARTIAL_LIQUIDATION",
        ]:
            if agent.category in category_liq_counts:
                category_liq_counts[agent.category] += 1

    for category, count in category_liq_counts.items():
        print(f"  {category.capitalize()}: {count} liquidated")

    # Time-based analysis for long simulations
    if n_steps >= 1440:  # Full day
        print("\nHourly Liquidation Pattern:")
        for hour in range(0, 24, 4):  # Every 4 hours
            start_tick = hour * 60
            end_tick = min((hour + 4) * 60, n_steps)
            if start_tick < len(full_liq_over_time):
                liq_in_period = (
                    full_liq_over_time[min(end_tick, len(full_liq_over_time) - 1)]
                    - full_liq_over_time[start_tick]
                )
                print(f"  Hours {hour:2d}-{hour+4:2d}: {liq_in_period} liquidations")

    return full_liq_over_time[-1], partial_liq_over_time[-1]


def main():
    print("🚀 MULTI-ASSET BOOST EXCHANGE TICK-BY-TICK STRESS TEST")
    print("=" * 50)

    # Define markets
    assets = ["BTC", "ETH", "SOL", "HYPE"]
    initial_prices = {"BTC": 116000.0, "ETH": 4500.0, "SOL": 240.0, "HYPE": 10.0}
    ltvs = {"BTC": 0.85, "ETH": 0.8, "SOL": 0.7, "HYPE": 0.5}
    max_leverage = {"BTC": 10.0, "ETH": 10.0, "SOL": 20.0, "HYPE": 50.0}

    # Volatilities
    sigma_list = [0.45, 0.55, 0.85, 1.5]  # BTC, ETH, SOL, HYPE

    # Correlation matrix (will be overridden in simulation)
    # corr_matrix = np.array([
    #     [1.0, 0.85, 0.7, 0.4],
    #     [0.85, 1.0, 0.75, 0.3],
    #     [0.7, 0.75, 1.0, 0.5],
    #     [0.4, 0.3, 0.5, 1.0],
    # ])

    # Create agents
    print("\n1. Creating agent population...")
    agents = create_agent_population(1000, assets=assets, initial_prices=initial_prices)

    # Setup exchange
    print("\n2. Setting up BoostExchange with 4 markets...")
    exchange = setup_exchange(agents, assets, initial_prices, ltvs, max_leverage)

    # Run full-day simulation
    print("\n3. Running full-day BLACK SWAN simulation...")

    # DEBUG: Print initial agent states
    print("\nDEBUG: Sample agent states before crash:")
    for aid, agent in list(exchange.agents.items())[:5]:
        collateral_value = sum(
            units * initial_prices[asset]
            for asset, units in agent.collateral_deposited.items()
        )
        print(
            f"Agent {aid} ({getattr(agent, 'category', 'unknown')}): Collateral ${collateral_value:,.0f}, Loan ${getattr(agent, 'loan', 0):,.0f}"
        )


if __name__ == "__main__":
    main()
