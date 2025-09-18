from typing import Dict, List, Optional
from agent import Agent, HealthStatus
import numpy as np


class BoostExchange:
    def __init__(
        self,
        ltvs: Dict[str, float],
        reserve_ratio: float,
        usdc_pool: float,
        max_leverage: Dict[str, float],
        markets: Dict[str, float],
        liquidation_slack: float = 0.03,
        agents: Optional[Dict[int, Agent]] = None,
    ):
        self.agents = agents if agents else {}
        self.ltvs = ltvs
        self.reserve_ratio = reserve_ratio
        self.usdc_pool = usdc_pool
        self.usdc_total = usdc_pool
        self.collateral_deposits = {}
        self.collateral_pool = {asset: 0.0 for asset in markets.keys()}

        self.max_leverage = max_leverage
        self.markets = markets
        self.positions = {}
        self.total_liquidation_ids = []
        self.partial_liquidation_ids = []
        self.total_loans_issued = 0.0
        self.liquidation_slack = liquidation_slack

        # Enhanced pool tracking
        self.liquidation_revenue = 0.0
        self.bad_debt_losses = 0.0
        self.initial_usdc_pool = usdc_pool

    def handle_agent_deposit(self, agent: Agent):
        """Handle agent collateral deposit"""
        self.agents[agent.id] = agent
        for asset, units in agent.collateral_deposited.items():
            if asset not in self.collateral_deposits:
                self.collateral_deposits[asset] = {}
            self.collateral_deposits[asset][agent.id] = units
            self.collateral_pool[asset] += units

    def calculate_available_usdc(self) -> float:
        """Calculate available USDC for borrowing per equation (12)"""
        return self.usdc_total * (1 - self.reserve_ratio) - self.total_loans_issued

    def handle_loan(self, agent_id: int, requested_amount: float):
        """Handle loan request with constraints from equation (11)"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]

        # Calculate max loanable value using asset-specific LTVs
        max_loanable_value = sum(
            units * self.markets[asset] * self.ltvs.get(asset, 0.0)
            for asset, units in agent.collateral_deposited.items()
        )

        # Apply loan constraints from equation (11)
        max_loan_available = self.calculate_available_usdc()
        max_loan = min(max_loanable_value, max_loan_available)

        if requested_amount <= max_loan:
            agent.loan = requested_amount
            self.total_loans_issued += requested_amount
            self.usdc_pool -= requested_amount
            # This USDC goes to agent's trading wallet
            agent.twm = requested_amount
        else:
            raise Exception(f"Cannot lend {requested_amount}. Max loan: {max_loan}")

    def update_mark_price(self, new_prices: Dict[str, float]):
        """Update mark prices and trigger health checks"""
        self.markets.update(new_prices)

        # Clear liquidation lists
        self.total_liquidation_ids.clear()
        self.partial_liquidation_ids.clear()

        # Run health check for all agents
        self.health_check()

    def calculate_collateral_value(self, agent: Agent) -> float:
        """Calculate C = Sz_c * P_t from equation (10)"""
        total = 0.0
        for asset, units in agent.collateral_deposited.items():
            mark_price = self.markets.get(asset, 0)
            total += units * mark_price
        return total

    def calculate_twe(self, agent: Agent) -> float:
        """Calculate TWE = Sz * ((Px_t - Px_0) * D) + TWM from equation (22)"""
        if not hasattr(agent, "twm"):
            agent.twm = 0.0

        twe = agent.twm  # Trading Wallet Margin

        if not agent.positions:
            return twe

        # Calculate PnL for all positions
        for asset, position in agent.positions.items():
            size = position.get("size", 0)
            entry_price = position.get("entry_price", 0)
            direction = position.get("direction", 1)  # 1 for long, -1 for short
            mark_price = self.markets.get(asset, entry_price)

            # PnL = Size * (Current - Entry) * Direction
            pnl = size * (mark_price - entry_price) * direction
            twe += pnl

        return twe

    def calculate_mmr(self, agent: Agent) -> float:
        """Calculate MMR = Px_t * (Sz / (2 * l_max)) from equation (23)"""
        if not agent.positions:
            return 0.0

        mmr = 0.0

        for asset, position in agent.positions.items():
            size = abs(position.get("size", 0))
            mark_price = self.markets.get(asset, 0)
            leverage = self.max_leverage.get(asset, 10)

            # MMR for this position
            mmr += mark_price * size / (2 * leverage)

        return mmr + agent.loan * self.liquidation_slack

    def calculate_weighted_collateral_value(self, agent: Agent) -> float:
        """Calculate C * LTV_max using asset-specific LTVs"""
        weighted_value = 0.0
        for asset, units in agent.collateral_deposited.items():
            mark_price = self.markets.get(asset, 0)
            ltv = self.ltvs.get(asset, 0.0)
            weighted_value += units * mark_price * ltv
        return weighted_value

    def calculate_bae(self, agent: Agent) -> float:
        """Calculate BAE = C * LTV_max + TWE - L from equation (21)"""
        weighted_collateral = self.calculate_weighted_collateral_value(agent)
        twe = self.calculate_twe(agent)
        bae = weighted_collateral + twe - agent.loan
        return bae

    def health_check(self):
        """Check health of all agents and update their status"""
        for agent_id, agent in self.agents.items():
            # Calculate all metrics
            c = self.calculate_collateral_value(agent)
            twe = self.calculate_twe(agent)
            mmr = self.calculate_mmr(agent)
            bae = self.calculate_bae(agent)

            # Update agent state
            agent.collateral_value = c
            agent.twe = twe
            agent.mmr = mmr
            agent.bae = bae

            # Check liquidation conditions from section 2.7
            # Solvency value uses asset-specific LTVs
            weighted_collateral = self.calculate_weighted_collateral_value(agent)
            solvency_value = weighted_collateral + twe

            # Full liquidation condition: C * LTV_max + TWE <= L
            if solvency_value <= agent.loan:
                agent.health_status = HealthStatus.FULL_LIQUIDATION
                if agent_id not in self.total_liquidation_ids:
                    self.total_liquidation_ids.append(agent_id)

            # Partial liquidation: BAE <= MMR but still solvent
            elif bae <= mmr and solvency_value > agent.loan:
                agent.health_status = HealthStatus.PARTIAL_LIQUIDATION
                if agent_id not in self.partial_liquidation_ids:
                    self.partial_liquidation_ids.append(agent_id)

            # Healthy: BAE > MMR
            else:
                agent.health_status = HealthStatus.HEALTHY

    def partial_liquidate_user(self, agent_id: int, reduction_factor: float = 0.25):
        """
        Partially liquidate positions to bring BAE back above MMR.
        Liquidation proceeds go back to the USDC pool.
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]

        if agent.health_status != HealthStatus.PARTIAL_LIQUIDATION:
            return

        liquidation_proceeds = 0.0

        # Reduce positions by reduction_factor
        for asset, position in list(agent.positions.items()):
            current_size = position.get("size", 0)
            reduced_size = current_size * (1 - reduction_factor)

            # Calculate liquidation proceeds (goes to protocol)
            mark_price = self.markets.get(asset, 0)
            liquidated_notional = abs(current_size * reduction_factor) * mark_price
            liquidation_proceeds += liquidated_notional

            # Protocol captures the liquidated value
            self.usdc_pool += liquidated_notional

            # Update or remove position
            if abs(reduced_size) < 0.001:
                del agent.positions[asset]
            else:
                position["size"] = reduced_size

        # Track liquidation revenue
        self.liquidation_revenue += liquidation_proceeds

        # Re-check health after partial liquidation
        self.health_check()

    def liquidate_user(self, agent_id: int):
        """
        Full liquidation - all positions closed, collateral seized by protocol.
        All proceeds go back to the USDC pool to ensure protocol solvency.
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]

        total_liquidation_value = 0.0

        # Liquidate all positions
        for asset, position in agent.positions.items():
            size = position.get("size", 0)
            mark_price = self.markets.get(asset, 0)
            total_liquidation_value += abs(size) * mark_price

        # Seize all collateral
        for asset, units in agent.collateral_deposited.items():
            mark_price = self.markets.get(asset, 0)
            total_liquidation_value += units * mark_price

            # Remove from collateral pool
            self.collateral_pool[asset] -= units
            if (
                asset in self.collateral_deposits
                and agent_id in self.collateral_deposits[asset]
            ):
                del self.collateral_deposits[asset][agent_id]

        # Repay loan first
        debt_repayment = min(agent.loan, total_liquidation_value)
        self.usdc_pool += debt_repayment
        self.total_loans_issued -= agent.loan

        # Track bad debt or liquidation profit
        if total_liquidation_value < agent.loan:
            # Bad debt - protocol takes a loss
            bad_debt = agent.loan - total_liquidation_value
            self.bad_debt_losses += bad_debt
        else:
            # Liquidation profit - protocol keeps excess
            protocol_profit = total_liquidation_value - agent.loan
            self.usdc_pool += protocol_profit
            self.liquidation_revenue += protocol_profit

        # Clear agent's state
        agent.positions.clear()
        agent.collateral_deposited.clear()
        agent.loan = 0.0
        agent.twm = 0.0
        agent.health_status = HealthStatus.FULL_LIQUIDATION
        agent.bae = 0.0
        agent.twe = 0.0
        agent.mmr = 0.0

    def get_pool_pnl(self) -> Dict[str, float]:
        """Calculate comprehensive pool PnL metrics"""
        pool_change = self.usdc_pool - self.initial_usdc_pool
        net_liquidation_pnl = self.liquidation_revenue - self.bad_debt_losses

        return {
            "pool_change": pool_change,
            "liquidation_revenue": self.liquidation_revenue,
            "bad_debt_losses": self.bad_debt_losses,
            "net_liquidation_pnl": net_liquidation_pnl,
            "utilization_ratio": (
                self.total_loans_issued / self.usdc_total if self.usdc_total > 0 else 0
            ),
            "loans_outstanding": self.total_loans_issued,
        }
