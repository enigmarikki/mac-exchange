from typing import Dict, Optional
from enum import Enum
import numpy as np


class HealthStatus(Enum):
    HEALTHY = "healthy"
    PARTIAL_LIQUIDATION = "partial_liquidation"
    FULL_LIQUIDATION = "full_liquidation"


class LiquidityProvider:
    # they can withdraw/deposit

    def __init__(self, agent_id: int, usdc_deposits: float):
        self.id = agent_id
        self.usdc_depositied = usdc_deposits


class TraderAgent:
    def __init__(self, agent_id: int):
        self.id = agent_id
        self.collateral_deposited: Dict[str, float] = {}
        self.loan: float = 0.0
        self.positions: Dict[str, Dict] = {}
        self.twm: float = 0.0  # Trading Wallet Margin (USDC in trading wallet)
        # Calculated values
        self.collateral_value: float = 0.0  # C value
        self.twe: float = 0.0  # Trading Wallet Equity
        self.bae: float = 0.0  # Boost Account Equity
        self.mmr: float = 0.0  # Maintenance Margin Requirement
        self.health_status: HealthStatus = HealthStatus.HEALTHY

        # Track initial values for PnL calculation
        self.initial_collateral_value: float = 0.0
        self.realized_pnl: float = 0.0
        self.category = None
        self.ltv_ratio = 0.0
        self.leverage_ratio = 1.0
