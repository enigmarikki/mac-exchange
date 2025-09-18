use std::collections::HashMap;

use crate::agents::Agent;
use crate::exchange_types::{AssetMetadata, LiquidationAction, Pool};

pub const USDC_POOL: [u8; 32] = [0xAA; 32];

#[derive(Debug, Clone)]
pub struct Position {
    pub size: f64,
    pub entry_price: f64,
    pub max_leverage: f64,
}

#[derive(Debug, Clone)]
pub struct CollateralPosition {
    pub units: f64,
    pub entry_price: f64,
    pub current_price: f64,
}

pub struct BoostExchange {
    pub liquidity_pool: HashMap<[u8; 32], Pool>,
    agents: HashMap<[u8; 32], Agent>,
    liquidation_slack: f64,
    active_loans: HashMap<[u8; 32], f64>,
    master_key: [u8; 32],
}

impl BoostExchange {
    pub fn new(master_key: [u8; 32]) -> Self {
        let mut exchange = BoostExchange {
            liquidity_pool: HashMap::new(),
            agents: HashMap::new(),
            liquidation_slack: 0.05,
            active_loans: HashMap::new(),
            master_key,
        };

        // Initialize USDC pool
        let usdc_metadata = AssetMetadata::new(USDC_POOL);
        let usdc_pool = Pool::new(usdc_metadata, 1.0, 10_000_000.0);
        exchange.liquidity_pool.insert(USDC_POOL, usdc_pool);

        exchange
    }

    pub fn calculate_collateral_value(&self, collateral_positions: &HashMap<[u8; 32], CollateralPosition>) -> f64 {
        collateral_positions.values().map(|pos| pos.units * pos.current_price)
            .sum()
    }

    pub fn calculate_available_usdc(&self) -> f64 {
        let total_loans: f64 = self.active_loans.values().sum();
        if let Some(usdc_pool) = self.liquidity_pool.get(&USDC_POOL) {
            usdc_pool.get_available_liquidity() - total_loans
        } else {
            0.0
        }
    }

    pub fn calculate_max_loan(&self, collateral_value: f64, collateral_asset_id: [u8; 32]) -> f64 {
        let collateral_limit = if let Some(collateral_pool) = self.liquidity_pool.get(&collateral_asset_id) {
            collateral_value * collateral_pool.get_ltv_max()
        } else {
            0.0
        };
        let liquidity_limit = self.calculate_available_usdc();
        collateral_limit.min(liquidity_limit)
    }

    pub fn calculate_trading_wallet_equity(&self, positions: &HashMap<[u8; 32], Position>, mark_prices: &HashMap<[u8; 32], f64>, trading_wallet_margin: f64) -> f64 {
        let position_pnl: f64 = positions.iter()
            .map(|(asset_id, pos)| {
                let current_price = mark_prices.get(asset_id).unwrap_or(&pos.entry_price);
                pos.size * (current_price - pos.entry_price)
            })
            .sum();

        position_pnl + trading_wallet_margin
    }

    pub fn calculate_maintenance_margin_requirement(&self, positions: &HashMap<[u8; 32], Position>, mark_prices: &HashMap<[u8; 32], f64>, loan: f64) -> f64 {
        let position_mmr: f64 = positions.iter()
            .map(|(asset_id, pos)| {
                let current_price = mark_prices.get(asset_id).unwrap_or(&pos.entry_price);
                current_price * pos.size.abs() / (2.0 * pos.max_leverage)
            })
            .sum();

        position_mmr + self.liquidation_slack * loan
    }

    pub fn calculate_boost_account_equity(&self, collateral_positions: &HashMap<[u8; 32], CollateralPosition>, trading_wallet_equity: f64, loan: f64) -> f64 {
        let weighted_collateral_value = collateral_positions.iter()
            .map(|(asset_id, pos)| {
                if let Some(pool) = self.liquidity_pool.get(asset_id) {
                    pos.units * pos.current_price * pool.get_ltv_max()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        weighted_collateral_value + trading_wallet_equity - loan
    }

    pub fn is_healthy(&self, boost_account_equity: f64, maintenance_margin_requirement: f64) -> bool {
        boost_account_equity > maintenance_margin_requirement
    }

    pub fn should_partial_liquidate(&self, boost_account_equity: f64, maintenance_margin_requirement: f64, collateral_positions: &HashMap<[u8; 32], CollateralPosition>, trading_wallet_equity: f64, loan: f64) -> bool {
        let weighted_collateral_value = collateral_positions.iter()
            .map(|(asset_id, pos)| {
                if let Some(pool) = self.liquidity_pool.get(asset_id) {
                    pos.units * pos.current_price * pool.get_ltv_max()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        boost_account_equity <= maintenance_margin_requirement &&
        weighted_collateral_value + trading_wallet_equity > loan * (1.0 + self.liquidation_slack)
    }

    pub fn should_full_liquidate(&self, collateral_positions: &HashMap<[u8; 32], CollateralPosition>, trading_wallet_equity: f64, loan: f64) -> bool {
        let weighted_collateral_value = collateral_positions.iter()
            .map(|(asset_id, pos)| {
                if let Some(pool) = self.liquidity_pool.get(asset_id) {
                    pos.units * pos.current_price * pool.get_ltv_max()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        weighted_collateral_value + trading_wallet_equity <= loan * (1.0 + self.liquidation_slack)
    }

    pub fn add_pool(&mut self, caller_key: [u8; 32], pool_id: [u8; 32], pool: Pool) -> Result<(), String> {
        if caller_key != self.master_key {
            return Err("Only master can add pools".to_string());
        }
        self.liquidity_pool.insert(pool_id, pool);
        Ok(())
    }

    pub fn add_agent(&mut self, agent: Agent) {
        self.agents.insert(agent.get_id(), agent);
    }

    pub fn register_loan(&mut self, agent_id: [u8; 32], loan_amount: f64) -> Result<(), String> {
        if loan_amount > self.calculate_available_usdc() {
            return Err("Insufficient liquidity in USDC pool".to_string());
        }

        // Update existing loan or create new one
        let current_loan = self.active_loans.get(&agent_id).copied().unwrap_or(0.0);
        self.active_loans.insert(agent_id, current_loan + loan_amount);
        Ok(())
    }

    pub fn accrue_interest(&mut self, interest_rate: f64) {
        // Apply interest to all active loans (interest_rate should be per-tick rate)
        for loan_amount in self.active_loans.values_mut() {
            *loan_amount *= 1.0 + interest_rate;
        }
    }

    pub fn get_total_loans_outstanding(&self) -> f64 {
        self.active_loans.values().sum()
    }

    pub fn repay_loan(&mut self, agent_id: [u8; 32], repay_amount: f64) -> Result<(), String> {
        if let Some(current_loan) = self.active_loans.get_mut(&agent_id) {
            if repay_amount > *current_loan {
                return Err("Repay amount exceeds outstanding loan".to_string());
            }

            *current_loan -= repay_amount;

            if *current_loan <= 0.0 {
                self.active_loans.remove(&agent_id);
            }

            Ok(())
        } else {
            Err("No active loan found for agent".to_string())
        }
    }

    pub fn get_agent_loan(&self, agent_id: &[u8; 32]) -> f64 {
        self.active_loans.get(agent_id).copied().unwrap_or(0.0)
    }

    pub fn update_prices(&mut self, price_updates: HashMap<[u8; 32], f64>) {
        for (asset_id, new_price) in price_updates {
            if let Some(pool) = self.liquidity_pool.get_mut(&asset_id) {
                pool.asset_info.oracle_price = Some(new_price);
            }
        }
    }

    pub fn monitor_all_agents(&self, all_positions: &HashMap<[u8; 32], HashMap<[u8; 32], Position>>, all_collateral: &HashMap<[u8; 32], HashMap<[u8; 32], CollateralPosition>>, all_twm: &HashMap<[u8; 32], f64>, mark_prices: &HashMap<[u8; 32], f64>) -> Vec<([u8; 32], LiquidationAction)> {
        let mut liquidation_events = Vec::new();
        let empty_collateral = HashMap::new();

        for (agent_id, positions) in all_positions {
            let collateral_positions = all_collateral.get(agent_id).unwrap_or(&empty_collateral);
            let trading_wallet_margin = all_twm.get(agent_id).copied().unwrap_or(0.0);
            let loan = self.get_agent_loan(agent_id);

            let collateral_value = self.calculate_collateral_value(collateral_positions);
            let trading_wallet_equity = self.calculate_trading_wallet_equity(positions, mark_prices, trading_wallet_margin);
            let maintenance_margin_requirement = self.calculate_maintenance_margin_requirement(positions, mark_prices, loan);
            let boost_account_equity = self.calculate_boost_account_equity(collateral_positions, trading_wallet_equity, loan);

            if self.should_full_liquidate(collateral_positions, trading_wallet_equity, loan) {
                liquidation_events.push((*agent_id, LiquidationAction::FullLiquidation));
            } else if self.should_partial_liquidate(boost_account_equity, maintenance_margin_requirement, collateral_positions, trading_wallet_equity, loan) {
                liquidation_events.push((*agent_id, LiquidationAction::PartialLiquidation));
            }
        }

        liquidation_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exchange_types::Collateral;

    fn create_test_exchange() -> BoostExchange {
        let mut exchange = BoostExchange::new([0xAA; 32]);

        // Add a test collateral pool with ltv_max 0.8
        let test_asset = AssetMetadata::new([0x01; 32]);
        let test_pool = Pool::new(test_asset, 0.8, 1_000_000.0);
        exchange.add_pool([0xAA; 32], [0x01; 32], test_pool).unwrap();

        // Seed USDC pool with funds
        if let Some(usdc_pool) = exchange.liquidity_pool.get_mut(&USDC_POOL) {
            usdc_pool.add_collateral([0xFF; 32], 1_000_000.0).unwrap();
        }

        exchange
    }

    fn create_test_agent() -> Agent {
        let collateral = vec![
            Collateral { asset_id: [0x01; 32], units: 100.0 },
            Collateral { asset_id: [0x02; 32], units: 50.0 },
        ];
        Agent::new(collateral)
    }

    #[test]
    fn test_new_exchange_creation() {
        let exchange = create_test_exchange();
        assert_eq!(exchange.liquidation_slack, 0.05);
        assert!(exchange.liquidity_pool.contains_key(&USDC_POOL));
        assert!(exchange.agents.is_empty());
        assert!(exchange.active_loans.is_empty());
        assert_eq!(exchange.master_key, [0xAA; 32]);
    }

    #[test]
    fn test_calculate_collateral_value() {
        let exchange = create_test_exchange();
        let mut collateral_positions = HashMap::new();

        collateral_positions.insert([0x01; 32], CollateralPosition {
            units: 100.0,
            entry_price: 50.0,
            current_price: 60.0,
        });

        collateral_positions.insert([0x02; 32], CollateralPosition {
            units: 50.0,
            entry_price: 100.0,
            current_price: 110.0,
        });

        let total_value = exchange.calculate_collateral_value(&collateral_positions);
        assert_eq!(total_value, 100.0 * 60.0 + 50.0 * 110.0);
        assert_eq!(total_value, 11_500.0);
    }

    #[test]
    fn test_calculate_collateral_value_empty() {
        let exchange = create_test_exchange();
        let collateral_positions = HashMap::new();
        let total_value = exchange.calculate_collateral_value(&collateral_positions);
        assert_eq!(total_value, 0.0);
    }

    #[test]
    fn test_calculate_available_usdc_no_loans() {
        let exchange = create_test_exchange();
        let available = exchange.calculate_available_usdc();
        // USDC pool has 1M units, 0.2 reserve ratio = 800K available
        assert_eq!(available, 800_000.0);
    }

    #[test]
    fn test_calculate_available_usdc_with_loans() {
        let mut exchange = create_test_exchange();
        exchange.active_loans.insert([0x01; 32], 100_000.0);
        exchange.active_loans.insert([0x02; 32], 50_000.0);

        let available = exchange.calculate_available_usdc();
        assert_eq!(available, 800_000.0 - 150_000.0);
        assert_eq!(available, 650_000.0);
    }

    #[test]
    fn test_calculate_max_loan_collateral_limited() {
        let exchange = create_test_exchange();
        let collateral_value = 1000.0;
        let max_loan = exchange.calculate_max_loan(collateral_value, [0x01; 32]);
        assert_eq!(max_loan, collateral_value * 0.8);
        assert_eq!(max_loan, 800.0);
    }

    #[test]
    fn test_calculate_max_loan_liquidity_limited() {
        let mut exchange = create_test_exchange();
        exchange.active_loans.insert([0x01; 32], 790_000.0);

        let collateral_value = 1_000_000.0;
        let max_loan = exchange.calculate_max_loan(collateral_value, [0x01; 32]);
        assert_eq!(max_loan, 10_000.0);
    }

    #[test]
    fn test_calculate_trading_wallet_equity() {
        let exchange = create_test_exchange();
        let mut positions = HashMap::new();
        let mut mark_prices = HashMap::new();

        positions.insert([0x01; 32], Position {
            size: 10.0,
            entry_price: 100.0,
            max_leverage: 5.0,
        });

        positions.insert([0x02; 32], Position {
            size: -5.0,
            entry_price: 200.0,
            max_leverage: 3.0,
        });

        mark_prices.insert([0x01; 32], 110.0);
        mark_prices.insert([0x02; 32], 190.0);

        let trading_wallet_margin = 1000.0;
        let equity = exchange.calculate_trading_wallet_equity(&positions, &mark_prices, trading_wallet_margin);

        let expected_pnl = 10.0 * (110.0 - 100.0) + (-5.0) * (190.0 - 200.0);
        assert_eq!(equity, expected_pnl + trading_wallet_margin);
        assert_eq!(equity, 100.0 + 50.0 + 1000.0);
        assert_eq!(equity, 1150.0);
    }

    #[test]
    fn test_calculate_trading_wallet_equity_no_mark_prices() {
        let exchange = create_test_exchange();
        let mut positions = HashMap::new();
        let mark_prices = HashMap::new();

        positions.insert([0x01; 32], Position {
            size: 10.0,
            entry_price: 100.0,
            max_leverage: 5.0,
        });

        let trading_wallet_margin = 1000.0;
        let equity = exchange.calculate_trading_wallet_equity(&positions, &mark_prices, trading_wallet_margin);

        assert_eq!(equity, trading_wallet_margin);
        assert_eq!(equity, 1000.0);
    }

    #[test]
    fn test_calculate_maintenance_margin_requirement() {
        let exchange = create_test_exchange();
        let mut positions = HashMap::new();
        let mut mark_prices = HashMap::new();

        positions.insert([0x01; 32], Position {
            size: 10.0,
            entry_price: 100.0,
            max_leverage: 5.0,
        });

        positions.insert([0x02; 32], Position {
            size: -8.0,
            entry_price: 200.0,
            max_leverage: 4.0,
        });

        mark_prices.insert([0x01; 32], 110.0);
        mark_prices.insert([0x02; 32], 190.0);

        let loan = 50_000.0;
        let mmr = exchange.calculate_maintenance_margin_requirement(&positions, &mark_prices, loan);

        let position1_mmr = 110.0 * 10.0 / (2.0 * 5.0);
        let position2_mmr = 190.0 * 8.0 / (2.0 * 4.0);
        let loan_slack = 0.05 * loan;

        assert_eq!(mmr, position1_mmr + position2_mmr + loan_slack);
        assert_eq!(mmr, 110.0 + 190.0 + 2500.0);
        assert_eq!(mmr, 2800.0);
    }

    #[test]
    fn test_calculate_boost_account_equity() {
        let exchange = create_test_exchange();
        let mut collateral_positions = HashMap::new();
        collateral_positions.insert([0x01; 32], CollateralPosition {
            units: 100.0,
            entry_price: 100.0,
            current_price: 100.0,
        });
        let trading_wallet_equity = 5_000.0;
        let loan = 8_000.0;

        let equity = exchange.calculate_boost_account_equity(&collateral_positions, trading_wallet_equity, loan);
        // 100 units * 100 price * 0.8 ltv = 8000 + 5000 - 8000 = 5000
        assert_eq!(equity, 5_000.0);
    }

    #[test]
    fn test_is_healthy_true() {
        let exchange = create_test_exchange();
        assert!(exchange.is_healthy(10_000.0, 5_000.0));
    }

    #[test]
    fn test_is_healthy_false() {
        let exchange = create_test_exchange();
        assert!(!exchange.is_healthy(5_000.0, 10_000.0));
    }

    #[test]
    fn test_is_healthy_equal() {
        let exchange = create_test_exchange();
        assert!(!exchange.is_healthy(5_000.0, 5_000.0));
    }

    #[test]
    fn test_should_partial_liquidate_true() {
        let exchange = create_test_exchange();
        let boost_account_equity = 4_000.0;
        let maintenance_margin_requirement = 5_000.0;
        let mut collateral_positions = HashMap::new();
        collateral_positions.insert([0x01; 32], CollateralPosition {
            units: 125.0,
            entry_price: 80.0,
            current_price: 80.0,
        });
        let trading_wallet_equity = 3_000.0;
        let loan = 8_000.0;

        assert!(exchange.should_partial_liquidate(
            boost_account_equity,
            maintenance_margin_requirement,
            &collateral_positions,
            trading_wallet_equity,
            loan
        ));
    }

    #[test]
    fn test_should_partial_liquidate_false_healthy() {
        let exchange = create_test_exchange();
        let boost_account_equity = 6_000.0;
        let maintenance_margin_requirement = 5_000.0;
        let mut collateral_positions = HashMap::new();
        collateral_positions.insert([0x01; 32], CollateralPosition {
            units: 125.0,
            entry_price: 80.0,
            current_price: 80.0,
        });
        let trading_wallet_equity = 3_000.0;
        let loan = 8_000.0;

        assert!(!exchange.should_partial_liquidate(
            boost_account_equity,
            maintenance_margin_requirement,
            &collateral_positions,
            trading_wallet_equity,
            loan
        ));
    }

    #[test]
    fn test_should_partial_liquidate_false_insufficient_collateral() {
        let exchange = create_test_exchange();
        let boost_account_equity = 4_000.0;
        let maintenance_margin_requirement = 5_000.0;
        let mut collateral_positions = HashMap::new();
        collateral_positions.insert([0x01; 32], CollateralPosition {
            units: 62.5,
            entry_price: 80.0,
            current_price: 80.0,
        });
        let trading_wallet_equity = 1_000.0;
        let loan = 8_000.0;

        assert!(!exchange.should_partial_liquidate(
            boost_account_equity,
            maintenance_margin_requirement,
            &collateral_positions,
            trading_wallet_equity,
            loan
        ));
    }

    #[test]
    fn test_should_full_liquidate_true() {
        let exchange = create_test_exchange();
        let mut collateral_positions = HashMap::new();
        collateral_positions.insert([0x01; 32], CollateralPosition {
            units: 62.5,
            entry_price: 80.0,
            current_price: 80.0,
        });
        let trading_wallet_equity = 1_000.0;
        let loan = 8_000.0;

        assert!(exchange.should_full_liquidate(&collateral_positions, trading_wallet_equity, loan));
    }

    #[test]
    fn test_should_full_liquidate_false() {
        let exchange = create_test_exchange();
        let mut collateral_positions = HashMap::new();
        collateral_positions.insert([0x01; 32], CollateralPosition {
            units: 125.0,
            entry_price: 80.0,
            current_price: 80.0,
        });
        let trading_wallet_equity = 3_000.0;
        let loan = 8_000.0;

        assert!(!exchange.should_full_liquidate(&collateral_positions, trading_wallet_equity, loan));
    }

    #[test]
    fn test_add_pool_success() {
        let mut exchange = create_test_exchange();
        let test_asset = AssetMetadata::new([0x99; 32]);
        let test_pool = Pool::new(test_asset, 0.75, 500_000.0);

        let result = exchange.add_pool([0xAA; 32], [0x99; 32], test_pool);
        assert!(result.is_ok());
        assert!(exchange.liquidity_pool.contains_key(&[0x99; 32]));
    }

    #[test]
    fn test_add_pool_unauthorized() {
        let mut exchange = create_test_exchange();
        let test_asset = AssetMetadata::new([0x99; 32]);
        let test_pool = Pool::new(test_asset, 0.75, 500_000.0);

        let result = exchange.add_pool([0xBB; 32], [0x99; 32], test_pool);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Only master can add pools");
        assert!(!exchange.liquidity_pool.contains_key(&[0x99; 32]));
    }

    #[test]
    fn test_add_agent() {
        let mut exchange = create_test_exchange();
        let agent = create_test_agent();
        let agent_id = agent.get_id();

        exchange.add_agent(agent);
        assert!(exchange.agents.contains_key(&agent_id));
    }

    #[test]
    fn test_register_loan_success() {
        let mut exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        let loan_amount = 100_000.0;

        let result = exchange.register_loan(agent_id, loan_amount);
        assert!(result.is_ok());
        assert_eq!(exchange.get_agent_loan(&agent_id), loan_amount);
    }

    #[test]
    fn test_register_loan_insufficient_liquidity() {
        let mut exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        let loan_amount = 900_000.0;

        let result = exchange.register_loan(agent_id, loan_amount);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Insufficient liquidity in USDC pool");
        assert_eq!(exchange.get_agent_loan(&agent_id), 0.0);
    }

    #[test]
    fn test_repay_loan_partial() {
        let mut exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        let loan_amount = 100_000.0;
        let repay_amount = 30_000.0;

        exchange.register_loan(agent_id, loan_amount).unwrap();
        let result = exchange.repay_loan(agent_id, repay_amount);

        assert!(result.is_ok());
        assert_eq!(exchange.get_agent_loan(&agent_id), 70_000.0);
    }

    #[test]
    fn test_repay_loan_full() {
        let mut exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        let loan_amount = 100_000.0;

        exchange.register_loan(agent_id, loan_amount).unwrap();
        let result = exchange.repay_loan(agent_id, loan_amount);

        assert!(result.is_ok());
        assert_eq!(exchange.get_agent_loan(&agent_id), 0.0);
        assert!(!exchange.active_loans.contains_key(&agent_id));
    }

    #[test]
    fn test_repay_loan_excess_amount() {
        let mut exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        let loan_amount = 100_000.0;
        let repay_amount = 150_000.0;

        exchange.register_loan(agent_id, loan_amount).unwrap();
        let result = exchange.repay_loan(agent_id, repay_amount);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Repay amount exceeds outstanding loan");
        assert_eq!(exchange.get_agent_loan(&agent_id), loan_amount);
    }

    #[test]
    fn test_repay_loan_no_active_loan() {
        let mut exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        let repay_amount = 50_000.0;

        let result = exchange.repay_loan(agent_id, repay_amount);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "No active loan found for agent");
    }

    #[test]
    fn test_get_agent_loan_exists() {
        let mut exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        let loan_amount = 100_000.0;

        exchange.register_loan(agent_id, loan_amount).unwrap();
        assert_eq!(exchange.get_agent_loan(&agent_id), loan_amount);
    }

    #[test]
    fn test_get_agent_loan_not_exists() {
        let exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        assert_eq!(exchange.get_agent_loan(&agent_id), 0.0);
    }

    #[test]
    fn test_update_prices() {
        let mut exchange = create_test_exchange();
        let mut price_updates = HashMap::new();
        price_updates.insert(USDC_POOL, 1.01);

        exchange.update_prices(price_updates);

        let usdc_pool = exchange.liquidity_pool.get(&USDC_POOL).unwrap();
        assert_eq!(usdc_pool.asset_info.oracle_price, Some(1.01));
    }

    #[test]
    fn test_monitor_all_agents_no_liquidation() {
        let exchange = create_test_exchange();
        let mut all_positions = HashMap::new();
        let mut all_collateral = HashMap::new();
        let mut all_twm = HashMap::new();
        let mut mark_prices = HashMap::new();

        let agent_id = [0x01; 32];

        let mut positions = HashMap::new();
        positions.insert([0x02; 32], Position {
            size: 1.0,
            entry_price: 100.0,
            max_leverage: 10.0,
        });
        all_positions.insert(agent_id, positions);

        let mut collateral = HashMap::new();
        collateral.insert([0x03; 32], CollateralPosition {
            units: 1000.0,
            entry_price: 100.0,
            current_price: 100.0,
        });
        all_collateral.insert(agent_id, collateral);

        all_twm.insert(agent_id, 10000.0);
        mark_prices.insert([0x02; 32], 105.0);

        let liquidations = exchange.monitor_all_agents(&all_positions, &all_collateral, &all_twm, &mark_prices);
        assert!(liquidations.is_empty());
    }

    #[test]
    fn test_monitor_all_agents_partial_liquidation() {
        let mut exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        exchange.register_loan(agent_id, 50000.0).unwrap();

        let mut all_positions = HashMap::new();
        let mut all_collateral = HashMap::new();
        let mut all_twm = HashMap::new();
        let mut mark_prices = HashMap::new();

        let mut positions = HashMap::new();
        positions.insert([0x02; 32], Position {
            size: 100.0,
            entry_price: 100.0,
            max_leverage: 2.0,
        });
        all_positions.insert(agent_id, positions);

        let mut collateral = HashMap::new();
        collateral.insert([0x03; 32], CollateralPosition {
            units: 500.0,
            entry_price: 100.0,
            current_price: 100.0,
        });
        all_collateral.insert(agent_id, collateral);

        all_twm.insert(agent_id, -10000.0);
        mark_prices.insert([0x02; 32], 80.0);

        let liquidations = exchange.monitor_all_agents(&all_positions, &all_collateral, &all_twm, &mark_prices);
        assert_eq!(liquidations.len(), 1);
        assert_eq!(liquidations[0].0, agent_id);
        assert!(matches!(liquidations[0].1, LiquidationAction::FullLiquidation));
    }

    #[test]
    fn test_monitor_all_agents_full_liquidation() {
        let mut exchange = create_test_exchange();
        let agent_id = [0x01; 32];
        exchange.register_loan(agent_id, 50000.0).unwrap();

        let mut all_positions = HashMap::new();
        let mut all_collateral = HashMap::new();
        let mut all_twm = HashMap::new();
        let mut mark_prices = HashMap::new();

        let mut positions = HashMap::new();
        positions.insert([0x02; 32], Position {
            size: 100.0,
            entry_price: 100.0,
            max_leverage: 2.0,
        });
        all_positions.insert(agent_id, positions);

        let mut collateral = HashMap::new();
        collateral.insert([0x03; 32], CollateralPosition {
            units: 100.0,
            entry_price: 100.0,
            current_price: 100.0,
        });
        all_collateral.insert(agent_id, collateral);

        all_twm.insert(agent_id, -40000.0);
        mark_prices.insert([0x02; 32], 50.0);

        let liquidations = exchange.monitor_all_agents(&all_positions, &all_collateral, &all_twm, &mark_prices);
        assert_eq!(liquidations.len(), 1);
        assert_eq!(liquidations[0].0, agent_id);
        assert!(matches!(liquidations[0].1, LiquidationAction::FullLiquidation));
    }
}
