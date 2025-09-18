use std::collections::HashMap;

pub struct Collateral {
    pub asset_id: [u8; 32],
    pub units: f64,
}

pub struct AssetMetadata {
    pub asset_id: [u8; 32],
    pub oracle_price: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum LiquidationAction {
    PartialLiquidation,
    FullLiquidation,
}

pub struct Pool {
    collateral_positions: HashMap<[u8; 32], f64>,
    total_units: f64,
    reserve_ratio: f64,
    units_borrowed: f64,
    ltv_max: f64,
    deposit_cap: f64,
    pub asset_info: AssetMetadata,
}

//Do asset shit here
impl AssetMetadata {
    pub fn new(id: [u8; 32]) -> Self {
        AssetMetadata {
            asset_id: id,
            oracle_price: None,
        }
    }
}

//Do pool shit here
impl Pool {
    pub fn new(asset_info: AssetMetadata, ltv_max: f64, cap: f64) -> Self {
        Pool {
            collateral_positions: HashMap::new(),
            total_units: 0.0,
            reserve_ratio: 0.2,
            units_borrowed: 0.0,
            asset_info,
            ltv_max,
            deposit_cap: cap
        }
    }

    pub fn add_collateral(&mut self, agent_id: [u8; 32], units: f64) -> Result<(), String> {
        if units <= 0.0 {
            return Err("Collateral amount must be positive".to_string());
        }

        if self.total_units + units > self.deposit_cap {
            return Err("Adding collateral would exceed deposit cap".to_string());
        }

        *self.collateral_positions.entry(agent_id).or_insert(0.0) += units;
        self.total_units += units;
        Ok(())
    }

    pub fn remove_collateral(&mut self, agent_id: [u8; 32], units: f64) -> Result<(), String> {
        if units <= 0.0 {
            return Err("Collateral amount must be positive".to_string());
        }

        let current_balance = self.collateral_positions.get(&agent_id).copied().unwrap_or(0.0);
        if current_balance < units {
            return Err("Insufficient collateral balance".to_string());
        }

        let new_balance = current_balance - units;
        if new_balance <= 0.0 {
            self.collateral_positions.remove(&agent_id);
        } else {
            self.collateral_positions.insert(agent_id, new_balance);
        }

        self.total_units -= units;
        Ok(())
    }

    pub fn borrow_units(&mut self, units: f64) -> Result<(), String> {
        if units <= 0.0 {
            return Err("Borrow amount must be positive".to_string());
        }

        let available_units = self.total_units * (1.0 - self.reserve_ratio) - self.units_borrowed;
        if units > available_units {
            return Err("Insufficient liquidity available".to_string());
        }

        self.units_borrowed += units;
        Ok(())
    }

    pub fn repay_units(&mut self, units: f64) -> Result<(), String> {
        if units <= 0.0 {
            return Err("Repay amount must be positive".to_string());
        }

        if units > self.units_borrowed {
            return Err("Repay amount exceeds borrowed units".to_string());
        }

        self.units_borrowed -= units;
        Ok(())
    }

    pub fn get_collateral_balance(&self, agent_id: &[u8; 32]) -> f64 {
        self.collateral_positions.get(agent_id).copied().unwrap_or(0.0)
    }

    pub fn get_total_units(&self) -> f64 {
        self.total_units
    }

    pub fn get_units_borrowed(&self) -> f64 {
        self.units_borrowed
    }

    pub fn get_available_liquidity(&self) -> f64 {
        self.total_units * (1.0 - self.reserve_ratio) - self.units_borrowed
    }

    pub fn get_utilization_ratio(&self) -> f64 {
        if self.total_units == 0.0 {
            0.0
        } else {
            self.units_borrowed / self.total_units
        }
    }

    pub fn update_oracle_price(&mut self, price: f64) {
        self.asset_info.oracle_price = Some(price);
    }

    pub fn get_ltv_max(&self) -> f64 {
        self.ltv_max
    }

    pub fn get_reserve_ratio(&self) -> f64 {
        self.reserve_ratio
    }
}

impl Collateral {
    pub fn new(asset_id: [u8; 32], units: f64) -> Self {
        Collateral { asset_id, units }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_asset_metadata() -> AssetMetadata {
        AssetMetadata::new([0x01; 32])
    }

    fn create_test_pool() -> Pool {
        let asset_info = create_test_asset_metadata();
        Pool::new(asset_info, 0.8, 10000.0)
    }

    #[test]
    fn test_asset_metadata_new() {
        let asset_id = [0x42; 32];
        let metadata = AssetMetadata::new(asset_id);

        assert_eq!(metadata.asset_id, asset_id);
        assert_eq!(metadata.oracle_price, None);
    }

    #[test]
    fn test_collateral_new() {
        let asset_id = [0x42; 32];
        let units = 100.5;
        let collateral = Collateral::new(asset_id, units);

        assert_eq!(collateral.asset_id, asset_id);
        assert_eq!(collateral.units, units);
    }

    #[test]
    fn test_liquidation_action_debug() {
        let partial = LiquidationAction::PartialLiquidation;
        let full = LiquidationAction::FullLiquidation;

        assert!(format!("{:?}", partial).contains("PartialLiquidation"));
        assert!(format!("{:?}", full).contains("FullLiquidation"));
    }

    #[test]
    fn test_pool_new() {
        let pool = create_test_pool();

        assert_eq!(pool.total_units, 0.0);
        assert_eq!(pool.reserve_ratio, 0.2);
        assert_eq!(pool.units_borrowed, 0.0);
        assert_eq!(pool.ltv_max, 0.8);
        assert_eq!(pool.deposit_cap, 10000.0);
        assert!(pool.collateral_positions.is_empty());
        assert_eq!(pool.asset_info.asset_id, [0x01; 32]);
        assert_eq!(pool.asset_info.oracle_price, None);
    }

    #[test]
    fn test_pool_add_collateral_success() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        let result = pool.add_collateral(agent_id, 500.0);
        assert!(result.is_ok());
        assert_eq!(pool.get_collateral_balance(&agent_id), 500.0);
        assert_eq!(pool.get_total_units(), 500.0);
    }

    #[test]
    fn test_pool_add_collateral_multiple_agents() {
        let mut pool = create_test_pool();
        let agent1 = [0x02; 32];
        let agent2 = [0x03; 32];

        pool.add_collateral(agent1, 300.0).unwrap();
        pool.add_collateral(agent2, 200.0).unwrap();

        assert_eq!(pool.get_collateral_balance(&agent1), 300.0);
        assert_eq!(pool.get_collateral_balance(&agent2), 200.0);
        assert_eq!(pool.get_total_units(), 500.0);
    }

    #[test]
    fn test_pool_add_collateral_same_agent_multiple_times() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 300.0).unwrap();
        pool.add_collateral(agent_id, 200.0).unwrap();

        assert_eq!(pool.get_collateral_balance(&agent_id), 500.0);
        assert_eq!(pool.get_total_units(), 500.0);
    }

    #[test]
    fn test_pool_add_collateral_exceeds_cap() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        let result = pool.add_collateral(agent_id, 15000.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Adding collateral would exceed deposit cap");
        assert_eq!(pool.get_collateral_balance(&agent_id), 0.0);
        assert_eq!(pool.get_total_units(), 0.0);
    }

    #[test]
    fn test_pool_add_collateral_exactly_at_cap() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        let result = pool.add_collateral(agent_id, 10000.0);
        assert!(result.is_ok());
        assert_eq!(pool.get_total_units(), 10000.0);
    }

    #[test]
    fn test_pool_add_collateral_zero_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        let result = pool.add_collateral(agent_id, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Collateral amount must be positive");
    }

    #[test]
    fn test_pool_add_collateral_negative_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        let result = pool.add_collateral(agent_id, -100.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Collateral amount must be positive");
    }

    #[test]
    fn test_pool_remove_collateral_success() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        let result = pool.remove_collateral(agent_id, 300.0);

        assert!(result.is_ok());
        assert_eq!(pool.get_collateral_balance(&agent_id), 700.0);
        assert_eq!(pool.get_total_units(), 700.0);
    }

    #[test]
    fn test_pool_remove_collateral_full_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        let result = pool.remove_collateral(agent_id, 1000.0);

        assert!(result.is_ok());
        assert_eq!(pool.get_collateral_balance(&agent_id), 0.0);
        assert_eq!(pool.get_total_units(), 0.0);
        assert!(!pool.collateral_positions.contains_key(&agent_id));
    }

    #[test]
    fn test_pool_remove_collateral_insufficient_balance() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 500.0).unwrap();
        let result = pool.remove_collateral(agent_id, 1000.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Insufficient collateral balance");
        assert_eq!(pool.get_collateral_balance(&agent_id), 500.0);
    }

    #[test]
    fn test_pool_remove_collateral_nonexistent_agent() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        let result = pool.remove_collateral(agent_id, 100.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Insufficient collateral balance");
    }

    #[test]
    fn test_pool_remove_collateral_zero_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        let result = pool.remove_collateral(agent_id, 0.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Collateral amount must be positive");
    }

    #[test]
    fn test_pool_remove_collateral_negative_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        let result = pool.remove_collateral(agent_id, -100.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Collateral amount must be positive");
    }

    #[test]
    fn test_pool_borrow_units_success() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        let result = pool.borrow_units(500.0);

        assert!(result.is_ok());
        assert_eq!(pool.get_units_borrowed(), 500.0);
        assert_eq!(pool.get_available_liquidity(), 300.0);
    }

    #[test]
    fn test_pool_borrow_units_max_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        let available = pool.get_available_liquidity();
        let result = pool.borrow_units(available);

        assert!(result.is_ok());
        assert_eq!(pool.get_units_borrowed(), 800.0);
        assert_eq!(pool.get_available_liquidity(), 0.0);
    }

    #[test]
    fn test_pool_borrow_units_insufficient_liquidity() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        let result = pool.borrow_units(900.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Insufficient liquidity available");
        assert_eq!(pool.get_units_borrowed(), 0.0);
    }

    #[test]
    fn test_pool_borrow_units_zero_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        let result = pool.borrow_units(0.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Borrow amount must be positive");
    }

    #[test]
    fn test_pool_borrow_units_negative_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        let result = pool.borrow_units(-100.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Borrow amount must be positive");
    }

    #[test]
    fn test_pool_repay_units_success() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        pool.borrow_units(500.0).unwrap();

        let result = pool.repay_units(200.0);
        assert!(result.is_ok());
        assert_eq!(pool.get_units_borrowed(), 300.0);
        assert_eq!(pool.get_available_liquidity(), 500.0);
    }

    #[test]
    fn test_pool_repay_units_full_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        pool.borrow_units(500.0).unwrap();

        let result = pool.repay_units(500.0);
        assert!(result.is_ok());
        assert_eq!(pool.get_units_borrowed(), 0.0);
        assert_eq!(pool.get_available_liquidity(), 800.0);
    }

    #[test]
    fn test_pool_repay_units_exceeds_borrowed() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        pool.borrow_units(300.0).unwrap();

        let result = pool.repay_units(500.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Repay amount exceeds borrowed units");
        assert_eq!(pool.get_units_borrowed(), 300.0);
    }

    #[test]
    fn test_pool_repay_units_no_debt() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();

        let result = pool.repay_units(100.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Repay amount exceeds borrowed units");
    }

    #[test]
    fn test_pool_repay_units_zero_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        pool.borrow_units(500.0).unwrap();

        let result = pool.repay_units(0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Repay amount must be positive");
    }

    #[test]
    fn test_pool_repay_units_negative_amount() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        pool.add_collateral(agent_id, 1000.0).unwrap();
        pool.borrow_units(500.0).unwrap();

        let result = pool.repay_units(-100.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Repay amount must be positive");
    }

    #[test]
    fn test_pool_get_utilization_ratio() {
        let mut pool = create_test_pool();
        let agent_id = [0x02; 32];

        assert_eq!(pool.get_utilization_ratio(), 0.0);

        pool.add_collateral(agent_id, 1000.0).unwrap();
        assert_eq!(pool.get_utilization_ratio(), 0.0);

        pool.borrow_units(200.0).unwrap();
        assert_eq!(pool.get_utilization_ratio(), 0.2);

        pool.borrow_units(300.0).unwrap();
        assert_eq!(pool.get_utilization_ratio(), 0.5);
    }

    #[test]
    fn test_pool_update_oracle_price() {
        let mut pool = create_test_pool();

        assert_eq!(pool.asset_info.oracle_price, None);

        pool.update_oracle_price(100.5);
        assert_eq!(pool.asset_info.oracle_price, Some(100.5));

        pool.update_oracle_price(95.75);
        assert_eq!(pool.asset_info.oracle_price, Some(95.75));
    }

    #[test]
    fn test_pool_get_collateral_balance_nonexistent() {
        let pool = create_test_pool();
        let agent_id = [0x99; 32];

        assert_eq!(pool.get_collateral_balance(&agent_id), 0.0);
    }

    #[test]
    fn test_pool_complex_scenario() {
        let mut pool = create_test_pool();
        let agent1 = [0x02; 32];
        let agent2 = [0x03; 32];

        pool.add_collateral(agent1, 3000.0).unwrap();
        pool.add_collateral(agent2, 2000.0).unwrap();
        assert_eq!(pool.get_total_units(), 5000.0);
        assert_eq!(pool.get_available_liquidity(), 4000.0);

        pool.borrow_units(1500.0).unwrap();
        assert_eq!(pool.get_units_borrowed(), 1500.0);
        assert_eq!(pool.get_available_liquidity(), 2500.0);
        assert_eq!(pool.get_utilization_ratio(), 0.3);

        pool.remove_collateral(agent1, 500.0).unwrap();
        assert_eq!(pool.get_total_units(), 4500.0);
        assert_eq!(pool.get_available_liquidity(), 2100.0);

        pool.repay_units(500.0).unwrap();
        assert_eq!(pool.get_units_borrowed(), 1000.0);
        assert_eq!(pool.get_available_liquidity(), 2600.0);

        pool.update_oracle_price(150.0);
        assert_eq!(pool.asset_info.oracle_price, Some(150.0));

        assert_eq!(pool.get_collateral_balance(&agent1), 2500.0);
        assert_eq!(pool.get_collateral_balance(&agent2), 2000.0);
    }
}
