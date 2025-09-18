use rand::{RngCore, thread_rng};
use sha2::Digest;
use std::collections::HashMap;

use crate::exchange_types::Collateral;
#[derive(Clone)]
pub struct Agent {
    id: [u8; 32],
    pnl: f64,
    loan: f64,
    boost_account_equity: f64,
    trading_account_equtity: f64,
    positions: HashMap<[u8; 32], f64>,
    collateral_deposits: HashMap<[u8; 32], f64>,
    collateral_value: f64,
}
// Agent can do these 
impl Agent {
    pub fn new(collateral_deposits: Vec<Collateral>) -> Self {
        let mut rng = thread_rng();
        let mut random_bytes = [0u8; 32];
        rng.fill_bytes(&mut random_bytes);
        let mut deposits: HashMap<[u8; 32], f64> = HashMap::new();
        for collateral in collateral_deposits {
            deposits.insert(collateral.asset_id, collateral.units);
        }

        Agent {
            id: random_bytes,
            pnl: 0.0,
            loan: 0.0,
            boost_account_equity: 0.0,
            trading_account_equtity: 0.0,
            positions: HashMap::new(),
            collateral_deposits: deposits,
            collateral_value: 0.0,
        }
    }
    pub fn get_id(&self)-> [u8 ;32]{
        self.id
    }

    pub fn deposit_collateral(&mut self, asset_id: [u8; 32], units: f64) -> Result<(), String> {
        if units <= 0.0 {
            return Err("Deposit amount must be positive".to_string());
        }

        *self.collateral_deposits.entry(asset_id).or_insert(0.0) += units;
        Ok(())
    }

    pub fn withdraw_collateral(&mut self, asset_id: [u8; 32], units: f64) -> Result<(), String> {
        if units <= 0.0 {
            return Err("Withdrawal amount must be positive".to_string());
        }

        let current_balance = self.collateral_deposits.get(&asset_id).copied().unwrap_or(0.0);
        if current_balance < units {
            return Err("Insufficient collateral balance".to_string());
        }

        let new_balance = current_balance - units;
        if new_balance <= 0.0 {
            self.collateral_deposits.remove(&asset_id);
        } else {
            self.collateral_deposits.insert(asset_id, new_balance);
        }

        Ok(())
    }

    pub fn borrow(&mut self, amount: f64) -> Result<(), String> {
        if amount <= 0.0 {
            return Err("Borrow amount must be positive".to_string());
        }

        self.loan += amount;
        Ok(())
    }

    pub fn repay(&mut self, amount: f64) -> Result<(), String> {
        if amount <= 0.0 {
            return Err("Repay amount must be positive".to_string());
        }

        if amount > self.loan {
            return Err("Repay amount exceeds outstanding loan".to_string());
        }

        self.loan -= amount;
        Ok(())
    }

    pub fn open_position(&mut self, asset_id: [u8; 32], size: f64) -> Result<(), String> {
        if size == 0.0 {
            return Err("Position size cannot be zero".to_string());
        }

        *self.positions.entry(asset_id).or_insert(0.0) += size;
        Ok(())
    }

    pub fn close_position(&mut self, asset_id: [u8; 32], size: f64) -> Result<(), String> {
        if size == 0.0 {
            return Err("Position size cannot be zero".to_string());
        }

        let current_position = self.positions.get(&asset_id).copied().unwrap_or(0.0);

        if size.abs() > current_position.abs() {
            return Err("Cannot close more than current position size".to_string());
        }

        if (current_position > 0.0 && size > 0.0) || (current_position < 0.0 && size < 0.0) {
            return Err("Close size must be opposite direction of current position".to_string());
        }

        let new_position = current_position + size;
        if new_position.abs() < 1e-10 {
            self.positions.remove(&asset_id);
        } else {
            self.positions.insert(asset_id, new_position);
        }

        Ok(())
    }

    pub fn update_pnl(&mut self, pnl: f64) {
        self.pnl = pnl;
    }

    pub fn update_boost_account_equity(&mut self, equity: f64) {
        self.boost_account_equity = equity;
    }

    pub fn update_trading_account_equity(&mut self, equity: f64) {
        self.trading_account_equtity = equity;
    }

    pub fn update_collateral_value(&mut self, value: f64) {
        self.collateral_value = value;
    }

    pub fn get_collateral_balance(&self, asset_id: &[u8; 32]) -> f64 {
        self.collateral_deposits.get(asset_id).copied().unwrap_or(0.0)
    }

    pub fn get_position_size(&self, asset_id: &[u8; 32]) -> f64 {
        self.positions.get(asset_id).copied().unwrap_or(0.0)
    }

    pub fn get_loan_amount(&self) -> f64 {
        self.loan
    }

    pub fn get_pnl(&self) -> f64 {
        self.pnl
    }

    pub fn get_boost_account_equity(&self) -> f64 {
        self.boost_account_equity
    }

    pub fn get_trading_account_equity(&self) -> f64 {
        self.trading_account_equtity
    }

    pub fn get_collateral_value(&self) -> f64 {
        self.collateral_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exchange_types::Collateral;

    #[test]
    fn test_agent_new_with_collateral() {
        let collateral_deposits = vec![
            Collateral { asset_id: [0x01; 32], units: 100.0 },
            Collateral { asset_id: [0x02; 32], units: 50.0 },
            Collateral { asset_id: [0x03; 32], units: 25.5 },
        ];

        let agent = Agent::new(collateral_deposits);

        assert_eq!(agent.pnl, 0.0);
        assert_eq!(agent.loan, 0.0);
        assert_eq!(agent.boost_account_equity, 0.0);
        assert_eq!(agent.trading_account_equtity, 0.0);
        assert_eq!(agent.collateral_value, 0.0);
        assert!(agent.positions.is_empty());

        assert_eq!(agent.collateral_deposits.len(), 3);
        assert_eq!(*agent.collateral_deposits.get(&[0x01; 32]).unwrap(), 100.0);
        assert_eq!(*agent.collateral_deposits.get(&[0x02; 32]).unwrap(), 50.0);
        assert_eq!(*agent.collateral_deposits.get(&[0x03; 32]).unwrap(), 25.5);
    }

    #[test]
    fn test_agent_new_empty_collateral() {
        let collateral_deposits = vec![];
        let agent = Agent::new(collateral_deposits);

        assert_eq!(agent.pnl, 0.0);
        assert_eq!(agent.loan, 0.0);
        assert_eq!(agent.boost_account_equity, 0.0);
        assert_eq!(agent.trading_account_equtity, 0.0);
        assert_eq!(agent.collateral_value, 0.0);
        assert!(agent.positions.is_empty());
        assert!(agent.collateral_deposits.is_empty());
    }

    #[test]
    fn test_agent_id_is_unique() {
        let collateral1 = vec![Collateral { asset_id: [0x01; 32], units: 100.0 }];
        let collateral2 = vec![Collateral { asset_id: [0x02; 32], units: 50.0 }];

        let agent1 = Agent::new(collateral1);
        let agent2 = Agent::new(collateral2);

        assert_ne!(agent1.get_id(), agent2.get_id());
    }

    #[test]
    fn test_agent_id_length() {
        let agent = Agent::new(vec![]);
        let id = agent.get_id();
        assert_eq!(id.len(), 32);
    }

    #[test]
    fn test_agent_collateral_with_duplicate_assets() {
        let collateral_deposits = vec![
            Collateral { asset_id: [0x01; 32], units: 100.0 },
            Collateral { asset_id: [0x01; 32], units: 50.0 },
        ];

        let agent = Agent::new(collateral_deposits);

        assert_eq!(agent.collateral_deposits.len(), 1);
        assert_eq!(*agent.collateral_deposits.get(&[0x01; 32]).unwrap(), 50.0);
    }

    #[test]
    fn test_agent_collateral_with_zero_units() {
        let collateral_deposits = vec![
            Collateral { asset_id: [0x01; 32], units: 0.0 },
            Collateral { asset_id: [0x02; 32], units: 100.0 },
        ];

        let agent = Agent::new(collateral_deposits);

        assert_eq!(agent.collateral_deposits.len(), 2);
        assert_eq!(*agent.collateral_deposits.get(&[0x01; 32]).unwrap(), 0.0);
        assert_eq!(*agent.collateral_deposits.get(&[0x02; 32]).unwrap(), 100.0);
    }

    #[test]
    fn test_agent_collateral_with_negative_units() {
        let collateral_deposits = vec![
            Collateral { asset_id: [0x01; 32], units: -50.0 },
            Collateral { asset_id: [0x02; 32], units: 100.0 },
        ];

        let agent = Agent::new(collateral_deposits);

        assert_eq!(agent.collateral_deposits.len(), 2);
        assert_eq!(*agent.collateral_deposits.get(&[0x01; 32]).unwrap(), -50.0);
        assert_eq!(*agent.collateral_deposits.get(&[0x02; 32]).unwrap(), 100.0);
    }

    #[test]
    fn test_agent_initial_state() {
        let agent = Agent::new(vec![]);

        assert_eq!(agent.pnl, 0.0);
        assert_eq!(agent.loan, 0.0);
        assert_eq!(agent.boost_account_equity, 0.0);
        assert_eq!(agent.trading_account_equtity, 0.0);
        assert_eq!(agent.collateral_value, 0.0);
        assert!(agent.positions.is_empty());
        assert!(agent.collateral_deposits.is_empty());
    }

    #[test]
    fn test_deposit_collateral_success() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        let result = agent.deposit_collateral(asset_id, 100.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_collateral_balance(&asset_id), 100.0);
    }

    #[test]
    fn test_deposit_collateral_existing_asset() {
        let collateral = vec![Collateral { asset_id: [0x01; 32], units: 50.0 }];
        let mut agent = Agent::new(collateral);
        let asset_id = [0x01; 32];

        let result = agent.deposit_collateral(asset_id, 75.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_collateral_balance(&asset_id), 125.0);
    }

    #[test]
    fn test_deposit_collateral_zero_amount() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        let result = agent.deposit_collateral(asset_id, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Deposit amount must be positive");
        assert_eq!(agent.get_collateral_balance(&asset_id), 0.0);
    }

    #[test]
    fn test_deposit_collateral_negative_amount() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        let result = agent.deposit_collateral(asset_id, -50.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Deposit amount must be positive");
        assert_eq!(agent.get_collateral_balance(&asset_id), 0.0);
    }

    #[test]
    fn test_withdraw_collateral_success() {
        let collateral = vec![Collateral { asset_id: [0x01; 32], units: 100.0 }];
        let mut agent = Agent::new(collateral);
        let asset_id = [0x01; 32];

        let result = agent.withdraw_collateral(asset_id, 30.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_collateral_balance(&asset_id), 70.0);
    }

    #[test]
    fn test_withdraw_collateral_full_amount() {
        let collateral = vec![Collateral { asset_id: [0x01; 32], units: 100.0 }];
        let mut agent = Agent::new(collateral);
        let asset_id = [0x01; 32];

        let result = agent.withdraw_collateral(asset_id, 100.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_collateral_balance(&asset_id), 0.0);
        assert!(!agent.collateral_deposits.contains_key(&asset_id));
    }

    #[test]
    fn test_withdraw_collateral_insufficient_balance() {
        let collateral = vec![Collateral { asset_id: [0x01; 32], units: 50.0 }];
        let mut agent = Agent::new(collateral);
        let asset_id = [0x01; 32];

        let result = agent.withdraw_collateral(asset_id, 100.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Insufficient collateral balance");
        assert_eq!(agent.get_collateral_balance(&asset_id), 50.0);
    }

    #[test]
    fn test_withdraw_collateral_nonexistent_asset() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        let result = agent.withdraw_collateral(asset_id, 10.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Insufficient collateral balance");
    }

    #[test]
    fn test_withdraw_collateral_zero_amount() {
        let collateral = vec![Collateral { asset_id: [0x01; 32], units: 100.0 }];
        let mut agent = Agent::new(collateral);
        let asset_id = [0x01; 32];

        let result = agent.withdraw_collateral(asset_id, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Withdrawal amount must be positive");
    }

    #[test]
    fn test_withdraw_collateral_negative_amount() {
        let collateral = vec![Collateral { asset_id: [0x01; 32], units: 100.0 }];
        let mut agent = Agent::new(collateral);
        let asset_id = [0x01; 32];

        let result = agent.withdraw_collateral(asset_id, -10.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Withdrawal amount must be positive");
    }

    #[test]
    fn test_borrow_success() {
        let mut agent = Agent::new(vec![]);

        let result = agent.borrow(10000.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_loan_amount(), 10000.0);
    }

    #[test]
    fn test_borrow_multiple_times() {
        let mut agent = Agent::new(vec![]);

        agent.borrow(5000.0).unwrap();
        agent.borrow(3000.0).unwrap();
        assert_eq!(agent.get_loan_amount(), 8000.0);
    }

    #[test]
    fn test_borrow_zero_amount() {
        let mut agent = Agent::new(vec![]);

        let result = agent.borrow(0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Borrow amount must be positive");
        assert_eq!(agent.get_loan_amount(), 0.0);
    }

    #[test]
    fn test_borrow_negative_amount() {
        let mut agent = Agent::new(vec![]);

        let result = agent.borrow(-1000.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Borrow amount must be positive");
        assert_eq!(agent.get_loan_amount(), 0.0);
    }

    #[test]
    fn test_repay_success() {
        let mut agent = Agent::new(vec![]);
        agent.borrow(10000.0).unwrap();

        let result = agent.repay(3000.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_loan_amount(), 7000.0);
    }

    #[test]
    fn test_repay_full_amount() {
        let mut agent = Agent::new(vec![]);
        agent.borrow(10000.0).unwrap();

        let result = agent.repay(10000.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_loan_amount(), 0.0);
    }

    #[test]
    fn test_repay_exceeds_loan() {
        let mut agent = Agent::new(vec![]);
        agent.borrow(5000.0).unwrap();

        let result = agent.repay(7000.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Repay amount exceeds outstanding loan");
        assert_eq!(agent.get_loan_amount(), 5000.0);
    }

    #[test]
    fn test_repay_no_loan() {
        let mut agent = Agent::new(vec![]);

        let result = agent.repay(1000.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Repay amount exceeds outstanding loan");
        assert_eq!(agent.get_loan_amount(), 0.0);
    }

    #[test]
    fn test_repay_zero_amount() {
        let mut agent = Agent::new(vec![]);
        agent.borrow(5000.0).unwrap();

        let result = agent.repay(0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Repay amount must be positive");
        assert_eq!(agent.get_loan_amount(), 5000.0);
    }

    #[test]
    fn test_repay_negative_amount() {
        let mut agent = Agent::new(vec![]);
        agent.borrow(5000.0).unwrap();

        let result = agent.repay(-1000.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Repay amount must be positive");
        assert_eq!(agent.get_loan_amount(), 5000.0);
    }

    #[test]
    fn test_open_position_long() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        let result = agent.open_position(asset_id, 10.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_position_size(&asset_id), 10.0);
    }

    #[test]
    fn test_open_position_short() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        let result = agent.open_position(asset_id, -5.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_position_size(&asset_id), -5.0);
    }

    #[test]
    fn test_open_position_add_to_existing() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        agent.open_position(asset_id, 10.0).unwrap();
        agent.open_position(asset_id, 5.0).unwrap();
        assert_eq!(agent.get_position_size(&asset_id), 15.0);
    }

    #[test]
    fn test_open_position_opposite_direction() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        agent.open_position(asset_id, 10.0).unwrap();
        agent.open_position(asset_id, -3.0).unwrap();
        assert_eq!(agent.get_position_size(&asset_id), 7.0);
    }

    #[test]
    fn test_open_position_zero_size() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        let result = agent.open_position(asset_id, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Position size cannot be zero");
        assert_eq!(agent.get_position_size(&asset_id), 0.0);
    }

    #[test]
    fn test_close_position_partial_long() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        agent.open_position(asset_id, 10.0).unwrap();
        let result = agent.close_position(asset_id, -3.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_position_size(&asset_id), 7.0);
    }

    #[test]
    fn test_close_position_partial_short() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        agent.open_position(asset_id, -10.0).unwrap();
        let result = agent.close_position(asset_id, 3.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_position_size(&asset_id), -7.0);
    }

    #[test]
    fn test_close_position_full_long() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        agent.open_position(asset_id, 10.0).unwrap();
        let result = agent.close_position(asset_id, -10.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_position_size(&asset_id), 0.0);
        assert!(!agent.positions.contains_key(&asset_id));
    }

    #[test]
    fn test_close_position_full_short() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        agent.open_position(asset_id, -10.0).unwrap();
        let result = agent.close_position(asset_id, 10.0);
        assert!(result.is_ok());
        assert_eq!(agent.get_position_size(&asset_id), 0.0);
        assert!(!agent.positions.contains_key(&asset_id));
    }

    #[test]
    fn test_close_position_wrong_direction() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        agent.open_position(asset_id, 10.0).unwrap();
        let result = agent.close_position(asset_id, 3.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Close size must be opposite direction of current position");
        assert_eq!(agent.get_position_size(&asset_id), 10.0);
    }

    #[test]
    fn test_close_position_exceeds_size() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        agent.open_position(asset_id, 5.0).unwrap();
        let result = agent.close_position(asset_id, -10.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Cannot close more than current position size");
        assert_eq!(agent.get_position_size(&asset_id), 5.0);
    }

    #[test]
    fn test_close_position_zero_size() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        agent.open_position(asset_id, 10.0).unwrap();
        let result = agent.close_position(asset_id, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Position size cannot be zero");
    }

    #[test]
    fn test_close_position_nonexistent() {
        let mut agent = Agent::new(vec![]);
        let asset_id = [0x01; 32];

        let result = agent.close_position(asset_id, -5.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Cannot close more than current position size");
    }

    #[test]
    fn test_update_and_get_methods() {
        let mut agent = Agent::new(vec![]);

        agent.update_pnl(1500.0);
        assert_eq!(agent.get_pnl(), 1500.0);

        agent.update_boost_account_equity(25000.0);
        assert_eq!(agent.get_boost_account_equity(), 25000.0);

        agent.update_trading_account_equity(15000.0);
        assert_eq!(agent.get_trading_account_equity(), 15000.0);

        agent.update_collateral_value(50000.0);
        assert_eq!(agent.get_collateral_value(), 50000.0);
    }

    #[test]
    fn test_get_collateral_balance_nonexistent() {
        let agent = Agent::new(vec![]);
        let asset_id = [0x99; 32];
        assert_eq!(agent.get_collateral_balance(&asset_id), 0.0);
    }

    #[test]
    fn test_get_position_size_nonexistent() {
        let agent = Agent::new(vec![]);
        let asset_id = [0x99; 32];
        assert_eq!(agent.get_position_size(&asset_id), 0.0);
    }

    #[test]
    fn test_complex_scenario() {
        let collateral = vec![
            Collateral { asset_id: [0x01; 32], units: 1000.0 },
            Collateral { asset_id: [0x02; 32], units: 500.0 },
        ];
        let mut agent = Agent::new(collateral);

        agent.borrow(50000.0).unwrap();
        agent.open_position([0x03; 32], 100.0).unwrap();
        agent.open_position([0x04; 32], -50.0).unwrap();
        agent.deposit_collateral([0x01; 32], 200.0).unwrap();
        agent.withdraw_collateral([0x02; 32], 100.0).unwrap();

        assert_eq!(agent.get_loan_amount(), 50000.0);
        assert_eq!(agent.get_position_size(&[0x03; 32]), 100.0);
        assert_eq!(agent.get_position_size(&[0x04; 32]), -50.0);
        assert_eq!(agent.get_collateral_balance(&[0x01; 32]), 1200.0);
        assert_eq!(agent.get_collateral_balance(&[0x02; 32]), 400.0);

        agent.close_position([0x03; 32], -25.0).unwrap();
        agent.repay(10000.0).unwrap();

        assert_eq!(agent.get_position_size(&[0x03; 32]), 75.0);
        assert_eq!(agent.get_loan_amount(), 40000.0);
    }
}
