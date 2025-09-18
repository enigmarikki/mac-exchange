#[cfg(test)]
mod integration_tests {
    use crate::exchange::{BoostExchange, Position, CollateralPosition, USDC_POOL};
    use crate::agents::Agent;
    use crate::exchange_types::{Collateral, LiquidationAction};
    use std::collections::HashMap;

    fn setup_exchange_with_agents() -> (BoostExchange, Vec<Agent>) {
        let mut exchange = BoostExchange::new(0.8, 0.2, 1_000_000.0);

        let mut agents = Vec::new();

        // Agent 1: High collateral, conservative
        let agent1_collateral = vec![
            Collateral::new([0x01; 32], 1000.0),
            Collateral::new([0x02; 32], 500.0),
        ];
        let agent1 = Agent::new(agent1_collateral);
        agents.push(agent1);

        // Agent 2: Medium collateral, moderate risk
        let agent2_collateral = vec![
            Collateral::new([0x01; 32], 500.0),
            Collateral::new([0x03; 32], 300.0),
        ];
        let agent2 = Agent::new(agent2_collateral);
        agents.push(agent2);

        // Agent 3: Low collateral, high risk
        let agent3_collateral = vec![
            Collateral::new([0x02; 32], 200.0),
        ];
        let agent3 = Agent::new(agent3_collateral);
        agents.push(agent3);

        // Add agents to exchange
        for agent in &agents {
            exchange.add_agent(agent.clone());
        }

        (exchange, agents)
    }

    #[test]
    fn test_full_trading_lifecycle() {
        let (mut exchange, mut agents) = setup_exchange_with_agents();
        let agent1_id = agents[0].get_id();
        let agent2_id = agents[1].get_id();

        // Agent 1 deposits additional collateral
        agents[0].deposit_collateral([0x04; 32], 2000.0).unwrap();

        // Agent 1 borrows USDC
        let loan_amount = 50_000.0;
        agents[0].borrow(loan_amount).unwrap();
        exchange.register_loan(agent1_id, loan_amount).unwrap();

        // Agent 1 opens long position
        agents[0].open_position([0x05; 32], 100.0).unwrap();

        // Agent 2 opens short position
        agents[1].open_position([0x05; 32], -50.0).unwrap();

        // Check agent states
        assert_eq!(agents[0].get_loan_amount(), loan_amount);
        assert_eq!(agents[0].get_position_size(&[0x05; 32]), 100.0);
        assert_eq!(agents[1].get_position_size(&[0x05; 32]), -50.0);
        assert_eq!(exchange.get_agent_loan(&agent1_id), loan_amount);

        // Agent 1 partially closes position
        agents[0].close_position([0x05; 32], -30.0).unwrap();
        assert_eq!(agents[0].get_position_size(&[0x05; 32]), 70.0);

        // Agent 1 repays part of loan
        let repay_amount = 15_000.0;
        agents[0].repay(repay_amount).unwrap();
        exchange.repay_loan(agent1_id, repay_amount).unwrap();

        assert_eq!(agents[0].get_loan_amount(), loan_amount - repay_amount);
        assert_eq!(exchange.get_agent_loan(&agent1_id), loan_amount - repay_amount);
    }

    #[test]
    fn test_liquidation_scenario() {
        let (mut exchange, mut agents) = setup_exchange_with_agents();
        let agent_id = agents[2].get_id(); // Agent 3 with low collateral

        // Agent 3 borrows heavily
        let loan_amount = 15_000.0;
        agents[2].borrow(loan_amount).unwrap();
        exchange.register_loan(agent_id, loan_amount).unwrap();

        // Agent 3 opens large leveraged position
        agents[2].open_position([0x05; 32], 200.0).unwrap();

        // Setup positions and collateral for monitoring
        let mut all_positions = HashMap::new();
        let mut all_collateral = HashMap::new();
        let mut all_twm = HashMap::new();
        let mut mark_prices = HashMap::new();

        // Agent 3's position
        let mut agent3_positions = HashMap::new();
        agent3_positions.insert([0x05; 32], Position {
            size: 200.0,
            entry_price: 100.0,
            max_leverage: 2.0,
        });
        all_positions.insert(agent_id, agent3_positions);

        // Agent 3's collateral (simulate low value)
        let mut agent3_collateral = HashMap::new();
        agent3_collateral.insert([0x02; 32], CollateralPosition {
            units: 200.0,
            entry_price: 100.0,
            current_price: 50.0, // Price dropped significantly
        });
        all_collateral.insert(agent_id, agent3_collateral);

        // Trading wallet margin (negative due to losses)
        all_twm.insert(agent_id, -5000.0);

        // Mark prices (unfavorable for long position)
        mark_prices.insert([0x05; 32], 80.0); // Price dropped from entry

        // Monitor for liquidations
        let liquidations = exchange.monitor_all_agents(&all_positions, &all_collateral, &all_twm, &mark_prices);

        assert_eq!(liquidations.len(), 1);
        assert_eq!(liquidations[0].0, agent_id);
        // Could be either partial or full liquidation depending on exact calculations
        assert!(matches!(liquidations[0].1, LiquidationAction::PartialLiquidation | LiquidationAction::FullLiquidation));
    }

    #[test]
    fn test_multi_agent_interaction() {
        let (mut exchange, mut agents) = setup_exchange_with_agents();
        let agent1_id = agents[0].get_id();
        let agent2_id = agents[1].get_id();
        let agent3_id = agents[2].get_id();

        // Multiple agents borrow different amounts
        exchange.register_loan(agent1_id, 100_000.0).unwrap();
        exchange.register_loan(agent2_id, 50_000.0).unwrap();
        exchange.register_loan(agent3_id, 25_000.0).unwrap();

        // Check total loans affect available liquidity
        let expected_available = 1_000_000.0 * 0.8 - 175_000.0;
        assert_eq!(exchange.calculate_available_usdc(), expected_available);

        // One agent repays loan
        exchange.repay_loan(agent2_id, 30_000.0).unwrap();
        let new_expected_available = expected_available + 30_000.0;
        assert_eq!(exchange.calculate_available_usdc(), new_expected_available);

        // Check individual loan amounts
        assert_eq!(exchange.get_agent_loan(&agent1_id), 100_000.0);
        assert_eq!(exchange.get_agent_loan(&agent2_id), 20_000.0);
        assert_eq!(exchange.get_agent_loan(&agent3_id), 25_000.0);
    }

    #[test]
    fn test_price_update_and_monitoring() {
        let (mut exchange, agents) = setup_exchange_with_agents();
        let agent1_id = agents[0].get_id();

        // Setup initial state
        exchange.register_loan(agent1_id, 40_000.0).unwrap();

        // Create price updates
        let mut price_updates = HashMap::new();
        price_updates.insert([0x01; 32], 120.0);
        price_updates.insert([0x02; 32], 95.0);
        price_updates.insert(USDC_POOL, 1.01);

        exchange.update_prices(price_updates);

        // Verify prices were updated
        let usdc_pool = exchange.liquidity_pool.get(&USDC_POOL).unwrap();
        assert_eq!(usdc_pool.asset_info.oracle_price, Some(1.01));

        // Test monitoring with updated prices
        let mut all_positions = HashMap::new();
        let mut all_collateral = HashMap::new();
        let mut all_twm = HashMap::new();
        let mut mark_prices = HashMap::new();

        // Agent with healthy position
        let mut agent1_positions = HashMap::new();
        agent1_positions.insert([0x05; 32], Position {
            size: 50.0,
            entry_price: 100.0,
            max_leverage: 5.0,
        });
        all_positions.insert(agent1_id, agent1_positions);

        let mut agent1_collateral = HashMap::new();
        agent1_collateral.insert([0x01; 32], CollateralPosition {
            units: 1000.0,
            entry_price: 100.0,
            current_price: 120.0, // Updated price
        });
        all_collateral.insert(agent1_id, agent1_collateral);

        all_twm.insert(agent1_id, 10_000.0);
        mark_prices.insert([0x05; 32], 105.0);

        let liquidations = exchange.monitor_all_agents(&all_positions, &all_collateral, &all_twm, &mark_prices);
        assert!(liquidations.is_empty()); // Should be healthy
    }

    #[test]
    fn test_collateral_and_position_calculations() {
        let exchange = BoostExchange::new(0.75, 0.25, 2_000_000.0);

        // Test collateral value calculation
        let mut collateral_positions = HashMap::new();
        collateral_positions.insert([0x01; 32], CollateralPosition {
            units: 500.0,
            entry_price: 200.0,
            current_price: 250.0,
        });
        collateral_positions.insert([0x02; 32], CollateralPosition {
            units: 200.0,
            entry_price: 150.0,
            current_price: 140.0,
        });

        let total_collateral_value = exchange.calculate_collateral_value(&collateral_positions);
        assert_eq!(total_collateral_value, 500.0 * 250.0 + 200.0 * 140.0);
        assert_eq!(total_collateral_value, 153_000.0);

        // Test max loan calculation
        let max_loan = exchange.calculate_max_loan(total_collateral_value);
        let collateral_limit = total_collateral_value * 0.75;
        let liquidity_limit = 2_000_000.0 * 0.75;
        assert_eq!(max_loan, collateral_limit.min(liquidity_limit));
        assert_eq!(max_loan, 114_750.0);

        // Test trading wallet equity
        let mut positions = HashMap::new();
        positions.insert([0x03; 32], Position {
            size: 100.0,
            entry_price: 50.0,
            max_leverage: 4.0,
        });
        positions.insert([0x04; 32], Position {
            size: -75.0,
            entry_price: 80.0,
            max_leverage: 3.0,
        });

        let mut mark_prices = HashMap::new();
        mark_prices.insert([0x03; 32], 55.0);
        mark_prices.insert([0x04; 32], 75.0);

        let trading_wallet_margin = 20_000.0;
        let trading_equity = exchange.calculate_trading_wallet_equity(&positions, &mark_prices, trading_wallet_margin);

        let expected_pnl = 100.0 * (55.0 - 50.0) + (-75.0) * (75.0 - 80.0);
        assert_eq!(trading_equity, expected_pnl + trading_wallet_margin);
        assert_eq!(trading_equity, 500.0 + 375.0 + 20_000.0);
        assert_eq!(trading_equity, 20_875.0);

        // Test maintenance margin requirement
        let loan = 80_000.0;
        let mmr = exchange.calculate_maintenance_margin_requirement(&positions, &mark_prices, loan);

        let pos1_mmr = 55.0 * 100.0 / (2.0 * 4.0);
        let pos2_mmr = 75.0 * 75.0 / (2.0 * 3.0);
        let liquidation_slack = 0.05 * loan;

        assert_eq!(mmr, pos1_mmr + pos2_mmr + liquidation_slack);
        assert_eq!(mmr, 687.5 + 937.5 + 4_000.0);
        assert_eq!(mmr, 5_625.0);

        // Test boost account equity
        let boost_equity = exchange.calculate_boost_account_equity(total_collateral_value, trading_equity, loan);
        assert_eq!(boost_equity, total_collateral_value * 0.75 + trading_equity - loan);
        assert_eq!(boost_equity, 114_750.0 + 20_875.0 - 80_000.0);
        assert_eq!(boost_equity, 55_625.0);

        // Test health checks
        assert!(exchange.is_healthy(boost_equity, mmr));
        assert!(!exchange.should_partial_liquidate(boost_equity, mmr, total_collateral_value, trading_equity, loan));
        assert!(!exchange.should_full_liquidate(total_collateral_value, trading_equity, loan));
    }

    #[test]
    fn test_extreme_market_conditions() {
        let (mut exchange, mut agents) = setup_exchange_with_agents();
        let agent_id = agents[0].get_id();

        // Agent takes maximum leverage
        let collateral_value = 100_000.0;
        let max_loan = exchange.calculate_max_loan(collateral_value);
        exchange.register_loan(agent_id, max_loan).unwrap();
        agents[0].borrow(max_loan).unwrap();

        // Agent opens large position
        agents[0].open_position([0x05; 32], 1000.0).unwrap();

        // Simulate extreme market crash
        let mut all_positions = HashMap::new();
        let mut all_collateral = HashMap::new();
        let mut all_twm = HashMap::new();
        let mut mark_prices = HashMap::new();

        let mut agent_positions = HashMap::new();
        agent_positions.insert([0x05; 32], Position {
            size: 1000.0,
            entry_price: 100.0,
            max_leverage: 2.0,
        });
        all_positions.insert(agent_id, agent_positions);

        let mut agent_collateral = HashMap::new();
        agent_collateral.insert([0x01; 32], CollateralPosition {
            units: 1000.0,
            entry_price: 100.0,
            current_price: 30.0, // 70% crash
        });
        all_collateral.insert(agent_id, agent_collateral);

        all_twm.insert(agent_id, -50_000.0); // Large losses
        mark_prices.insert([0x05; 32], 50.0); // 50% drop in position

        let liquidations = exchange.monitor_all_agents(&all_positions, &all_collateral, &all_twm, &mark_prices);

        assert_eq!(liquidations.len(), 1);
        assert_eq!(liquidations[0].0, agent_id);
        assert!(matches!(liquidations[0].1, LiquidationAction::FullLiquidation));
    }

    #[test]
    fn test_complex_multi_asset_scenario() {
        let (mut exchange, mut agents) = setup_exchange_with_agents();

        // Each agent trades different assets with different characteristics
        for (i, agent) in agents.iter_mut().enumerate() {
            let agent_id = agent.get_id();
            let loan_amount = (i + 1) as f64 * 20_000.0;

            // Register loans
            exchange.register_loan(agent_id, loan_amount).unwrap();
            agent.borrow(loan_amount).unwrap();

            // Open different positions
            let asset_id = [i as u8 + 1; 32];
            let position_size = if i % 2 == 0 { 100.0 } else { -100.0 };
            agent.open_position(asset_id, position_size).unwrap();

            // Add more collateral
            agent.deposit_collateral([i as u8 + 10; 32], 1000.0).unwrap();
        }

        // Verify state
        assert_eq!(exchange.get_agent_loan(&agents[0].get_id()), 20_000.0);
        assert_eq!(exchange.get_agent_loan(&agents[1].get_id()), 40_000.0);
        assert_eq!(exchange.get_agent_loan(&agents[2].get_id()), 60_000.0);

        assert_eq!(agents[0].get_position_size(&[1; 32]), 100.0);
        assert_eq!(agents[1].get_position_size(&[2; 32]), -100.0);
        assert_eq!(agents[2].get_position_size(&[3; 32]), 100.0);

        // Calculate total system utilization
        let total_loans: f64 = (0..3).map(|i| exchange.get_agent_loan(&agents[i].get_id())).sum();
        assert_eq!(total_loans, 120_000.0);

        let available_liquidity = exchange.calculate_available_usdc();
        assert_eq!(available_liquidity, 800_000.0 - 120_000.0);
    }
}