use boost::{exchange::{BoostExchange, Position, CollateralPosition}, exchange_types::{AssetMetadata, Pool, Collateral, LiquidationAction}, agents::Agent};
use rand::{thread_rng, Rng};
use std::collections::HashMap;

pub const BTC_POOL: [u8; 32] = [0xBB; 32];
pub const HYPE_POOL: [u8; 32] = [0xCC; 32];
pub const SOL_POOL: [u8; 32] = [0xDD; 32];
pub const ETH_POOL: [u8; 32] = [0xEE; 32];
pub const BOOST_POOL: [u8; 32] = [0xFF; 32];
pub const USDC_POOL: [u8; 32] = [0xAA; 32];
pub const MASTER_KEY: [u8; 32] = [0x00; 32];

#[derive(Debug, Clone)]
struct SimulationTick {
    tick: u32,
    btc_price: f64,
    eth_price: f64,
    sol_price: f64,
    hype_price: f64,
    boost_price: f64,
    usdc_price: f64,
    liquidations: Vec<([u8; 32], LiquidationAction)>,
    usdc_pool_pnl: f64,
    total_loans_outstanding: f64,
}


fn initialize_exchange() -> BoostExchange {
    // Add trading pools
    let mut exchange = BoostExchange::new(MASTER_KEY);
    let btc_asset = AssetMetadata::new(BTC_POOL);
    let btc_pool = Pool::new(btc_asset, 0.8, 100_000.0);
    exchange.add_pool(MASTER_KEY, BTC_POOL, btc_pool).unwrap();

    let hype_asset = AssetMetadata::new(HYPE_POOL);
    let hype_pool = Pool::new(hype_asset, 0.7, 500_000.0);
    exchange.add_pool(MASTER_KEY, HYPE_POOL, hype_pool).unwrap();

    let sol_asset = AssetMetadata::new(SOL_POOL);
    let sol_pool = Pool::new(sol_asset, 0.75, 750_000.0);
    exchange.add_pool(MASTER_KEY, SOL_POOL, sol_pool).unwrap();

    let eth_asset = AssetMetadata::new(ETH_POOL);
    let eth_pool = Pool::new(eth_asset, 0.85, 2_000_000.0);
    exchange.add_pool(MASTER_KEY, ETH_POOL, eth_pool).unwrap();

    let boost_asset = AssetMetadata::new(BOOST_POOL);
    let boost_pool = Pool::new(boost_asset, 0.6, 500_000.0);
    exchange.add_pool(MASTER_KEY, BOOST_POOL, boost_pool).unwrap();

    let usdc_asset = AssetMetadata::new(USDC_POOL);
    let usdc_pool = Pool::new(usdc_asset, 0.9, 10_000_000.0);
    exchange.add_pool(MASTER_KEY, USDC_POOL, usdc_pool).unwrap();

    exchange
}

fn create_whales(count: u64) -> Vec<Agent> {
    let mut whales = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..count {
        let collateral_amount = rng.gen_range(100_000.0..1_000_000.0);

        let collateral = vec![
            Collateral {
                asset_id: BTC_POOL,
                units: collateral_amount * 0.4,
            },
            Collateral {
                asset_id: ETH_POOL,
                units: collateral_amount * 0.3,
            },
            Collateral {
                asset_id: SOL_POOL,
                units: collateral_amount * 0.2,
            },
            Collateral {
                asset_id: BOOST_POOL,
                units: collateral_amount * 0.1,
            },
        ];

        whales.push(Agent::new(collateral));
    }

    whales
}

fn create_degens(count: u64) -> Vec<Agent> {
    let mut degens = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..count {
        let collateral_amount = rng.gen_range(5_000.0..50_000.0);

        let collateral = vec![
            Collateral {
                asset_id: HYPE_POOL,
                units: collateral_amount * 0.6,
            },
            Collateral {
                asset_id: BOOST_POOL,
                units: collateral_amount * 0.3,
            },
            Collateral {
                asset_id: SOL_POOL,
                units: collateral_amount * 0.1,
            },
        ];

        degens.push(Agent::new(collateral));
    }

    degens
} 

fn create_plebs(count: u64) -> Vec<Agent> {
    let mut plebs = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..count {
        let collateral_amount = rng.gen_range(100.0..2_000.0);

        let collateral = vec![
            Collateral {
                asset_id: HYPE_POOL,
                units: collateral_amount * 0.7,
            },
            Collateral {
                asset_id: BOOST_POOL,
                units: collateral_amount * 0.3,
            },
        ];

        plebs.push(Agent::new(collateral));
    }

    plebs
}
fn create_sane_retail(count: u64) -> Vec<Agent> {
    let mut sane_retail = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..count {
        let collateral_amount = rng.gen_range(1_000.0..20_000.0);

        let collateral = vec![
            Collateral {
                asset_id: BTC_POOL,
                units: collateral_amount * 0.5,
            },
            Collateral {
                asset_id: ETH_POOL,
                units: collateral_amount * 0.4,
            },
            Collateral {
                asset_id: SOL_POOL,
                units: collateral_amount * 0.1,
            },
        ];

        sane_retail.push(Agent::new(collateral));
    }

    sane_retail
}

fn create_lps(count: u64) -> Vec<Agent> {
    let mut lps = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..count {
        let collateral_amount = rng.gen_range(1_000_000.0..50_000_000.0);

        let collateral = vec![
            Collateral {
                asset_id: USDC_POOL,
                units: collateral_amount,
            },
        ];

        lps.push(Agent::new(collateral));
    }

    lps
}

fn main() {
    let exchange = initialize_exchange();
    println!("Exchange initialized with {} pools", exchange.liquidity_pool.len());

    let whales = create_whales(5);
    let degens = create_degens(50);
    let plebs = create_plebs(200);
    let sane_retail = create_sane_retail(100);
    let lps = create_lps(1000);

    println!("Created {} whale agents", whales.len());
    println!("Created {} degen agents", degens.len());
    println!("Created {} pleb agents", plebs.len());
    println!("Created {} sane retail agents", sane_retail.len());
    println!("Created {} LP agents", lps.len());
    println!("Total agents: {}", whales.len() + degens.len() + plebs.len() + sane_retail.len() + lps.len());

    let mut all_agents = Vec::new();
    all_agents.extend(whales);
    all_agents.extend(degens);
    all_agents.extend(plebs);
    all_agents.extend(sane_retail);
    all_agents.extend(lps);

    println!("\n=== STARTING BLACK SWAN SIMULATION ===");
    println!("Simulating BTC -50%, other assets -75-85% price crash");
}
