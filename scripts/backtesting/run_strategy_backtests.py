import argparse
import json
import pandas as pd
import numpy as np # Added import
from pathlib import Path
from datetime import datetime

# Adjust import paths based on project structure
# Assuming the script is in scripts/backtesting/ and src/ is at the project root
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader
from research.strategy_prototypes.entropy_stat_arb import EntropyDrivenStatArb
from src.strategies.base_strategy import BaseStrategy # For type hinting if needed

# Define constants
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "backtests"
IN_SAMPLE_START_DATE = "2017-01-01"
IN_SAMPLE_END_DATE = "2024-05-24" # As per conversation summary

def prepare_results_for_json(results: dict) -> dict:
    """Convert backtest results to JSON-serializable format."""
    serializable_results = {}
    
    for key, value in results.items():
        if key == 'trades':
            # Convert trade objects to dictionaries
            serializable_results[key] = []
            for trade in value:
                trade_dict = trade.__dict__.copy()
                # Convert timestamps to ISO format
                if 'entry_time' in trade_dict:
                    trade_dict['entry_time'] = trade_dict['entry_time'].isoformat()
                if 'exit_time' in trade_dict:
                    trade_dict['exit_time'] = trade_dict['exit_time'].isoformat()
                # Convert timedelta to string
                if 'duration' in trade_dict:
                    trade_dict['duration'] = str(trade_dict['duration'])
                # Convert signal enum to string
                if 'signal' in trade_dict:
                    trade_dict['signal'] = trade_dict['signal'].name if hasattr(trade_dict['signal'], 'name') else str(trade_dict['signal'])
                serializable_results[key].append(trade_dict)
        elif key == 'signals':
            # Convert signals history
            serializable_results[key] = []
            for timestamp, signal in value:
                serializable_results[key].append({
                    'timestamp': timestamp.isoformat(),
                    'signal': signal.name if hasattr(signal, 'name') else str(signal)
                })
        elif key == 'equity_curve' and value is not None:
            # Convert pandas series to list of dictionaries
            serializable_results[key] = [
                {'timestamp': idx.isoformat(), 'value': val}
                for idx, val in value.items()
            ]
        elif key == 'metrics' and value is not None:
            # Convert metrics with potential numpy types
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.integer, np.floating, np.bool_)):
                    serializable_results[key][k] = v.item()
                elif isinstance(v, pd.Timestamp):
                    serializable_results[key][k] = v.isoformat()
                else:
                    serializable_results[key][k] = v
        elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            # Portfolio value and returns lists
            serializable_results[key] = value
        else:
            try:
                json.dumps(value)  # Test if it's JSON serializable
                serializable_results[key] = value
            except (TypeError, ValueError):
                serializable_results[key] = str(value)
    
    return serializable_results

def run_backtest(
    strategy_name: str,
    strategy_class: type[BaseStrategy],
    strategy_params: dict,
    asset_pair: tuple[str, str],
    interval: str,
    start_date: str,
    end_date: str,
    results_dir: Path
):
    """Runs a backtest for a given strategy and saves the results."""
    print(f"Starting backtest for strategy: {strategy_name} on pair {asset_pair} from {start_date} to {end_date}")

    # 1. Load Data for both assets
    loader = DataLoader()
    symbol_x, symbol_y = asset_pair
    
    # Convert symbols to the format used in data storage (hyphens to underscores)
    symbol_x_file = symbol_x.replace('-', '_')
    symbol_y_file = symbol_y.replace('-', '_')

    print(f"Loading data for asset X: {symbol_x} (file: {symbol_x_file})")
    data_x = loader.load_partitioned_crypto_data(
        symbol=symbol_x_file,
        interval=interval,
        start_date=start_date,
        end_date=end_date
    )
    if data_x.empty:
        print(f"No data found for asset X: {symbol_x} (tried {symbol_x_file}). Skipping backtest.")
        return

    print(f"Loading data for asset Y: {symbol_y} (file: {symbol_y_file})")
    data_y = loader.load_partitioned_crypto_data(
        symbol=symbol_y_file,
        interval=interval,
        start_date=start_date,
        end_date=end_date
    )
    if data_y.empty:
        print(f"No data found for asset Y: {symbol_y} (tried {symbol_y_file}). Skipping backtest.")
        return

    # Check data alignment
    print(f"Data X shape: {data_x.shape}, date range: {data_x.index.min()} to {data_x.index.max()}")
    print(f"Data Y shape: {data_y.shape}, date range: {data_y.index.min()} to {data_y.index.max()}")

    # 2. Create combined data dictionary for the strategy  
    # Use the original symbol names (with hyphens) as keys since that's what the strategy expects
    market_data = {symbol_x: data_x, symbol_y: data_y}

    # 3. Initialize Strategy
    strategy = strategy_class(name=strategy_name, parameters=strategy_params)
    print(f"Initialized strategy: {strategy.name}")
    
    # Print strategy info
    if hasattr(strategy, 'get_strategy_info'):
        info = strategy.get_strategy_info()
        print("Strategy Configuration:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    # 4. Run Backtest
    print("Running backtest...")
    try:
        # For pair strategies, we need to pass the dictionary and the primary symbol
        results = strategy.backtest(
            data=market_data,
            symbol=symbol_x  # Primary trading symbol
        )
        
        print(f"Backtest completed successfully!")
        print(f"Final portfolio value: ${results['portfolio_value'][-1]:,.2f}")
        print(f"Total return: {((results['portfolio_value'][-1] / results['portfolio_value'][0]) - 1) * 100:.2f}%")
        print(f"Number of trades: {len(results['trades'])}")

        # 5. Save Results
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"{strategy_name}_{symbol_x}_{symbol_y}_{interval}_{timestamp}.json"
        results_path = results_dir / results_filename
        
        # Prepare results for JSON serialization
        serializable_results = prepare_results_for_json(results)
        
        # Add metadata
        serializable_results['metadata'] = {
            'strategy_name': strategy_name,
            'asset_pair': asset_pair,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date,
            'timestamp': timestamp,
            'strategy_params': strategy_params
        }
        
        # Save to JSON
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def main():
    parser = argparse.ArgumentParser(description="Run strategy backtests.")
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="EntropyDrivenStatArb", 
        help="Name of the strategy to run."
    )
    parser.add_argument(
        "--pair", 
        type=str, 
        default="BTC-USD,ETH-USD", 
        help="Comma-separated asset pair (e.g., BTC-USD,ETH-USD)."
    )
    parser.add_argument(
        "--interval", 
        type=str, 
        default="1d", 
        help="Data interval (e.g., 1h, 1d)."
    )
    parser.add_argument(
        "--start_date", 
        type=str, 
        default=IN_SAMPLE_START_DATE, 
        help="Start date for backtest (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end_date", 
        type=str, 
        default=IN_SAMPLE_END_DATE, 
        help="End date for backtest (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to save backtest results."
    )
    # Strategy-specific parameters
    parser.add_argument("--window_entropy", type=int, default=20)
    parser.add_argument("--window_transfer_entropy", type=int, default=20)  
    parser.add_argument("--te_lag", type=int, default=1)
    parser.add_argument("--threshold_composite_score", type=float, default=1.0)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0001)

    args = parser.parse_args()

    # Parse asset pair
    asset_list = [s.strip() for s in args.pair.split(',')]
    if len(asset_list) != 2:
        raise ValueError("Asset pair must consist of two symbols separated by a comma.")
    asset_pair_tuple = tuple(asset_list)

    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if args.strategy == "EntropyDrivenStatArb":
        strategy_cls = EntropyDrivenStatArb
        strategy_params = {
            'asset_pair': asset_pair_tuple,
            'window_entropy': args.window_entropy,
            'window_transfer_entropy': args.window_transfer_entropy,
            'te_lag': args.te_lag,
            'threshold_composite_score': args.threshold_composite_score,
            'n_bins': args.n_bins,
            'transaction_cost': args.transaction_cost,
            'slippage': args.slippage,
        }
    else:
        print(f"Strategy {args.strategy} not recognized.")
        return

    print(f"Running {args.strategy} backtest on {asset_pair_tuple} from {args.start_date} to {args.end_date}")
    run_backtest(
        strategy_name=args.strategy,
        strategy_class=strategy_cls,
        strategy_params=strategy_params,
        asset_pair=asset_pair_tuple,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        results_dir=results_path
    )

if __name__ == "__main__":
    main()
