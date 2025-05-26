\
import argparse
import pandas as pd
from openbb import obb
from pathlib import Path
from datetime import datetime
import os # Ensure os is imported at the top
from dotenv import load_dotenv # Ensure load_dotenv is imported at the top

# Define the root directory for processed data
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "crypto" / "processed"
IN_SAMPLE_END_DATE = "2024-05-24"

def fetch_data(symbol: str, start_date: str, end_date: str, interval: str, provider: str = "fmp", fmp_api_key: str | None = None):
    """
    Fetches historical OHLCV data for a given symbol and date range.
    """
    print(f"Fetching {interval} data for {symbol} from {start_date} to {end_date} using {provider}...")
    
    if provider == "fmp":
        try:
            obb_symbol = symbol.replace("-", "").upper() # FMP typically uses "BTCUSD"
            fmp_interval_map = {
                "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1hour", "4h": "4hour", "1d": "1day"
            }
            api_interval = fmp_interval_map.get(interval, interval)
            
            # Diagnostic: Check what os.getenv() sees inside fetch_data, before using fmp_api_key
            current_env_fmp_key = os.getenv("OPENBB_FMP_API_KEY")
            if current_env_fmp_key:
                print(f"[fetch_data] os.getenv('OPENBB_FMP_API_KEY') reports: '{current_env_fmp_key[:4]}...{current_env_fmp_key[-4:]}'")
            else:
                print("[fetch_data] os.getenv('OPENBB_FMP_API_KEY') reports: Not found")

            credentials = None
            if fmp_api_key: # Key explicitly passed to fetch_data from main
                credentials = {"fmp_api_key": fmp_api_key}
                print(f"[fetch_data] Using explicitly passed fmp_api_key: '{fmp_api_key[:4]}...{fmp_api_key[-4:]}'")
                print(f"[fetch_data] Passing credentials to FMP call: {{'fmp_api_key': '{fmp_api_key[:4]}...{fmp_api_key[-4:]}'}}")
            else:
                # This case should ideally not happen if main always tries to load and pass the key.
                # If it does, OpenBB would rely on its own environment check.
                print("[fetch_data] No explicit fmp_api_key passed to function; OpenBB will rely on its global/env config.")

            data = obb.crypto.price.historical(
                symbol=obb_symbol,
                start_date=start_date,
                end_date=end_date,
                interval=api_interval,
                provider="fmp",
                credentials=credentials # Pass credentials if fmp_api_key was available, otherwise None
            )
            if data and not data.to_df().empty:
                df = data.to_df()
                print(f"Successfully fetched {len(df)} records for {symbol} using FMP with symbol {obb_symbol}.")
                return df
            else:
                print(f"No data returned for {symbol} with provider FMP and symbol format {obb_symbol}.")
                print("Trying with provider 'yfinance' as fallback...")
                return fetch_with_yfinance(symbol, start_date, end_date, interval)
        except Exception as e_fmp:
            print(f"Error fetching data for {symbol} with provider FMP: {e_fmp}.")
            print("Trying with provider 'yfinance' as fallback...")
            return fetch_with_yfinance(symbol, start_date, end_date, interval)

    elif provider == "yfinance":
        return fetch_with_yfinance(symbol, start_date, end_date, interval)
        
    else: # Other providers (not detailed with API key handling here
        try:
            obb_symbol = symbol.replace("-", "").upper()
            data = obb.crypto.price.historical(
                symbol=obb_symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval, # Assuming direct interval mapping for other providers
                provider=provider
            )
            if data and not data.to_df().empty:
                df = data.to_df()
                print(f"Successfully fetched {len(df)} records for {symbol} using {provider} with symbol {obb_symbol}.")
                return df
            else:
                print(f"No data returned for {symbol} with provider {provider}. No further fallback.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching data for {symbol} ({obb_symbol}) with provider {provider}: {e}. No further fallback.")
            return pd.DataFrame()


def fetch_with_yfinance(symbol: str, start_date: str, end_date: str, interval: str):
    """
    Helper function to fetch data using yfinance.
    """
    print(f"Attempting to fetch {interval} data for {symbol} from {start_date} to {end_date} using yfinance...")
    try:
        obb_symbol = symbol # yfinance typically uses "BTC-USD"
        # yfinance intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # No specific interval mapping needed for yfinance as it's quite flexible
        
        data = obb.crypto.price.historical(
            symbol=obb_symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            provider="yfinance"
        )
        if data and not data.to_df().empty:
            df = data.to_df()
            print(f"Successfully fetched {len(df)} records for {symbol} using yfinance.")
            return df
        else:
            print(f"No data returned for {symbol} with provider yfinance.")
            return pd.DataFrame()
    except Exception as e_yf:
        print(f"Error fetching data for {symbol} with provider yfinance: {e_yf}.")
        return pd.DataFrame()

def save_partitioned_parquet(df: pd.DataFrame, symbol: str, interval: str, base_path: Path):
    """
    Saves the DataFrame as partitioned Parquet files.
    Partitions: symbol / interval / year / month / data.parquet
    """
    if df.empty:
        print(f"No data to save for {symbol} at {interval} interval.")
        return

    # Ensure the index is a DatetimeIndex for partitioning
    if not isinstance(df.index, pd.DatetimeIndex):
        # Assuming the index is date but not yet DatetimeIndex, try to convert
        # If 'date' is a column: df.set_index(pd.to_datetime(df['date']), inplace=True)
        # If index is object: df.index = pd.to_datetime(df.index)
        # OpenBB usually returns with DatetimeIndex, but good to be robust
        try:
            df.index = pd.to_datetime(df.index, utc=True) # Ensure timezone aware for consistency
        except Exception as e:
            print(f"Failed to convert index to DatetimeIndex for {symbol}: {e}. Skipping save.")
            return
            
    # Sanitize symbol for directory name (e.g., BTC-USD -> BTC_USD)
    sanitized_symbol = symbol.replace("/", "_").replace("-", "_")

    df['year'] = df.index.year
    df['month'] = df.index.month

    for (year, month), group in df.groupby(['year', 'month']):
        partition_path = base_path / sanitized_symbol / interval / str(year) / f"{month:02d}"
        partition_path.mkdir(parents=True, exist_ok=True)
        
        file_path = partition_path / "data.parquet"
        try:
            # Drop temporary columns before saving
            group_to_save = group.drop(columns=['year', 'month'])
            group_to_save.to_parquet(file_path, engine='pyarrow', index=True)
            print(f"Saved data to {file_path}")
        except Exception as e:
            print(f"Error saving data to {file_path}: {e}")

def main():
    # --- Start of .env loading logic ---
    project_root = Path(__file__).resolve().parent.parent.parent
    dotenv_path = project_root / ".env"
    print(f"[main] Looking for .env file at: {dotenv_path}")
    if dotenv_path.exists():
        print(f"[main] Found .env file. Attempting to load with verbose=True, override=True.")
        # load_dotenv will not override existing environment variables by default.
        # override=True ensures that .env values take precedence.
        # verbose=True will print messages to stdout about what it's doing.
        loaded_successfully = load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
        print(f"[main] load_dotenv call completed. Result (True if .env was loaded): {loaded_successfully}")
        
        # Check immediately after loading
        key_after_load = os.getenv("OPENBB_FMP_API_KEY")
        if key_after_load:
            print(f"[main] OPENBB_FMP_API_KEY from os.getenv after load_dotenv: '{key_after_load[:4]}...{key_after_load[-4:]}'")
        else:
            print("[main] OPENBB_FMP_API_KEY NOT FOUND in os.getenv after load_dotenv call.")
    else:
        print(f"[main] .env file not found at: {dotenv_path}")
    # --- End of .env loading logic ---

    parser = argparse.ArgumentParser(description="Fetch and store cryptocurrency OHLCV data.")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC-USD,ETH-USD",
        help="Comma-separated list of crypto symbols (e.g., BTC-USD,ETH-USD)."
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (e.g., 1m, 5m, 15m, 1h, 1d)."
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2010-01-01",
        help="Start date for data fetching (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2024-07-01", # Updated to a more recent default for testing
        help="End date for data fetching (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="fmp", 
        help="Data provider (e.g., fmp, yfinance)."
    )
    # Removed --fmp_api_key argument from parser

    args = parser.parse_args()
    symbols_list = [s.strip() for s in args.symbols.split(',')]
    
    # Get the API key from environment (should be set by load_dotenv if .env exists)
    fmp_api_key_to_use = os.getenv("OPENBB_FMP_API_KEY")

    if fmp_api_key_to_use:
        print(f"[main] FMP API Key to be used for fetch_data calls: '{fmp_api_key_to_use[:4]}...{fmp_api_key_to_use[-4:]}'")
    else:
        print("[main] FMP API Key was NOT loaded from environment for fetch_data calls (it's None or empty).")

    for symbol in symbols_list:
        print(f"Processing symbol: {symbol}")
        
        full_data_df = fetch_data(
            symbol, 
            args.start_date, 
            args.end_date, 
            args.interval, 
            args.provider,
            fmp_api_key=fmp_api_key_to_use # Pass the API key loaded from .env
        )

        if full_data_df.empty:
            print(f"No data fetched for {symbol}. Skipping.")
            continue
        
        # Ensure index is datetime and timezone-aware (UTC)
        if not isinstance(full_data_df.index, pd.DatetimeIndex):
            full_data_df.index = pd.to_datetime(full_data_df.index)
        if full_data_df.index.tz is None:
            full_data_df.index = full_data_df.index.tz_localize('UTC')
        else:
            full_data_df.index = full_data_df.index.tz_convert('UTC')

        # Split data
        # In-sample: start_date to IN_SAMPLE_END_DATE
        # Out-of-sample: day after IN_SAMPLE_END_DATE to end_date
        
        in_sample_end_dt = pd.to_datetime(IN_SAMPLE_END_DATE).tz_localize('UTC')
        
        # In-sample data
        in_sample_df = full_data_df[full_data_df.index <= in_sample_end_dt]
        
        # Out-of-sample data
        # Start OOS data from the day after in_sample_end_dt
        out_of_sample_start_dt = in_sample_end_dt + pd.Timedelta(days=1) 
        # Ensure OOS start is not before the overall start_date if IN_SAMPLE_END_DATE is very early
        actual_oos_start_dt = max(pd.to_datetime(args.start_date).tz_localize('UTC'), out_of_sample_start_dt)
        
        out_of_sample_df = full_data_df[full_data_df.index >= actual_oos_start_dt]

        print(f"In-sample data for {symbol}: {len(in_sample_df)} records from {in_sample_df.index.min()} to {in_sample_df.index.max()}")
        print(f"Out-of-sample data for {symbol}: {len(out_of_sample_df)} records from {out_of_sample_df.index.min()} to {out_of_sample_df.index.max()}")

        # Save partitioned data
        # Note: The problem description implies saving all data (2017-2025) partitioned.
        # The split is for conceptual understanding/later use in backtesting.
        # The script will save the entire fetched range, partitioned.
        # The data_loader will then be responsible for loading specific ranges (in-sample/out-of-sample).
        
        print(f"Saving all fetched data for {symbol} (will be partitioned)...")
        save_partitioned_parquet(full_data_df, symbol, args.interval, DATA_ROOT)

    print("Data fetching and storage process complete.")

if __name__ == "__main__":
    # Create the root data directory if it doesn't exist
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    main()
