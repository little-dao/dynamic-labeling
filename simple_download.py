"""
Simple script to download stock data without rate limiting issues
"""

import yfinance as yf
import pandas as pd
import time
import pickle
import os

def download_single_stock(symbol="AAPL", start="2020-01-01", end="2024-01-01"):
    """Download data for a single stock with retry logic"""
    max_retries = 3
    delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {symbol} (attempt {attempt + 1}/{max_retries})...")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval="1d")
            
            if len(df) > 0:
                # Clean column names
                df.columns = [col.replace(' ', '_') for col in df.columns]
                df = df.dropna()
                print(f"✅ Successfully downloaded {len(df)} records for {symbol}")
                return df
            else:
                print(f"⚠️ No data returned for {symbol}")
                
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
    print(f"❌ Failed to download {symbol} after {max_retries} attempts")
    return None

def main():
    """Download data for a few key stocks"""
    os.makedirs("data", exist_ok=True)
    
    symbols = ["AAPL", "MSFT"]  # Just 2 symbols to start
    data = {}
    
    for symbol in symbols:
        df = download_single_stock(symbol)
        if df is not None:
            data[symbol] = df
        
        # Wait between symbols to avoid rate limiting
        if symbol != symbols[-1]:  # Don't wait after the last symbol
            print("Waiting 3 seconds before next download...")
            time.sleep(3)
    
    if data:
        # Save the raw downloaded data
        with open("data/raw_data.pkl", "wb") as f:
            pickle.dump(data, f)
        
        print(f"\n🎉 Successfully downloaded {len(data)} symbols")
        print("Raw data saved to data/raw_data.pkl")
        
        # Show some info
        for symbol, df in data.items():
            print(f"{symbol}: {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")
            print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return True
    else:
        print("❌ No data downloaded successfully")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Data download complete! You can now run the main pipeline.")
    else:
        print("\n❌ Data download failed. Please try again later.")
