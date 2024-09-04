import pandas as pd
import numpy as np
from typing import Tuple
import os
from datetime import timedelta

def load_data(file_path: str) -> pd.DataFrame:
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print("Columns in the CSV file:", df.columns.tolist())
    print("\nFirst few rows of raw data:")
    print(df.head())

    try:
        # Check if 'Date' and 'Timestamp' columns exist
        if 'Date' in df.columns and 'Timestamp' in df.columns:
            df['Date'] = df['Date'].astype(str)
            df['Timestamp'] = df['Timestamp'].astype(str)
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], format='%Y%m%d %H:%M:%S')
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            raise ValueError("Could not find 'Date' and 'Timestamp' columns or 'datetime' column")

        df.set_index('datetime', inplace=True)
        df.drop(['Date', 'Timestamp'], axis=1, errors='ignore', inplace=True)
    except Exception as e:
        print(f"Error processing datetime: {e}")
        print("DataFrame info:")
        print(df.info())
        raise

    df.columns = df.columns.str.lower()
    
    print("\nProcessed data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)

    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Use only the last month of data
    end_date = df.index.max()
    start_date = end_date - timedelta(days=30)
    df = df[df.index >= start_date]

    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Add basic technical indicators
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['rsi'] = calculate_rsi(df['close'])
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def split_data(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_ratio)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]
    return train_data, test_data

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'USDJPY_1m_data.csv')
    
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        print("Current working directory:", os.getcwd())
        print("Contents of the data/raw directory:")
        raw_dir = os.path.join(current_dir, '..', '..', 'data', 'raw')
        if os.path.exists(raw_dir):
            print(os.listdir(raw_dir))
        else:
            print(f"The directory {raw_dir} does not exist.")
        exit(1)
    
    raw_data = load_data(file_path)
    preprocessed_data = preprocess_data(raw_data)
    train_data, test_data = split_data(preprocessed_data)
    
    print("\nRaw data shape:", raw_data.shape)
    print("Preprocessed data shape:", preprocessed_data.shape)
    print("Training data shape:", train_data.shape)
    print("Testing data shape:", test_data.shape)
    print("\nPreprocessed data columns:", preprocessed_data.columns.tolist())
    print("\nFirst few rows of preprocessed data:")
    print(preprocessed_data.head())