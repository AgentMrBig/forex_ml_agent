import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data, preprocess_data
import os

def plot_price_and_ma(df, vis_dir, days=30):
    # Get the last 'days' worth of data
    recent_data = df.last(f'{days}D')
    
    plt.figure(figsize=(15, 7))
    plt.plot(recent_data.index, recent_data['close'], label='Close Price')
    plt.plot(recent_data.index, recent_data['sma_10'], label='10-day MA')
    plt.title(f'USDJPY Price and 10-day MA (Last {days} days)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(vis_dir, f'price_and_ma_{days}days.png')
    plt.savefig(save_path)
    print(f"Saving price and MA plot to: {save_path}")
    plt.close()

def plot_price_movement(df, vis_dir):
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['close'])
    plt.title('USDJPY Price Movement')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(vis_dir, 'price_movement.png')
    plt.savefig(save_path)
    print(f"Saving price movement plot to: {save_path}")
    plt.close()

def plot_returns_distribution(df, vis_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['returns'].dropna(), kde=True)
    plt.title('Distribution of Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    save_path = os.path.join(vis_dir, 'returns_distribution.png')
    plt.savefig(save_path)
    print(f"Saving returns distribution plot to: {save_path}")
    plt.close()

def plot_feature_correlations(df, vis_dir):
    corr = df[['open', 'high', 'low', 'close', 'volume', 'returns', 'sma_10', 'sma_30', 'rsi', 'macd']].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    save_path = os.path.join(vis_dir, 'feature_correlations.png')
    plt.savefig(save_path)
    print(f"Saving feature correlations plot to: {save_path}")
    plt.close()

def plot_technical_indicators(df, vis_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    
    # Price and SMAs
    ax1.plot(df.index, df['close'], label='Close Price')
    ax1.plot(df.index, df['sma_10'], label='SMA 10')
    ax1.plot(df.index, df['sma_30'], label='SMA 30')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.set_title('Price and SMAs')

    # RSI
    ax2.plot(df.index, df['rsi'])
    ax2.set_ylabel('RSI')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='g', linestyle='--')
    ax2.set_title('RSI')

    # MACD
    ax3.plot(df.index, df['macd'], label='MACD')
    ax3.plot(df.index, df['macd_signal'], label='Signal Line')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.set_title('MACD')

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(vis_dir, 'technical_indicators.png')
    plt.savefig(save_path)
    print(f"Saving technical indicators plot to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Get the absolute path to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Load and preprocess data
    file_path = os.path.join(project_root, 'data', 'raw', 'USDJPY_1m_data.csv')
    raw_data = load_data(file_path)
    df = preprocess_data(raw_data)

    # Create visualizations directory if it doesn't exist
    vis_dir = os.path.join(project_root, 'data', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Visualization directory: {vis_dir}")

    # Generate plots
    plot_price_movement(df, vis_dir)
    plot_returns_distribution(df, vis_dir)
    plot_feature_correlations(df, vis_dir)
    plot_technical_indicators(df, vis_dir)
    plot_price_and_ma(df, vis_dir, days=30)  # You can adjust the number of days

    print("Visualizations have been saved in the 'data/visualizations' directory.")
    print("Contents of the visualization directory:")
    print(os.listdir(vis_dir))