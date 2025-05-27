import pandas as pd
import yfinance as yf
from advanced_trading_algorithms import AdvancedTradingAlgorithms
import matplotlib.pyplot as plt

def download_crypto_data(symbol='ETH-USD', period='6mo', interval='1d'):
    """Download cryptocurrency data"""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    return data

def main():
    # Initialize our advanced algorithms
    algo = AdvancedTradingAlgorithms()
    
    # Download sample data
    print("Downloading market data...")
    data = download_crypto_data()
    
    # Calculate RSI (needed for divergence)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    print("\nGenerating trading signals...")
    
    # 1. Get MACD signals
    macd_signals, macd, signal, hist, strength = algo.calculate_advanced_macd(data['Close'])
    print("MACD Signals generated:", len(macd_signals[macd_signals != 0]), "signals found")
    
    # 2. Get RSI divergence signals
    rsi_div_signals = algo.detect_rsi_divergence(data['Close'], rsi)
    print("RSI Divergence Signals generated:", len(rsi_div_signals[rsi_div_signals != 0]), "signals found")
    
    # 3. Calculate support/resistance levels
    sr_levels, sr_strengths = algo.calculate_ml_support_resistance(data['Close'])
    print("\nSupport/Resistance Levels:")
    for level, strength in zip(sr_levels, sr_strengths):
        print(f"Level: ${level:.2f} - Strength: {strength}")
    
    # 4. Get Bollinger Band signals
    bb_signals, upper, lower, vol_ratio = algo.calculate_dynamic_bollinger(data['Close'])
    print("\nBollinger Band Signals generated:", len(bb_signals[bb_signals != 0]), "signals found")
    
    # 5. Get VWAP signals
    vwap_signals, vwap, vol_profile = algo.calculate_vwap_signals(data['Close'], data['Volume'])
    print("VWAP Signals generated:", len(vwap_signals[vwap_signals != 0]), "signals found")
    
    # 6. Combine all signals with custom weights
    weights = [0.3, 0.2, 0.2, 0.3]  # Adjust these weights based on your preference
    combined_signals = algo.combine_signals(
        macd_signals,
        rsi_div_signals,
        bb_signals,
        vwap_signals,
        weights=weights
    )
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Price and signals
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='Price', color='blue', alpha=0.6)
    
    # Plot buy signals
    buy_signals = combined_signals[combined_signals > 0.5]
    sell_signals = combined_signals[combined_signals < -0.5]
    
    plt.scatter(buy_signals.index, data['Close'][buy_signals.index], 
                marker='^', color='green', s=100, label='Buy')
    plt.scatter(sell_signals.index, data['Close'][sell_signals.index], 
                marker='v', color='red', s=100, label='Sell')
    
    # Plot support/resistance levels
    for level in sr_levels:
        plt.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Trading Signals Analysis')
    plt.legend()
    
    # Signal strength
    plt.subplot(2, 1, 2)
    plt.plot(combined_signals.index, combined_signals, label='Signal Strength', color='purple')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
    plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3)
    plt.title('Combined Signal Strength')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print final recommendations
    last_signal = combined_signals.iloc[-1]
    print("\nCurrent Market Analysis:")
    if last_signal > 0.5:
        print("STRONG BUY SIGNAL")
    elif last_signal > 0.2:
        print("WEAK BUY SIGNAL")
    elif last_signal < -0.5:
        print("STRONG SELL SIGNAL")
    elif last_signal < -0.2:
        print("WEAK SELL SIGNAL")
    else:
        print("NEUTRAL - NO CLEAR SIGNAL")

if __name__ == "__main__":
    main()
