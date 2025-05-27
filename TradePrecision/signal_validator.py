import pandas as pd
import numpy as np
from datetime import datetime

def validate_signals(df, quality_threshold=60, min_confirmation=2):
    """
    Validate trading signals based on multiple criteria
    
    Parameters:
    df (pd.DataFrame): DataFrame with signals and indicators
    quality_threshold (int): Minimum quality score for a valid signal (0-100)
    min_confirmation (int): Minimum number of indicators confirming the signal
    
    Returns:
    pd.DataFrame: DataFrame with validated signals
    """
    df = df.copy()
    
    # Initialize signal quality column
    if 'signal_quality' not in df.columns:
        df['signal_quality'] = 0
    
    # Filter signals below quality threshold
    low_quality_mask = (df['signal'] != 'neutral') & (df['signal_strength'] < quality_threshold)
    df.loc[low_quality_mask, 'signal'] = 'neutral'
    
    # Check for trend confirmation
    # The ADX is already added in technical_indicators.py add_all_indicators function
    
    # Check for trend strength - weak trends are less reliable
    weak_trend_mask = (df['signal'] != 'neutral') & (df['adx'] < 20)
    df.loc[weak_trend_mask, 'signal_quality'] *= 0.8  # Reduce quality for weak trends
    
    # Check for overbought/oversold conditions with RSI
    if 'rsi' in df.columns:
        # Buy signals in extremely oversold conditions are stronger
        strong_buy_mask = (df['signal'] == 'buy') & (df['rsi'] < 30)
        df.loc[strong_buy_mask, 'signal_quality'] += 15
        
        # Sell signals in extremely overbought conditions are stronger
        strong_sell_mask = (df['signal'] == 'sell') & (df['rsi'] > 70)
        df.loc[strong_sell_mask, 'signal_quality'] += 15
        
        # Buy signals in overbought conditions are weaker
        weak_buy_mask = (df['signal'] == 'buy') & (df['rsi'] > 65)
        df.loc[weak_buy_mask, 'signal_quality'] -= 15
        
        # Sell signals in oversold conditions are weaker
        weak_sell_mask = (df['signal'] == 'sell') & (df['rsi'] < 35)
        df.loc[weak_sell_mask, 'signal_quality'] -= 15
    
    # Check for bollinger band validation
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        # Buy signals near lower band are stronger
        bb_buy_mask = (df['signal'] == 'buy') & (df['close'] <= df['bb_lower'] * 1.01)
        df.loc[bb_buy_mask, 'signal_quality'] += 15
        
        # Sell signals near upper band are stronger
        bb_sell_mask = (df['signal'] == 'sell') & (df['close'] >= df['bb_upper'] * 0.99)
        df.loc[bb_sell_mask, 'signal_quality'] += 15
    
    # Check for volume confirmation
    if 'volume_ratio' in df.columns:
        # Signals with above-average volume are stronger
        volume_confirm_mask = (df['signal'] != 'neutral') & (df['volume_ratio'] > 1.5)
        df.loc[volume_confirm_mask, 'signal_quality'] += 10
        
        # Signals with low volume are weaker
        low_volume_mask = (df['signal'] != 'neutral') & (df['volume_ratio'] < 0.7)
        df.loc[low_volume_mask, 'signal_quality'] -= 10
    
    # Ensure signal quality is within bounds
    df['signal_quality'] = df['signal_quality'].clip(0, 100)
    
    # Filter out low quality signals based on threshold
    final_filter_mask = (df['signal'] != 'neutral') & (df['signal_quality'] < quality_threshold)
    df.loc[final_filter_mask, 'signal'] = 'neutral'
    
    return df

def enhance_signal_dataframe(signals_df):
    """
    Enhance the signals DataFrame with additional information for display
    
    Parameters:
    signals_df (pd.DataFrame): DataFrame with signals
    
    Returns:
    pd.DataFrame: Enhanced DataFrame with additional columns
    """
    if signals_df.empty:
        return signals_df
    
    df = signals_df.copy()
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate time since signal
    current_time = datetime.now()
    if 'timestamp' in df.columns:
        df['time_since_signal'] = (current_time - df['timestamp']).dt.total_seconds() / 60  # minutes
    
    # Calculate price change since signal
    if 'price' in df.columns and 'entry_point' in df.columns:
        df['price_change'] = ((df['price'] / df['entry_point']) - 1) * 100
    
    # Calculate profit potential
    if 'entry_point' in df.columns and 'take_profit' in df.columns:
        # For buy signals, profit = (take_profit / entry_point) - 1
        buy_mask = df['signal_type'] == 'buy'
        df.loc[buy_mask, 'profit_potential'] = ((df.loc[buy_mask, 'take_profit'] / df.loc[buy_mask, 'entry_point']) - 1) * 100
        
        # For sell signals, profit = (entry_point / take_profit) - 1
        sell_mask = df['signal_type'] == 'sell'
        df.loc[sell_mask, 'profit_potential'] = ((df.loc[sell_mask, 'entry_point'] / df.loc[sell_mask, 'take_profit']) - 1) * 100
    
    # Risk calculation
    if 'entry_point' in df.columns and 'stop_loss' in df.columns:
        # For buy signals, risk = 1 - (stop_loss / entry_point)
        buy_mask = df['signal_type'] == 'buy'
        df.loc[buy_mask, 'risk'] = (1 - (df.loc[buy_mask, 'stop_loss'] / df.loc[buy_mask, 'entry_point'])) * 100
        
        # For sell signals, risk = (stop_loss / entry_point) - 1
        sell_mask = df['signal_type'] == 'sell'
        df.loc[sell_mask, 'risk'] = ((df.loc[sell_mask, 'stop_loss'] / df.loc[sell_mask, 'entry_point']) - 1) * 100
    
    # Calculate risk/reward ratio
    if 'risk' in df.columns and 'profit_potential' in df.columns:
        df['risk_reward'] = df['profit_potential'].abs() / df['risk'].abs()
    
    # Round numeric columns for display
    numeric_cols = ['price', 'entry_point', 'stop_loss', 'take_profit', 
                    'price_change', 'profit_potential', 'risk', 'risk_reward']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)
    
    # Add signal quality if not present
    if 'signal_quality' not in df.columns and 'signal_strength' in df.columns:
        df['signal_quality'] = df['signal_strength']
    
    return df

def get_signal_validation_metrics(signal):
    """
    Get validation metrics for a single signal
    
    Parameters:
    signal (dict): Signal dictionary
    
    Returns:
    dict: Validation metrics
    """
    metrics = {
        "quality_score": signal.get("signal_quality", signal.get("signal_strength", 0)),
        "confirmation_count": 0,
        "total_indicators": 5,  # RSI, MACD, MA, BB, Volume
        "risk_reward_ratio": 0,
        "market_condition": "Unknown"
    }
    
    # Calculate risk/reward ratio
    entry = signal.get("entry_point", 0)
    stop = signal.get("stop_loss", 0)
    target = signal.get("take_profit", 0)
    
    if entry > 0 and stop > 0 and target > 0:
        if signal.get("signal_type") == "buy":
            risk = (entry - stop) / entry
            reward = (target - entry) / entry
        else:  # sell
            risk = (stop - entry) / entry
            reward = (entry - target) / entry
        
        if risk > 0:
            metrics["risk_reward_ratio"] = reward / risk
    
    # Determine market condition based on signal type and strength
    signal_type = signal.get("signal_type", "neutral")
    signal_strength = signal.get("signal_strength", 0)
    
    if signal_type == "buy" and signal_strength > 80:
        metrics["market_condition"] = "Strong Bullish"
    elif signal_type == "buy" and signal_strength > 60:
        metrics["market_condition"] = "Bullish"
    elif signal_type == "buy":
        metrics["market_condition"] = "Mildly Bullish"
    elif signal_type == "sell" and signal_strength > 80:
        metrics["market_condition"] = "Strong Bearish"
    elif signal_type == "sell" and signal_strength > 60:
        metrics["market_condition"] = "Bearish"
    elif signal_type == "sell":
        metrics["market_condition"] = "Mildly Bearish"
    else:
        metrics["market_condition"] = "Neutral"
    
    # Estimate number of confirming indicators from signal strength
    # Each indicator contributes 20 points to strength (5 indicators * 20 = 100 max)
    metrics["confirmation_count"] = min(5, int(signal_strength / 20))
    
    return metrics

def validate_signal_quality(symbol, signal_type, indicators):
    """
    Validate signal quality based on multiple indicators
    
    Parameters:
    symbol (str): Cryptocurrency symbol
    signal_type (str): Signal type ('buy', 'sell', 'neutral')
    indicators (dict): Dictionary with indicator values
    
    Returns:
    dict: Validation results with quality score and confirmation indicators
    """
    quality_score = 0
    confirmations = []
    warnings = []
    
    # Check RSI
    rsi = indicators.get('rsi', 50)
    if signal_type == 'buy' and rsi < 40:
        quality_score += 20
        confirmations.append("RSI indicates oversold conditions")
    elif signal_type == 'buy' and rsi > 60:
        quality_score -= 10
        warnings.append("RSI is not in buy zone")
    elif signal_type == 'sell' and rsi > 60:
        quality_score += 20
        confirmations.append("RSI indicates overbought conditions")
    elif signal_type == 'sell' and rsi < 40:
        quality_score -= 10
        warnings.append("RSI is not in sell zone")
    
    # Check MACD
    macd = indicators.get('macd', 0)
    macd_signal = indicators.get('macd_signal', 0)
    macd_hist = indicators.get('macd_hist', 0)
    
    if signal_type == 'buy' and macd > macd_signal and macd_hist > 0:
        quality_score += 20
        confirmations.append("MACD confirms bullish momentum")
    elif signal_type == 'sell' and macd < macd_signal and macd_hist < 0:
        quality_score += 20
        confirmations.append("MACD confirms bearish momentum")
    
    # Check Moving Averages
    ma_fast = indicators.get('ma_fast', 0)
    ma_slow = indicators.get('ma_slow', 0)
    close = indicators.get('close', 0)
    
    if signal_type == 'buy' and ma_fast > ma_slow and close > ma_fast:
        quality_score += 20
        confirmations.append("Price above MAs in uptrend")
    elif signal_type == 'sell' and ma_fast < ma_slow and close < ma_fast:
        quality_score += 20
        confirmations.append("Price below MAs in downtrend")
    
    # Check Bollinger Bands
    bb_upper = indicators.get('bb_upper', 0)
    bb_lower = indicators.get('bb_lower', 0)
    
    if signal_type == 'buy' and close <= bb_lower * 1.01:
        quality_score += 20
        confirmations.append("Price near lower Bollinger Band")
    elif signal_type == 'sell' and close >= bb_upper * 0.99:
        quality_score += 20
        confirmations.append("Price near upper Bollinger Band")
    
    # Check Volume
    volume = indicators.get('volume', 0)
    volume_avg = indicators.get('volume_ma', 0)
    
    if volume > volume_avg * 1.5:
        quality_score += 20
        confirmations.append("Above average volume confirms signal")
    elif volume < volume_avg * 0.7:
        quality_score -= 10
        warnings.append("Low volume reduces signal reliability")
    
    # Ensure quality score is within bounds
    quality_score = max(0, min(100, quality_score))
    
    return {
        "symbol": symbol,
        "signal_type": signal_type,
        "quality_score": quality_score,
        "confirmations": confirmations,
        "warnings": warnings,
        "is_valid": quality_score >= 60
    }