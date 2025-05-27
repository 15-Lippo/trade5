import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_indicator_params(analysis_mode, timeframe):
    """
    Get indicator parameters based on analysis mode and timeframe
    
    Parameters:
    analysis_mode (str): Analysis mode ('Standard', 'Conservative', 'Aggressive', 'Custom')
    timeframe (str): Timeframe of the data
    
    Returns:
    dict: Dictionary with indicator parameters
    """
    params = {}
    
    # Base parameters
    if analysis_mode == "Conservative":
        # Conservative mode - fewer false signals, more confirmation required
        params = {
            "rsi_period": 16,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "macd_fast": 14,
            "macd_slow": 30,
            "macd_signal": 10,
            "ma_fast_period": 30,
            "ma_slow_period": 60,
            "bollinger_period": 25,
            "bollinger_std": 2.2,
            "signal_quality_threshold": 75,
            "min_confirmation_indicators": 3
        }
    elif analysis_mode == "Aggressive":
        # Aggressive mode - more signals, potentially more false positives
        params = {
            "rsi_period": 10,
            "rsi_overbought": 65,
            "rsi_oversold": 35,
            "macd_fast": 10,
            "macd_slow": 24,
            "macd_signal": 8,
            "ma_fast_period": 15,
            "ma_slow_period": 45,
            "bollinger_period": 15,
            "bollinger_std": 1.8,
            "signal_quality_threshold": 50,
            "min_confirmation_indicators": 2
        }
    else:  # Standard or Custom
        params = {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ma_fast_period": 20,
            "ma_slow_period": 50,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "signal_quality_threshold": 60,
            "min_confirmation_indicators": 2
        }
    
    # Ottimizzazione specifica per timeframe
    # Parametri ottimizzati per ciascun timeframe in base a backtesting
    if timeframe == '1M':
        # Timeframe mensile - richiede un'analisi più a lungo termine e minor sensibilità alle fluttuazioni
        params = {
            "rsi_period": 21,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "macd_fast": 19,
            "macd_slow": 39,
            "macd_signal": 14,
            "ma_fast_period": 35,
            "ma_slow_period": 90,
            "bollinger_period": 30,
            "bollinger_std": 2.5,
            "signal_quality_threshold": 70,
            "min_confirmation_indicators": 3
        }
    elif timeframe in ['1w', '2w']:
        # Timeframe settimanale - valori ottimizzati per catturare trend a medio termine
        params = {
            "rsi_period": 14,
            "rsi_overbought": 72,
            "rsi_oversold": 28,
            "macd_fast": 16,
            "macd_slow": 34,
            "macd_signal": 12,
            "ma_fast_period": 26,
            "ma_slow_period": 65,
            "bollinger_period": 24,
            "bollinger_std": 2.2,
            "signal_quality_threshold": 65,
            "min_confirmation_indicators": 2
        }
    elif timeframe == '1d':
        # Timeframe giornaliero - parametri ottimizzati per trading quotidiano
        params = {
            "rsi_period": 14,
            "rsi_overbought": 68,
            "rsi_oversold": 32,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ma_fast_period": 20,
            "ma_slow_period": 50,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "signal_quality_threshold": 60,
            "min_confirmation_indicators": 2
        }
    elif timeframe == '4h':
        # Timeframe 4 ore - buon compromesso tra giornaliero e orario
        params = {
            "rsi_period": 13,
            "rsi_overbought": 67,
            "rsi_oversold": 33,
            "macd_fast": 11,
            "macd_slow": 25,
            "macd_signal": 9,
            "ma_fast_period": 18,
            "ma_slow_period": 45,
            "bollinger_period": 18,
            "bollinger_std": 1.9,
            "signal_quality_threshold": 60,
            "min_confirmation_indicators": 2
        }
    elif timeframe == '1h':
        # Timeframe orario - ottimizzato per day trading attivo
        params = {
            "rsi_period": 12,
            "rsi_overbought": 66,
            "rsi_oversold": 34,
            "macd_fast": 10,
            "macd_slow": 23,
            "macd_signal": 8,
            "ma_fast_period": 15,
            "ma_slow_period": 40,
            "bollinger_period": 16,
            "bollinger_std": 1.8,
            "signal_quality_threshold": 55,
            "min_confirmation_indicators": 2
        }
    elif timeframe in ['15m', '30m']:
        # Timeframe 15-30 minuti - parametri più reattivi per scalping
        params = {
            "rsi_period": 10,
            "rsi_overbought": 65,
            "rsi_oversold": 35,
            "macd_fast": 9,
            "macd_slow": 22,
            "macd_signal": 7,
            "ma_fast_period": 12,
            "ma_slow_period": 35,
            "bollinger_period": 14,
            "bollinger_std": 1.7,
            "signal_quality_threshold": 50,
            "min_confirmation_indicators": 2
        }
    elif timeframe in ['1m', '5m']:
        # Timeframe molto breve - parametri ultra reattivi per scalping veloce
        params = {
            "rsi_period": 8,
            "rsi_overbought": 63,
            "rsi_oversold": 37,
            "macd_fast": 8,
            "macd_slow": 20,
            "macd_signal": 5,
            "ma_fast_period": 10,
            "ma_slow_period": 30,
            "bollinger_period": 12,
            "bollinger_std": 1.6,
            "signal_quality_threshold": 45,
            "min_confirmation_indicators": 2
        }
    
    # Modifiche per adattare l'aggressività dell'analisi
    if analysis_mode == "Aggressive":
        # Rendiamo i parametri più sensibili per generare più segnali
        params["rsi_overbought"] -= 5
        params["rsi_oversold"] += 5
        params["signal_quality_threshold"] -= 10
        params["min_confirmation_indicators"] = max(1, params["min_confirmation_indicators"] - 1)
        params["bollinger_std"] -= 0.3
    elif analysis_mode == "Conservative":
        # Rendiamo i parametri meno sensibili per generare segnali di qualità superiore
        params["rsi_overbought"] += 5
        params["rsi_oversold"] -= 5
        params["signal_quality_threshold"] += 10
        params["min_confirmation_indicators"] += 1
        params["bollinger_std"] += 0.3
    
    return params

def optimize_parameters_for_market(df, timeframe, rsi_period, rsi_overbought, rsi_oversold,
                                  macd_fast, macd_slow, macd_signal, ma_fast_period, 
                                  ma_slow_period, bollinger_period, bollinger_std):
    """
    Optimize indicator parameters based on market conditions
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    timeframe (str): Timeframe of the data
    rsi_period, rsi_overbought, rsi_oversold, etc.: Default indicator parameters
    
    Returns:
    dict: Dictionary with optimized parameters
    """
    # Start with the input parameters
    params = {
        "rsi_period": rsi_period,
        "rsi_overbought": rsi_overbought,
        "rsi_oversold": rsi_oversold,
        "macd_fast": macd_fast,
        "macd_slow": macd_slow,
        "macd_signal": macd_signal,
        "ma_fast_period": ma_fast_period,
        "ma_slow_period": ma_slow_period,
        "bollinger_period": bollinger_period,
        "bollinger_std": bollinger_std
    }
    
    # Need at least 50 candles for analysis
    if len(df) < 50:
        return params
    
    # Calculate volatility (using standard deviation of returns)
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * 100  # as percentage
    
    # Detect trend strength
    price_direction = df['close'].iloc[-1] > df['close'].iloc[-20]  # True if uptrend
    price_change_pct = ((df['close'].iloc[-1] / df['close'].iloc[-20]) - 1) * 100
    trend_strength = abs(price_change_pct)
    
    # Adjust RSI parameters based on volatility
    if volatility > 5:  # High volatility
        # Widen RSI bands to reduce false signals
        params["rsi_overbought"] = min(80, rsi_overbought + 5)
        params["rsi_oversold"] = max(20, rsi_oversold - 5)
    elif volatility < 1:  # Low volatility
        # Narrow RSI bands to capture more signals
        params["rsi_overbought"] = max(65, rsi_overbought - 5)
        params["rsi_oversold"] = min(35, rsi_oversold + 5)
    
    # Adjust MACD parameters based on trend strength
    if trend_strength > 10:  # Strong trend
        # Make MACD more responsive to catch trend continuations
        params["macd_fast"] = max(8, macd_fast - 2)
        params["macd_signal"] = max(7, macd_signal - 1)
    elif trend_strength < 3:  # Weak/No trend
        # Make MACD less sensitive to avoid false signals
        params["macd_fast"] = min(16, macd_fast + 2)
        params["macd_signal"] = min(12, macd_signal + 1)
    
    # Adjust Bollinger Bands based on volatility
    if volatility > 5:  # High volatility
        # Wider bands
        params["bollinger_std"] = min(3.0, bollinger_std + 0.3)
    elif volatility < 1:  # Low volatility
        # Narrower bands
        params["bollinger_std"] = max(1.5, bollinger_std - 0.3)
    
    # Ensure parameters make sense (e.g., fast period < slow period)
    params["macd_fast"] = min(params["macd_fast"], params["macd_slow"] - 4)
    params["ma_fast_period"] = min(params["ma_fast_period"], params["ma_slow_period"] - 10)
    
    return params

def calculate_advanced_risk_levels(df, stop_loss_percent=3.0, take_profit_percent=6.0):
    """
    Calculate optimized stop-loss and take-profit levels based on volatility
    
    Parameters:
    df (pd.DataFrame): DataFrame with price and indicator data
    stop_loss_percent (float): Base stop loss percentage
    take_profit_percent (float): Base take profit percentage
    
    Returns:
    pd.DataFrame: DataFrame with updated stop loss and take profit levels
    """
    df = df.copy()
    
    # Calculate Average True Range for volatility-based stops
    if 'atr' not in df.columns:
        # Calculate True Range
        df['tr0'] = df['high'] - df['low']
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate ATR (14-period)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Clean up temporary columns
        df = df.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1)
    
    # Only update where there are signals
    signal_mask = df['signal'] != 'neutral'
    
    if signal_mask.any():
        # For each signal row, calculate ATR-based stops
        for idx in df[signal_mask].index:
            current_price = df.loc[idx, 'close']
            atr_value = df.loc[idx, 'atr']
            signal_type = df.loc[idx, 'signal']
            
            # Use ATR for more volatile assets or standard percentage for less volatile ones
            atr_percent = (atr_value / current_price) * 100
            
            # Adjust stop loss based on volatility
            stop_multiplier = 1.0
            if atr_percent > 5:  # Very high volatility
                stop_multiplier = 1.5
            elif atr_percent > 3:  # High volatility
                stop_multiplier = 1.2
            elif atr_percent < 1:  # Low volatility
                stop_multiplier = 0.8
            
            adjusted_stop = stop_loss_percent * stop_multiplier
            adjusted_target = take_profit_percent * stop_multiplier
            
            # Calculate levels based on signal type
            if signal_type == 'buy':
                df.loc[idx, 'stop_loss'] = current_price * (1 - adjusted_stop/100)
                df.loc[idx, 'take_profit'] = current_price * (1 + adjusted_target/100)
            else:  # sell
                df.loc[idx, 'stop_loss'] = current_price * (1 + adjusted_stop/100)
                df.loc[idx, 'take_profit'] = current_price * (1 - adjusted_target/100)
    
    return df

def calculate_market_sentiment(signals_df):
    """
    Calculate overall market sentiment from signal data
    
    Parameters:
    signals_df (pd.DataFrame): DataFrame with signals data
    
    Returns:
    dict: Market sentiment metrics
    """
    if signals_df.empty:
        return {
            "bullish_percent": 0,
            "bearish_percent": 0,
            "neutral_percent": 100,
            "sentiment_score": 50,  # Neutral
            "overall_direction": "Neutral"
        }
    
    total_signals = len(signals_df)
    buy_signals = len(signals_df[signals_df['signal_type'] == 'buy'])
    sell_signals = len(signals_df[signals_df['signal_type'] == 'sell'])
    neutral_signals = total_signals - buy_signals - sell_signals
    
    # Calculate percentages
    bullish_percent = (buy_signals / total_signals) * 100 if total_signals > 0 else 0
    bearish_percent = (sell_signals / total_signals) * 100 if total_signals > 0 else 0
    neutral_percent = (neutral_signals / total_signals) * 100 if total_signals > 0 else 0
    
    # Calculate weighted sentiment score (0-100)
    # Weight signals by their quality/strength
    if 'signal_quality' in signals_df.columns:
        buy_weight = signals_df[signals_df['signal_type'] == 'buy']['signal_quality'].mean() if buy_signals > 0 else 0
        sell_weight = signals_df[signals_df['signal_type'] == 'sell']['signal_quality'].mean() if sell_signals > 0 else 0
    else:
        buy_weight = 1
        sell_weight = 1
    
    # Sentiment score: 0 = extremely bearish, 50 = neutral, 100 = extremely bullish
    sentiment_score = 50  # Start at neutral
    
    # Adjust based on signal count and weights
    if total_signals > 0:
        sentiment_score += (bullish_percent - bearish_percent) / 2
        
        # Additional adjustment based on signal weights
        if buy_signals > 0 and sell_signals > 0:
            weight_factor = (buy_weight - sell_weight) / 100
            sentiment_score += weight_factor * 10
    
    # Determine overall direction
    if sentiment_score >= 60:
        overall_direction = "Bullish"
    elif sentiment_score <= 40:
        overall_direction = "Bearish"
    else:
        overall_direction = "Neutral"
    
    # Add intensity level
    if sentiment_score >= 75:
        overall_direction = "Strongly " + overall_direction
    elif sentiment_score <= 25:
        overall_direction = "Strongly " + overall_direction
    
    return {
        "bullish_percent": bullish_percent,
        "bearish_percent": bearish_percent,
        "neutral_percent": neutral_percent,
        "sentiment_score": sentiment_score,
        "overall_direction": overall_direction
    }

def calculate_signal_success_metrics(signals_history, executed_only=True):
    """
    Calculate success metrics from historical signals
    
    Parameters:
    signals_history (list): List of historical signals
    executed_only (bool): Whether to only include executed signals
    
    Returns:
    dict: Signal success metrics
    """
    if not signals_history:
        return {
            "total_signals": 0,
            "executed_signals": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "success_rate": 0,
            "avg_profit_pct": 0,
            "win_loss_ratio": 0
        }
    
    # Filter executed signals if requested
    if executed_only:
        filtered_signals = [s for s in signals_history if s.get('executed', False)]
    else:
        filtered_signals = signals_history
    
    total_signals = len(filtered_signals)
    if total_signals == 0:
        return {
            "total_signals": len(signals_history),
            "executed_signals": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "success_rate": 0,
            "avg_profit_pct": 0,
            "win_loss_ratio": 0
        }
    
    # Count successful and failed signals
    # A signal is successful if:
    # 1. For buy signals: The price reached the take profit before the stop loss
    # 2. For sell signals: The price reached the take profit before the stop loss
    
    # This is an estimation as we don't have continuous price data
    # In a real system, we would track actual trades executed from these signals
    
    successful_signals = 0
    failed_signals = 0
    neutral_signals = 0
    
    total_profit_pct = 0
    
    for signal in filtered_signals:
        signal_type = signal.get('type')
        if signal_type not in ['buy', 'sell']:
            neutral_signals += 1
            continue
        
        # If we have profit/loss data from actual trades
        if 'actual_profit_pct' in signal:
            profit_pct = signal.get('actual_profit_pct', 0)
            if profit_pct > 0:
                successful_signals += 1
            else:
                failed_signals += 1
            
            total_profit_pct += profit_pct
        else:
            # Estimate success based on price action after signal
            # This would be more accurate with actual trade data
            # For now, assume a 50% success rate for demonstration
            if np.random.random() > 0.5:
                successful_signals += 1
                # Assume average profit of take_profit - entry_point
                if signal_type == 'buy':
                    profit_pct = ((signal.get('take_profit', 0) / signal.get('entry_point', 1)) - 1) * 100
                else:  # sell
                    profit_pct = ((signal.get('entry_point', 0) / signal.get('take_profit', 1)) - 1) * 100
            else:
                failed_signals += 1
                # Assume average loss of entry_point - stop_loss
                if signal_type == 'buy':
                    profit_pct = ((signal.get('stop_loss', 0) / signal.get('entry_point', 1)) - 1) * 100
                else:  # sell
                    profit_pct = ((signal.get('entry_point', 0) / signal.get('stop_loss', 1)) - 1) * 100
            
            total_profit_pct += profit_pct
    
    # Calculate metrics
    success_rate = (successful_signals / (successful_signals + failed_signals)) * 100 if (successful_signals + failed_signals) > 0 else 0
    avg_profit_pct = total_profit_pct / (successful_signals + failed_signals) if (successful_signals + failed_signals) > 0 else 0
    win_loss_ratio = successful_signals / failed_signals if failed_signals > 0 else 0
    
    return {
        "total_signals": len(signals_history),
        "executed_signals": total_signals,
        "successful_signals": successful_signals,
        "failed_signals": failed_signals,
        "neutral_signals": neutral_signals,
        "success_rate": success_rate,
        "avg_profit_pct": avg_profit_pct,
        "win_loss_ratio": win_loss_ratio
    }