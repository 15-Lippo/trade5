import pandas as pd
import numpy as np

def add_rsi(df, period=14):
    """
    Add Relative Strength Index (RSI) to the dataframe
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    period (int): RSI period
    
    Returns:
    pd.DataFrame: DataFrame with RSI
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate price change
    delta = df['close'].diff()
    
    # Calculate gain and loss
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Add Moving Average Convergence Divergence (MACD) to the dataframe
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    fast_period (int): Fast EMA period
    slow_period (int): Slow EMA period
    signal_period (int): Signal EMA period
    
    Returns:
    pd.DataFrame: DataFrame with MACD
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate EMAs
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD and signal line
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

def add_moving_averages(df, fast_period=20, slow_period=50):
    """
    Add Moving Averages to the dataframe
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    fast_period (int): Fast MA period
    slow_period (int): Slow MA period
    
    Returns:
    pd.DataFrame: DataFrame with MAs
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate simple moving averages
    df['ma_fast'] = df['close'].rolling(window=fast_period).mean()
    df['ma_slow'] = df['close'].rolling(window=slow_period).mean()
    
    return df

def add_bollinger_bands(df, period=20, std_dev=2.0):
    """
    Add Bollinger Bands to the dataframe
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    period (int): Period for moving average
    std_dev (float): Number of standard deviations
    
    Returns:
    pd.DataFrame: DataFrame with Bollinger Bands
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate middle band (simple moving average)
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    # Create an alias for compatibility with visualization code
    df['bb_ma'] = df['bb_middle']
    
    # Calculate standard deviation
    rolling_std = df['close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
    
    return df

def add_stochastic_oscillator(df, k_period=14, d_period=3):
    """
    Add Stochastic Oscillator to the dataframe
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    k_period (int): Period for %K
    d_period (int): Period for %D
    
    Returns:
    pd.DataFrame: DataFrame with Stochastic Oscillator
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate %K
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    # Calculate %D
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    
    return df

def add_atr(df, period=14):
    """
    Add Average True Range (ATR) to the dataframe
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    period (int): Period for ATR calculation
    
    Returns:
    pd.DataFrame: DataFrame with ATR
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    # Find the greatest of the three values
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # Calculate ATR
    df['atr'] = true_range.rolling(window=period).mean()
    
    return df

def add_adx(df, period=14):
    """
    Add Average Directional Index (ADX) to the dataframe
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    period (int): Period for ADX calculation
    
    Returns:
    pd.DataFrame: DataFrame with ADX and directional indicators
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate True Range
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # Calculate directional movement
    df['up_move'] = df['high'] - df['high'].shift()
    df['down_move'] = df['low'].shift() - df['low']
    
    # Calculate +DM and -DM (versione più efficiente senza warning)
    # Creiamo array vuoti
    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    
    # Condizioni per +DM
    condition_plus = (df['up_move'] > df['down_move']) & (df['up_move'] > 0)
    plus_dm[condition_plus] = df['up_move'][condition_plus]
    
    # Condizioni per -DM
    condition_minus = (df['down_move'] > df['up_move']) & (df['down_move'] > 0)
    minus_dm[condition_minus] = df['down_move'][condition_minus]
    
    # Aggiungiamo al dataframe
    df['plus_dm'] = plus_dm
    df['minus_dm'] = minus_dm
    
    # Calculate smoothed averages
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # Gestiamo divisione per zero in modo sicuro
    atr_non_zero = df['atr'].replace(0, np.nan)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / atr_non_zero)
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / atr_non_zero)
    
    # Calculate directional index and ADX (gestiamo divisione per zero)
    denom = df['plus_di'] + df['minus_di']
    df['dx'] = np.where(denom != 0, 
                         100 * abs(df['plus_di'] - df['minus_di']) / denom,
                         0)
    df['adx'] = df['dx'].rolling(window=period).mean()
    
    # Sostituisci NaN con 0 
    df.fillna(value={'plus_di': 0, 'minus_di': 0, 'dx': 0, 'adx': 0}, inplace=True)
    
    # Clean up temporary columns
    df.drop(['tr0', 'tr1', 'tr2', 'up_move', 'down_move'], axis=1, inplace=True)
    
    return df

def add_all_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9,
                      ma_fast_period=20, ma_slow_period=50, bollinger_period=20, 
                      bollinger_std=2.0, stoch_period=14, adx_period=14):
    """
    Add all technical indicators to the dataframe
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    rsi_period (int): Period for RSI calculation
    macd_fast (int): Fast period for MACD
    macd_slow (int): Slow period for MACD
    macd_signal (int): Signal period for MACD
    ma_fast_period (int): Fast MA period
    ma_slow_period (int): Slow MA period
    bollinger_period (int): Period for Bollinger Bands
    bollinger_std (float): Standard deviation for Bollinger Bands
    stoch_period (int): Period for Stochastic Oscillator
    adx_period (int): Period for ADX
    
    Returns:
    pd.DataFrame: DataFrame with all indicators
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Add RSI
    df = add_rsi(df, period=rsi_period)
    
    # Add MACD
    df = add_macd(df, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
    
    # Add Moving Averages
    df = add_moving_averages(df, fast_period=ma_fast_period, slow_period=ma_slow_period)
    
    # Add Bollinger Bands
    df = add_bollinger_bands(df, period=bollinger_period, std_dev=bollinger_std)
    
    # Add Stochastic Oscillator
    df = add_stochastic_oscillator(df, k_period=stoch_period)
    
    # Add ADX indicator - implementazione migliorata per evitare warning
    # Usa la funzione ottimizzata aggiornata
    df = add_adx(df, period=adx_period)
    
    return df
