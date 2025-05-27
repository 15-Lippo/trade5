import pandas as pd
import numpy as np
from datetime import datetime

def generate_signals(df, rsi_overbought=70, rsi_oversold=30, timeframe=None):
    """
    Generate buy/sell signals based on technical indicators
    
    Parameters:
    df (pd.DataFrame): DataFrame with price and indicators data
    rsi_overbought (int): RSI overbought level
    rsi_oversold (int): RSI oversold level
    timeframe (str, optional): Timeframe dei dati per regolare la sensibilità
    
    Returns:
    pd.DataFrame: DataFrame with signals
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Configura la sensibilità in base al timeframe (se specificato)
    sensitivity = 1.0  # Sensibilità standard
    
    # Timeframe mensile -> sensibilità ancora più alta per generare segnali nonostante la cautela
    if timeframe == '1M':
        sensitivity = 1.5  # Molto più sensibile per timeframe mensile
    # Timeframe lunghi -> maggiore sensibilità per catturare più movimenti significativi
    elif timeframe in ['4h', '1d', '1w', '2w']:
        sensitivity = 1.25  # Più sensibile per timeframe lunghi
    # Timeframe brevi -> sensibilità media-alta per segnali frequenti
    elif timeframe in ['1m', '5m', '15m']:
        sensitivity = 1.2  # Sensibilità media-alta per timeframe corti
    
    # Initialize signal column
    df['signal'] = 'neutral'
    
    # 1. RSI Conditions (più sensibili)
    # Oversold condition (buy signal) - più sensibile per individuare inversioni di tendenza
    rsi_buy = (
        # Condizione standard (RSI sotto il livello di ipervenduto e sta salendo)
        ((df['rsi'] < rsi_oversold) & (df['rsi'].shift(1) <= df['rsi'])) |
        # Condizione aggiuntiva (RSI vicino al livello di ipervenduto e sta salendo rapidamente)
        ((df['rsi'] < rsi_oversold + 5) & (df['rsi'] - df['rsi'].shift(1) > 2))
    )
    
    # Overbought condition (sell signal) - più sensibile per individuare inversioni di tendenza
    rsi_sell = (
        # Condizione standard (RSI sopra il livello di ipercomprato e sta scendendo)
        ((df['rsi'] > rsi_overbought) & (df['rsi'].shift(1) >= df['rsi'])) |
        # Condizione aggiuntiva (RSI vicino al livello di ipercomprato e sta scendendo rapidamente)
        ((df['rsi'] > rsi_overbought - 5) & (df['rsi'].shift(1) - df['rsi'] > 2))
    )
    
    # 2. MACD Conditions (più sensibili)
    # MACD line crosses above signal line (buy signal)
    macd_buy = (
        # Incrocio standard verso l'alto
        ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))) |
        # Condizione per anticipare l'incrocio (le linee si stanno avvicinando rapidamente)
        ((df['macd'] < df['macd_signal']) & (df['macd'] - df['macd'].shift(1) > 0) & 
         (df['macd_signal'] - df['macd'] < (df['macd_signal'].shift(1) - df['macd'].shift(1)) * 0.5))
    )
    
    # MACD line crosses below signal line (sell signal)
    macd_sell = (
        # Incrocio standard verso il basso
        ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))) |
        # Condizione per anticipare l'incrocio (le linee si stanno avvicinando rapidamente)
        ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) - df['macd'] > 0) & 
         (df['macd'] - df['macd_signal'] < (df['macd'].shift(1) - df['macd_signal'].shift(1)) * 0.5))
    )
    
    # 3. Moving Average Conditions (più sensibili)
    # Price crosses or approaches MA (buy signal)
    ma_buy = (
        # Incrocio standard (prezzo attraversa la media mobile veloce dal basso)
        ((df['close'] > df['ma_fast']) & (df['close'].shift(1) <= df['ma_fast'].shift(1))) |
        # Condizione per prezzo che si avvicina alla media da sotto
        ((df['close'] < df['ma_fast']) & (df['ma_fast'] - df['close'] < df['ma_fast'] * 0.005) & 
         (df['close'] - df['close'].shift(1) > 0))
    )
    
    # Price crosses or approaches MA (sell signal)
    ma_sell = (
        # Incrocio standard (prezzo attraversa la media mobile veloce dall'alto)
        ((df['close'] < df['ma_fast']) & (df['close'].shift(1) >= df['ma_fast'].shift(1))) |
        # Condizione per prezzo che si avvicina alla media da sopra
        ((df['close'] > df['ma_fast']) & (df['close'] - df['ma_fast'] < df['ma_fast'] * 0.005) & 
         (df['close'].shift(1) - df['close'] > 0))
    )
    
    # 4. MA Cross Conditions (più sensibili)
    # Fast MA crosses or approaches slow MA (buy signal)
    ma_cross_buy = (
        # Incrocio standard
        ((df['ma_fast'] > df['ma_slow']) & (df['ma_fast'].shift(1) <= df['ma_slow'].shift(1))) |
        # Condizione per medie che si stanno avvicinando
        ((df['ma_fast'] < df['ma_slow']) & (df['ma_slow'] - df['ma_fast'] < df['ma_slow'] * 0.01) & 
         (df['ma_fast'] - df['ma_fast'].shift(1) > 0))
    )
    
    # Fast MA crosses or approaches slow MA (sell signal)
    ma_cross_sell = (
        # Incrocio standard
        ((df['ma_fast'] < df['ma_slow']) & (df['ma_fast'].shift(1) >= df['ma_slow'].shift(1))) |
        # Condizione per medie che si stanno avvicinando
        ((df['ma_fast'] > df['ma_slow']) & (df['ma_fast'] - df['ma_slow'] < df['ma_slow'] * 0.01) & 
         (df['ma_fast'].shift(1) - df['ma_fast'] > 0))
    )
    
    # 5. Bande di Bollinger (nuova condizione)
    # Calcola le bande di Bollinger se non sono già presenti
    if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
        # Calcola la media mobile a 20 periodi se non presente
        if 'bb_ma' not in df.columns:
            df['bb_ma'] = df['close'].rolling(window=20).mean()
        
        # Calcola la deviazione standard
        bb_std = df['close'].rolling(window=20).std()
        
        # Calcola le bande superiore e inferiore (2 deviazioni standard)
        df['bb_upper'] = df['bb_ma'] + (bb_std * 2)
        df['bb_lower'] = df['bb_ma'] - (bb_std * 2)
    
    # Segnali di Bollinger
    # Prezzo tocca o supera la banda inferiore (segnale di acquisto)
    bb_buy = (df['close'] <= df['bb_lower']) | ((df['low'] <= df['bb_lower']) & (df['close'] > df['bb_lower']))
    
    # Prezzo tocca o supera la banda superiore (segnale di vendita)
    bb_sell = (df['close'] >= df['bb_upper']) | ((df['high'] >= df['bb_upper']) & (df['close'] < df['bb_upper']))
    
    # Combined signals
    # Buy signals - ora includiamo anche le bande di Bollinger
    buy_count = rsi_buy.astype(int) + macd_buy.astype(int) + ma_buy.astype(int) + ma_cross_buy.astype(int) + bb_buy.astype(int)
    # Sell signals - ora includiamo anche le bande di Bollinger
    sell_count = rsi_sell.astype(int) + macd_sell.astype(int) + ma_sell.astype(int) + ma_cross_sell.astype(int) + bb_sell.astype(int)
    
    # Applica il fattore di sensibilità in base al timeframe
    # Per timeframe più lunghi (sensitivity < 1.0) richiederà più conferme
    # Per timeframe più brevi (sensitivity > 1.0) richiederà meno conferme
    
    # Calcola la soglia minima in base alla sensibilità
    min_threshold = 2  # Soglia standard
    
    # Più sensibilità = meno conferme richieste per generare segnali
    if sensitivity >= 1.5:
        # Per timeframe mensile o con alta sensibilità, meno conferme necessarie
        min_threshold = 1  # Basta un solo indicatore per generare segnali 
    elif sensitivity >= 1.25:
        # Per sensibilità medio-alta, soglia standard
        min_threshold = 2  # Due indicatori confermati
    else:
        # Per sensibilità più bassa, richiederebbe più conferme
        min_threshold = 2  # Manteniamo comunque 2 per evitare di avere troppi pochi segnali
    
    # Aggiorna la logica per tener conto della sensibilità
    # Set signal based on counts with sensitivity adjustment
    df.loc[buy_count >= min_threshold, 'signal'] = 'buy'
    df.loc[sell_count >= min_threshold, 'signal'] = 'sell'
    
    # Segnale extra forte con meno indicatori richiesti per alta sensibilità
    strong_threshold = 3  # Default per sensibilità standard
    if sensitivity >= 1.5:
        strong_threshold = 2  # Per alta sensibilità, bastano 2 indicatori per segnale forte
    elif sensitivity <= 0.9:
        strong_threshold = 4  # Per bassa sensibilità, servono 4 indicatori per segnale forte
    df.loc[buy_count >= strong_threshold, 'signal'] = 'strong_buy'
    df.loc[sell_count >= strong_threshold, 'signal'] = 'strong_sell'
    
    # Semplifichiamo in buy/sell per la compatibilità con il resto del codice
    df.loc[df['signal'] == 'strong_buy', 'signal'] = 'buy'
    df.loc[df['signal'] == 'strong_sell', 'signal'] = 'sell'
    
    # Calculate signal strength (0-100) - ora con 5 indicatori
    df['signal_strength'] = 0
    
    # For buy signals
    df.loc[df['signal'] == 'buy', 'signal_strength'] = (
        (rsi_buy.astype(int) * 20) + 
        (macd_buy.astype(int) * 20) + 
        (ma_buy.astype(int) * 20) + 
        (ma_cross_buy.astype(int) * 20) +
        (bb_buy.astype(int) * 20)
    )
    
    # For sell signals
    df.loc[df['signal'] == 'sell', 'signal_strength'] = (
        (rsi_sell.astype(int) * 20) + 
        (macd_sell.astype(int) * 20) + 
        (ma_sell.astype(int) * 20) + 
        (ma_cross_sell.astype(int) * 20) +
        (bb_sell.astype(int) * 20)
    )
    
    # Calculate entry points, stop-loss and take profit levels
    df['entry_point'] = np.nan
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    
    # Entry point is the close price
    df.loc[df['signal'] != 'neutral', 'entry_point'] = df['close']
    
    # Stop-loss calculation
    # For buy signals, stop loss is below the entry point
    df.loc[df['signal'] == 'buy', 'stop_loss'] = df['entry_point'] * 0.97  # 3% below entry
    
    # For sell signals, stop loss is above the entry point
    df.loc[df['signal'] == 'sell', 'stop_loss'] = df['entry_point'] * 1.03  # 3% above entry
    
    # Take profit calculation
    # For buy signals, take profit is above the entry point (risk-reward ratio 1:2)
    df.loc[df['signal'] == 'buy', 'take_profit'] = df['entry_point'] * 1.06  # 6% above entry
    
    # For sell signals, take profit is below the entry point (risk-reward ratio 1:2)
    df.loc[df['signal'] == 'sell', 'take_profit'] = df['entry_point'] * 0.94  # 6% below entry
    
    return df

def get_recent_signals(df, symbol, lookback=5, timeframe=None):
    """
    Get the most recent signals for a cryptocurrency, anche quelli generati nelle barre recenti
    
    Parameters:
    df (pd.DataFrame): DataFrame with signals
    symbol (str): Cryptocurrency symbol
    lookback (int): Numero di barre da controllare per segnali recenti
    timeframe (str): Timeframe dei dati, usato per adattare il lookback
    
    Returns:
    list: List of dictionaries with recent signals
    """
    # Adatta il lookback in base al timeframe per i timeframe più lunghi
    if timeframe in ['4h', '1d', '1w', '2w', '1M']:
        # Per timeframe lunghi, controlliamo più barre per avere più segnali
        lookback = max(lookback, 10)  # Minimo 10 barre per timeframe lunghi
        if timeframe in ['1w', '2w', '1M']:
            lookback = max(lookback, 20)  # Minimo 20 barre per timeframe molto lunghi
    # Make a copy and reset index to access timestamp as a column
    df_reset = df.reset_index()
    
    # Get recent signals
    recent_signals = []
    
    # Get the most recent data point
    latest = df_reset.iloc[-1]
    
    # Create a dictionary with signal information for the latest data point
    signal_info = {
        'symbol': symbol,
        'price': latest['close'],
        'signal_type': latest['signal'],
        'timestamp': latest['timestamp'],
    }
    
    # Add entry point, stop loss and take profit if there's a buy or sell signal
    if latest['signal'] != 'neutral':
        signal_info['entry_point'] = latest['entry_point']
        signal_info['stop_loss'] = latest['stop_loss']
        signal_info['take_profit'] = latest['take_profit']
        signal_info['signal_strength'] = latest['signal_strength']
        recent_signals.append(signal_info)
    
    # Use previous signals if no current signal or always check for recent signals
    # Look back at most 'lookback' bars
    for i in range(2, min(lookback + 1, len(df_reset) + 1)):
        previous = df_reset.iloc[-i]
        if previous['signal'] != 'neutral':
            prev_signal = {
                'symbol': symbol,
                'price': previous['close'],
                'signal_type': previous['signal'],
                'entry_point': previous['entry_point'],
                'stop_loss': previous['stop_loss'],
                'take_profit': previous['take_profit'],
                'signal_strength': previous['signal_strength'],
                'timestamp': previous['timestamp'],
            }
            recent_signals.append(prev_signal)
    
    # Se non abbiamo trovato nessun segnale, restituisci almeno il punto attuale (anche se neutrale)
    if not recent_signals:
        recent_signals.append(signal_info)
    
    return recent_signals

def get_historical_signals(df, symbol):
    """
    Get historical signals for a cryptocurrency
    
    Parameters:
    df (pd.DataFrame): DataFrame with signals
    symbol (str): Cryptocurrency symbol
    
    Returns:
    list: List of dictionaries with historical signals
    """
    # Make a copy and reset index to access timestamp as a column
    df_reset = df.reset_index()
    
    # Filter for buy and sell signals
    signal_rows = df_reset[df_reset['signal'] != 'neutral']
    
    if signal_rows.empty:
        return []
    
    # Create list of signal dictionaries
    historical_signals = []
    
    for _, row in signal_rows.iterrows():
        # Create signal dictionary
        signal_dict = {
            'symbol': symbol,
            'signal_type': row['signal'],
            'price': row['close'],
            'entry_point': row['entry_point'],
            'stop_loss': row['stop_loss'],
            'take_profit': row['take_profit'],
            'timestamp': row['timestamp'],
            'signal_strength': row['signal_strength'],
        }
        
        # Simulate exit price for completed signals (not the most recent one)
        if row['timestamp'] < df_reset.iloc[-1]['timestamp']:
            # Find the next opposite signal or use last price
            exit_index = signal_rows[(signal_rows['timestamp'] > row['timestamp']) & 
                                    (signal_rows['signal'] != row['signal'])].index
            
            if len(exit_index) > 0:
                # Get the first opposite signal
                exit_row = df_reset.loc[exit_index[0]]
                exit_price = exit_row['close']
            else:
                # Use the last price if no opposite signal
                exit_price = df_reset.iloc[-1]['close']
            
            signal_dict['exit_price'] = exit_price
            
            # Calculate profit/loss
            if row['signal'] == 'buy':
                signal_dict['profit_loss'] = ((exit_price / row['entry_point']) - 1) * 100
            else:  # sell signal
                signal_dict['profit_loss'] = ((row['entry_point'] / exit_price) - 1) * 100
        
        historical_signals.append(signal_dict)
    
    return historical_signals

def calculate_performance(hist_df, price_df):
    """
    Calculate performance metrics for historical signals
    
    Parameters:
    hist_df (pd.DataFrame): DataFrame with historical signals
    price_df (pd.DataFrame): DataFrame with price data
    
    Returns:
    dict: Dictionary with performance metrics
    """
    # Filter out signals without exit prices
    completed_signals = hist_df[hist_df['exit_price'].notna()]
    
    if completed_signals.empty:
        return {
            'win_rate': 0,
            'avg_profit': 0,
            'max_drawdown': 0,
            'total_signals': 0
        }
    
    # Calculate win rate
    winning_trades = completed_signals[completed_signals['profit_loss'] > 0]
    win_rate = len(winning_trades) / len(completed_signals) * 100
    
    # Calculate average profit
    avg_profit = completed_signals['profit_loss'].mean()
    
    # Calculate max drawdown
    # This is a simplified version
    profits = completed_signals['profit_loss'].sort_values()
    max_drawdown = profits.iloc[0] if len(profits) > 0 else 0
    
    return {
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'max_drawdown': max_drawdown,
        'total_signals': len(completed_signals)
    }
