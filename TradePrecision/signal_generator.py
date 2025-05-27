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
    
    # Rileva la tendenza attuale del mercato
    # Calcola la direzione e la forza della tendenza
    df['trend_direction'] = 0
    df['trend_strength'] = 0
    
    # Utilizzo delle medie mobili per identificare la tendenza
    if 'ma_fast' in df.columns and 'ma_slow' in df.columns:
        # Tendenza rialzista: media veloce sopra la media lenta
        df.loc[df['ma_fast'] > df['ma_slow'], 'trend_direction'] = 1
        # Tendenza ribassista: media veloce sotto la media lenta
        df.loc[df['ma_fast'] < df['ma_slow'], 'trend_direction'] = -1
        
        # Calcola la distanza percentuale tra le medie come misura della forza della tendenza
        df['ma_dist_pct'] = abs((df['ma_fast'] - df['ma_slow']) / df['ma_slow'] * 100)
        
        # Normalizza la forza della tendenza in scala 0-100
        max_dist = df['ma_dist_pct'].rolling(window=50).max().fillna(df['ma_dist_pct'])
        df['trend_strength'] = (df['ma_dist_pct'] / max_dist * 100).fillna(0).clip(0, 100)
    
    # Utilizza ADX per confermare la forza della tendenza se disponibile
    if 'adx' in df.columns:
        # Modifica la forza della tendenza in base all'ADX
        # ADX > 25 indica una tendenza forte
        df.loc[df['adx'] > 25, 'trend_strength'] = df.loc[df['adx'] > 25, 'trend_strength'] * 1.2
        # ADX > 40 indica una tendenza molto forte
        df.loc[df['adx'] > 40, 'trend_strength'] = df.loc[df['adx'] > 40, 'trend_strength'] * 1.3
    
    # 1. RSI Conditions - migliorato con analisi di tendenza
    # Condizioni di RSI per segnali di acquisto
    rsi_buy = (
        # Condizione standard: RSI sotto livello di ipervenduto e in salita
        ((df['rsi'] < rsi_oversold) & (df['rsi'].shift(1) <= df['rsi'])) |
        # Condizione vicino al livello di ipervenduto e in rapida salita
        ((df['rsi'] < rsi_oversold + 5) & (df['rsi'] - df['rsi'].shift(1) > 2)) |
        # Nuovo: RSI che esce dalla zona di ipervenduto (ottimo segnale di inversione)
        ((df['rsi'].shift(1) < rsi_oversold) & (df['rsi'] > rsi_oversold)) |
        # Nuovo: Divergenza positiva (prezzo in diminuzione ma RSI in aumento)
        ((df['close'] < df['close'].shift(3)) & (df['rsi'] > df['rsi'].shift(3)) & (df['rsi'] < 45))
    )
    
    # Condizioni di RSI per segnali di vendita
    rsi_sell = (
        # Condizione standard: RSI sopra livello di ipercomprato e in discesa
        ((df['rsi'] > rsi_overbought) & (df['rsi'].shift(1) >= df['rsi'])) |
        # Condizione vicino al livello di ipercomprato e in rapida discesa
        ((df['rsi'] > rsi_overbought - 5) & (df['rsi'].shift(1) - df['rsi'] > 2)) |
        # Nuovo: RSI che esce dalla zona di ipercomprato
        ((df['rsi'].shift(1) > rsi_overbought) & (df['rsi'] < rsi_overbought)) |
        # Nuovo: Divergenza negativa (prezzo in aumento ma RSI in diminuzione)
        ((df['close'] > df['close'].shift(3)) & (df['rsi'] < df['rsi'].shift(3)) & (df['rsi'] > 55))
    )
    
    # 2. MACD Conditions - migliorato
    # MACD incrocia sopra la linea del segnale (segnale di acquisto)
    macd_buy = (
        # Incrocio standard verso l'alto
        ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))) |
        # Anticipo dell'incrocio (le linee si stanno avvicinando rapidamente)
        ((df['macd'] < df['macd_signal']) & (df['macd'] - df['macd'].shift(1) > 0) & 
         (df['macd_signal'] - df['macd'] < (df['macd_signal'].shift(1) - df['macd'].shift(1)) * 0.5)) |
        # Nuovo: MACD attraversa lo zero verso l'alto (forte segnale di cambio tendenza)
        ((df['macd'] > 0) & (df['macd'].shift(1) <= 0)) |
        # Nuovo: MACD forma un minimo più alto mentre è sotto lo zero (divergenza rialzista)
        ((df['macd'] < 0) & (df['macd'] > df['macd'].shift(2)) & (df['macd'].shift(2) < df['macd'].shift(4)) &
         (df['close'] < df['close'].shift(2)))
    )
    
    # MACD incrocia sotto la linea del segnale (segnale di vendita)
    macd_sell = (
        # Incrocio standard verso il basso
        ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))) |
        # Anticipo dell'incrocio (le linee si stanno avvicinando rapidamente)
        ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) - df['macd'] > 0) & 
         (df['macd'] - df['macd_signal'] < (df['macd'].shift(1) - df['macd_signal'].shift(1)) * 0.5)) |
        # Nuovo: MACD attraversa lo zero verso il basso (forte segnale di cambio tendenza)
        ((df['macd'] < 0) & (df['macd'].shift(1) >= 0)) |
        # Nuovo: MACD forma un massimo più basso mentre è sopra lo zero (divergenza ribassista)
        ((df['macd'] > 0) & (df['macd'] < df['macd'].shift(2)) & (df['macd'].shift(2) > df['macd'].shift(4)) &
         (df['close'] > df['close'].shift(2)))
    )
    
    # 3. Moving Average Conditions - migliorato
    # Prezzo incrocia o si avvicina alla MA (segnale di acquisto)
    ma_buy = (
        # Incrocio standard verso l'alto
        ((df['close'] > df['ma_fast']) & (df['close'].shift(1) <= df['ma_fast'].shift(1))) |
        # Prezzo si avvicina alla media da sotto
        ((df['close'] < df['ma_fast']) & (df['ma_fast'] - df['close'] < df['ma_fast'] * 0.005) & 
         (df['close'] - df['close'].shift(1) > 0)) |
        # Nuovo: Rimbalzo dalla media mobile in tendenza rialzista
        ((df['trend_direction'] > 0) & (df['low'] <= df['ma_fast']) & (df['close'] > df['ma_fast']) &
         (df['close'] > df['open']))
    )
    
    # Prezzo incrocia o si avvicina alla MA (segnale di vendita)
    ma_sell = (
        # Incrocio standard verso il basso
        ((df['close'] < df['ma_fast']) & (df['close'].shift(1) >= df['ma_fast'].shift(1))) |
        # Prezzo si avvicina alla media da sopra
        ((df['close'] > df['ma_fast']) & (df['close'] - df['ma_fast'] < df['ma_fast'] * 0.005) & 
         (df['close'].shift(1) - df['close'] > 0)) |
        # Nuovo: Rimbalzo negativo dalla media mobile in tendenza ribassista
        ((df['trend_direction'] < 0) & (df['high'] >= df['ma_fast']) & (df['close'] < df['ma_fast']) &
         (df['close'] < df['open']))
    )
    
    # 4. MA Cross Conditions - migliorato
    # Media veloce incrocia o si avvicina alla media lenta (segnale di acquisto)
    ma_cross_buy = (
        # Incrocio standard
        ((df['ma_fast'] > df['ma_slow']) & (df['ma_fast'].shift(1) <= df['ma_slow'].shift(1))) |
        # Medie si stanno avvicinando
        ((df['ma_fast'] < df['ma_slow']) & (df['ma_slow'] - df['ma_fast'] < df['ma_slow'] * 0.01) & 
         (df['ma_fast'] - df['ma_fast'].shift(1) > 0)) |
        # Nuovo: Divergenza nelle medie (veloce accelera rispetto alla lenta)
        ((df['ma_fast'] - df['ma_fast'].shift(3)) > (df['ma_slow'] - df['ma_slow'].shift(3)) * 1.5)
    )
    
    # Media veloce incrocia o si avvicina alla media lenta (segnale di vendita)
    ma_cross_sell = (
        # Incrocio standard
        ((df['ma_fast'] < df['ma_slow']) & (df['ma_fast'].shift(1) >= df['ma_slow'].shift(1))) |
        # Medie si stanno avvicinando
        ((df['ma_fast'] > df['ma_slow']) & (df['ma_fast'] - df['ma_slow'] < df['ma_slow'] * 0.01) & 
         (df['ma_fast'].shift(1) - df['ma_fast'] > 0)) |
        # Nuovo: Divergenza nelle medie (veloce decelera rispetto alla lenta)
        ((df['ma_fast'].shift(3) - df['ma_fast']) > (df['ma_slow'].shift(3) - df['ma_slow']) * 1.5)
    )
    
    # 5. Bande di Bollinger - migliorato
    # Verifica che le bande di Bollinger siano presenti
    if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
        if 'bb_ma' not in df.columns:
            df['bb_ma'] = df['close'].rolling(window=20).mean()
        
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_ma'] + (bb_std * 2)
        df['bb_lower'] = df['bb_ma'] - (bb_std * 2)
    
    # Calcola la larghezza delle bande come misura della volatilità
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_ma'] * 100
    
    # Calcola la posizione del prezzo all'interno delle bande (0-100%)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
    df['bb_position'] = df['bb_position'].clip(0, 100)  # Limita tra 0 e 100%
    
    # Segnali di Bollinger migliorati
    bb_buy = (
        # Prezzo tocca o supera la banda inferiore
        (df['close'] <= df['bb_lower']) | ((df['low'] <= df['bb_lower']) & (df['close'] > df['bb_lower'])) |
        # Nuovo: Squeeze delle bande (compressione) seguita da espansione verso l'alto
        ((df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.85) & 
         (df['bb_width'] > df['bb_width'].shift(1)) & 
         (df['close'] > df['close'].shift(1)) & 
         (df['bb_position'] < 30)) |
        # Nuovo: Prezzo rimbalza dalla banda inferiore (conferma di inversione)
        ((df['bb_position'].shift(1) < 10) & 
         (df['bb_position'] > df['bb_position'].shift(1) + 5) & 
         (df['close'] > df['open']))
    )
    
    bb_sell = (
        # Prezzo tocca o supera la banda superiore
        (df['close'] >= df['bb_upper']) | ((df['high'] >= df['bb_upper']) & (df['close'] < df['bb_upper'])) |
        # Nuovo: Squeeze delle bande seguita da espansione verso il basso
        ((df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.85) & 
         (df['bb_width'] > df['bb_width'].shift(1)) & 
         (df['close'] < df['close'].shift(1)) & 
         (df['bb_position'] > 70)) |
        # Nuovo: Prezzo rimbalza dalla banda superiore (conferma di inversione)
        ((df['bb_position'].shift(1) > 90) & 
         (df['bb_position'] < df['bb_position'].shift(1) - 5) & 
         (df['close'] < df['open']))
    )
    
    # 6. Nuovo indicatore: Pattern di Candele
    # Inizializza colonne per i pattern
    df['bullish_pattern'] = False
    df['bearish_pattern'] = False
    
    # Calcola range delle candele
    df['body_size'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']
    df['body_percent'] = (df['body_size'] / df['total_range'] * 100).fillna(0)
    
    # Patterns rialzisti
    # Hammer (martello) - ombra inferiore lunga, corpo piccolo in alto
    hammer = (
        (df['body_percent'] < 30) &  # Corpo piccolo
        ((df['high'] - np.maximum(df['open'], df['close'])) < 
         (np.minimum(df['open'], df['close']) - df['low']) * 0.3) &  # Ombra superiore corta
        ((np.minimum(df['open'], df['close']) - df['low']) > 
         df['body_size'] * 2)  # Ombra inferiore lunga
    )
    
    # Bullish Engulfing - candela rialzista che ingloba la precedente
    bullish_engulfing = (
        (df['close'] > df['open']) &  # Candela rialzista (verde)
        (df['open'] < df['close'].shift(1)) &  # Apre sotto la chiusura precedente
        (df['close'] > df['open'].shift(1)) &  # Chiude sopra l'apertura precedente
        (df['close'].shift(1) < df['open'].shift(1))  # Precedente ribassista (rossa)
    )
    
    # Doji rialzista - indecisione seguita da movimento rialzista
    bullish_doji = (
        (df['body_percent'].shift(1) < 10) &  # Doji (corpo molto piccolo)
        (df['close'] > df['close'].shift(1) + df['total_range'].shift(1) * 0.6)  # Chiusura successiva rialzista
    )
    
    # Imposta pattern rialzisti
    df.loc[hammer | bullish_engulfing | bullish_doji, 'bullish_pattern'] = True
    
    # Patterns ribassisti
    # Shooting Star - ombra superiore lunga, corpo piccolo in basso
    shooting_star = (
        (df['body_percent'] < 30) &  # Corpo piccolo
        ((np.minimum(df['open'], df['close']) - df['low']) < 
         (df['high'] - np.maximum(df['open'], df['close'])) * 0.3) &  # Ombra inferiore corta
        ((df['high'] - np.maximum(df['open'], df['close'])) > 
         df['body_size'] * 2)  # Ombra superiore lunga
    )
    
    # Bearish Engulfing - candela ribassista che ingloba la precedente
    bearish_engulfing = (
        (df['close'] < df['open']) &  # Candela ribassista (rossa)
        (df['open'] > df['close'].shift(1)) &  # Apre sopra la chiusura precedente
        (df['close'] < df['open'].shift(1)) &  # Chiude sotto l'apertura precedente
        (df['close'].shift(1) > df['open'].shift(1))  # Precedente rialzista (verde)
    )
    
    # Doji ribassista - indecisione seguita da movimento ribassista
    bearish_doji = (
        (df['body_percent'].shift(1) < 10) &  # Doji (corpo molto piccolo)
        (df['close'] < df['close'].shift(1) - df['total_range'].shift(1) * 0.6)  # Chiusura successiva ribassista
    )
    
    # Imposta pattern ribassisti
    df.loc[shooting_star | bearish_engulfing | bearish_doji, 'bearish_pattern'] = True
    
    # Aggiungi segnali basati sui pattern
    pattern_buy = df['bullish_pattern']
    pattern_sell = df['bearish_pattern']
    
    # Combina tutti i segnali
    # Buy signals - ora con pattern di candele
    buy_count = (
        rsi_buy.astype(int) + 
        macd_buy.astype(int) + 
        ma_buy.astype(int) + 
        ma_cross_buy.astype(int) + 
        bb_buy.astype(int) +
        pattern_buy.astype(int)
    )
    
    # Sell signals - ora con pattern di candele
    sell_count = (
        rsi_sell.astype(int) + 
        macd_sell.astype(int) + 
        ma_sell.astype(int) + 
        ma_cross_sell.astype(int) + 
        bb_sell.astype(int) +
        pattern_sell.astype(int)
    )
    
    # Pesi degli indicatori basati sulla tendenza attuale
    # Adatta i segnali di acquisto/vendita in base alla tendenza
    for i in range(len(df)):
        if df['trend_direction'].iloc[i] > 0 and df['trend_strength'].iloc[i] > 30:
            # In una tendenza rialzista, aumenta il peso dei segnali di acquisto e diminuisci quelli di vendita
            buy_weight = 1.0 + (df['trend_strength'].iloc[i] / 200)  # Max +50%
            sell_weight = 1.0 - (df['trend_strength'].iloc[i] / 200)  # Max -50%
            # Converti in int per evitare warning
            buy_count.iloc[i] = int(buy_count.iloc[i] * buy_weight)
            sell_count.iloc[i] = int(sell_count.iloc[i] * sell_weight)
        elif df['trend_direction'].iloc[i] < 0 and df['trend_strength'].iloc[i] > 30:
            # In una tendenza ribassista, aumenta il peso dei segnali di vendita e diminuisci quelli di acquisto
            buy_weight = 1.0 - (df['trend_strength'].iloc[i] / 200)  # Max -50%
            sell_weight = 1.0 + (df['trend_strength'].iloc[i] / 200)  # Max +50%
            # Converti in int per evitare warning
            buy_count.iloc[i] = int(buy_count.iloc[i] * buy_weight)
            sell_count.iloc[i] = int(sell_count.iloc[i] * sell_weight)
    
    # Calcola soglia minima per generare segnali
    min_threshold = 1  # Soglia base
    
    # Basato sulla sensibilità del timeframe
    if sensitivity >= 1.5:
        min_threshold = 1  # Per timeframe mensile/alta sensibilità
    elif sensitivity >= 1.25:
        min_threshold = 1  # Per sensibilità media-alta
    else:
        min_threshold = 1  # Sensibilità standard
    
    # Genera segnali in base alle soglie
    df.loc[buy_count >= min_threshold, 'signal'] = 'buy'
    df.loc[sell_count >= min_threshold, 'signal'] = 'sell'
    
    # Segnali forti (con più indicatori in accordo)
    strong_threshold = 2
    if sensitivity >= 1.5:
        strong_threshold = 2  # Alta sensibilità
    elif sensitivity <= 0.9:
        strong_threshold = 3  # Bassa sensibilità
    
    df.loc[buy_count >= strong_threshold, 'signal'] = 'strong_buy'
    df.loc[sell_count >= strong_threshold, 'signal'] = 'strong_sell'
    
    # Semplifica i segnali per compatibilità
    df.loc[df['signal'] == 'strong_buy', 'signal'] = 'buy'
    df.loc[df['signal'] == 'strong_sell', 'signal'] = 'sell'
    
    # Adatta la forza del segnale in base al numero di indicatori (0-100)
    # Con 6 indicatori, il massimo è 6*20 = 120, limita a 100
    df['signal_strength'] = 0
    
    # Per segnali di acquisto
    df.loc[df['signal'] == 'buy', 'signal_strength'] = (
        (rsi_buy.astype(int) * 17) + 
        (macd_buy.astype(int) * 17) + 
        (ma_buy.astype(int) * 17) + 
        (ma_cross_buy.astype(int) * 17) +
        (bb_buy.astype(int) * 17) +
        (pattern_buy.astype(int) * 17)
    ).clip(0, 100)
    
    # Per segnali di vendita
    df.loc[df['signal'] == 'sell', 'signal_strength'] = (
        (rsi_sell.astype(int) * 17) + 
        (macd_sell.astype(int) * 17) + 
        (ma_sell.astype(int) * 17) + 
        (ma_cross_sell.astype(int) * 17) +
        (bb_sell.astype(int) * 17) +
        (pattern_sell.astype(int) * 17)
    ).clip(0, 100)
    
    # Aggiungi un fattore di tendenza alla forza del segnale
    # Nella direzione della tendenza, aumenta la forza
    for i in range(len(df)):
        if df['signal'].iloc[i] == 'buy' and df['trend_direction'].iloc[i] > 0:
            trend_bonus = min(df['trend_strength'].iloc[i] * 0.2, 15)  # Max +15
            # Converti in int per evitare warning
            new_strength = int(min(100, df['signal_strength'].iloc[i] + trend_bonus))
            df.loc[df.index[i], 'signal_strength'] = new_strength
        elif df['signal'].iloc[i] == 'sell' and df['trend_direction'].iloc[i] < 0:
            trend_bonus = min(df['trend_strength'].iloc[i] * 0.2, 15)  # Max +15
            # Converti in int per evitare warning
            new_strength = int(min(100, df['signal_strength'].iloc[i] + trend_bonus))
            df.loc[df.index[i], 'signal_strength'] = new_strength
    
    # Calcola livelli di ingresso, stop loss e take profit
    df['entry_point'] = np.nan
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    
    # Imposta punti di ingresso
    df.loc[df['signal'] != 'neutral', 'entry_point'] = df['close']
    
    # Calcola stop loss e take profit basati sulla volatilità (ATR)
    if 'atr' in df.columns:
        for i in range(len(df)):
            if df['signal'].iloc[i] == 'neutral':
                continue
                
            price = df['close'].iloc[i]
            atr = df['atr'].iloc[i]
            
            # Calcola multipli ATR ottimizzati per il timeframe
            if timeframe in ['1m', '5m', '15m']:
                # Timeframe brevi - stop più stretti
                sl_mult = 1.5
                tp_mult = 3.0
            elif timeframe in ['30m', '1h', '4h']:
                # Timeframe medi
                sl_mult = 2.0
                tp_mult = 4.0
            else:
                # Timeframe lunghi - stop più ampi
                sl_mult = 2.5
                tp_mult = 5.0
                
            # Aggiungi variazione basata sulla volatilità
            if 'bb_width' in df.columns:
                # Se la volatilità è alta, aumenta la distanza dello stop
                bb_width_norm = df['bb_width'].iloc[i] / df['bb_width'].rolling(20).mean().iloc[i]
                if bb_width_norm > 1.3:  # Alta volatilità
                    sl_mult *= 1.3
                    tp_mult *= 1.3
                elif bb_width_norm < 0.7:  # Bassa volatilità
                    sl_mult *= 0.8
                    tp_mult *= 0.8
            
            # Imposta livelli per segnali di acquisto
            if df['signal'].iloc[i] == 'buy':
                df.loc[df.index[i], 'stop_loss'] = price - (atr * sl_mult)
                df.loc[df.index[i], 'take_profit'] = price + (atr * tp_mult)
            # Imposta livelli per segnali di vendita
            else:  # sell
                df.loc[df.index[i], 'stop_loss'] = price + (atr * sl_mult)
                df.loc[df.index[i], 'take_profit'] = price - (atr * tp_mult)
    else:
        # Se ATR non è disponibile, usa la percentuale standard
        # Stop loss per segnali di acquisto (3% sotto l'ingresso)
        df.loc[df['signal'] == 'buy', 'stop_loss'] = df['entry_point'] * 0.97
        # Stop loss per segnali di vendita (3% sopra l'ingresso)
        df.loc[df['signal'] == 'sell', 'stop_loss'] = df['entry_point'] * 1.03
        
        # Take profit per segnali di acquisto (6% sopra l'ingresso)
        df.loc[df['signal'] == 'buy', 'take_profit'] = df['entry_point'] * 1.06
        # Take profit per segnali di vendita (6% sotto l'ingresso)
        df.loc[df['signal'] == 'sell', 'take_profit'] = df['entry_point'] * 0.94
    
    # Pulisce le colonne temporanee
    cols_to_drop = ['body_size', 'total_range', 'body_percent', 'bullish_pattern', 'bearish_pattern']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
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
