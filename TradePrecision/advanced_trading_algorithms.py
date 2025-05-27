import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import technical_indicators
import signal_generator
import market_sentiment

class AdvancedTradingAlgorithm:
    """
    Implementazione di algoritmi di trading avanzati con molteplici strategie e approcci.
    Questa classe fornisce un'analisi di trading sofisticata utilizzando tecniche quantitative avanzate.
    """
    
    def __init__(self):
        """Inizializza l'algoritmo di trading avanzato"""
        self.available_strategies = [
            'long_term_trend_analysis',
            'ml_enhanced_trend_following',
            'volatility_breakout',
            'mean_reversion',
            'smart_momentum',
            'channel_trading',
            'multi_timeframe_confluence',
            'advanced_support_resistance',
            'volume_profile_analysis',
            'pattern_recognition'
        ]
        
        self.strategy_descriptions = {
            'long_term_trend_analysis': 'Analisi avanzata di trend a lungo termine con identificazione di supporti e resistenze storici (ideale per timeframe 1w-1M)',
            'ml_enhanced_trend_following': 'Strategia di trend following potenziata con machine learning per identificare la forza del trend',
            'volatility_breakout': 'Individua breakout basati sulla volatilità con filtri per ridurre i falsi segnali',
            'mean_reversion': 'Strategia di ritorno alla media con identificazione delle zone di ipercomprato/ipervenduto',
            'smart_momentum': 'Analisi di momentum avanzata con filtri dinamici basati sulle condizioni di mercato',
            'channel_trading': 'Trading basato su canali di prezzo con identificazione di supporti e resistenze dinamici',
            'multi_timeframe_confluence': 'Analisi multi-timeframe con identificazione di confluence tra diversi indicatori',
            'advanced_support_resistance': 'Analisi avanzata di supporti e resistenze basata su price action, volume e pattern',
            'volume_profile_analysis': 'Analisi del profilo del volume per identificare livelli di prezzo significativi',
            'pattern_recognition': 'Riconoscimento di pattern di candele e formazioni grafiche avanzate'
        }
    
    def long_term_trend_analysis(self, df, **params):
        """
        Analisi avanzata di trend a lungo termine con identificazione di supporti e resistenze storici.
        Questa strategia è specificamente progettata per timeframe settimanali e mensili.
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV e indicatori
        params: Parametri opzionali per personalizzare la strategia
        
        Returns:
        pd.DataFrame: DataFrame con i segnali generati
        """
        df = df.copy()
        
        # Assicura che gli indicatori necessari siano presenti
        if 'ma_fast' not in df.columns or 'ma_slow' not in df.columns:
            ma_slow_period = params.get('ma_slow_period', 50)
            ma_very_slow_period = params.get('ma_very_slow_period', 200)
            df = technical_indicators.add_moving_averages(df, fast_period=ma_slow_period, slow_period=ma_very_slow_period)
        
        # Aggiungi media mobile a lunghissimo termine (per trend secolare)
        longest_ma_period = params.get('longest_ma_period', 200)
        if len(df) > longest_ma_period:
            df['ma_longest'] = df['close'].rolling(window=longest_ma_period).mean()
        else:
            df['ma_longest'] = df['ma_slow']
            
        # Aggiungi ATR per volatilità e calcolo dei livelli
        if 'atr' not in df.columns:
            atr_period = params.get('atr_period', 21)  # ~1 mese di trading per timeframe settimanali
            df = technical_indicators.add_atr(df, period=atr_period)
        
        # 1. Identificazione di supporti e resistenze storici
        
        # Calcola livelli chiave su finestre di lungo periodo
        # Utilizziamo 6 mesi, 1 anno e 2 anni per timeframe settimanali
        lookback_periods = [26, 52, 104]  # Settimane (~6 mesi, ~1 anno, ~2 anni)
        
        # Adatta in base ai dati disponibili
        valid_lookbacks = [lp for lp in lookback_periods if lp < len(df)]
        
        if not valid_lookbacks:
            valid_lookbacks = [max(int(len(df)/4), 4), max(int(len(df)/2), 8)]
        
        # Inizializza liste per memorizzare i livelli chiave
        support_levels = []
        resistance_levels = []
        
        # Per ogni periodo di lookback, trova i massimi e minimi significativi
        for period in valid_lookbacks:
            # Calcola i massimi locali (massimi che sono più alti dei punti adiacenti)
            for i in range(period, len(df) - period):
                # Un punto è un massimo locale se è più alto di tutti i punti nel periodo considerato
                window = df['high'].iloc[i-period:i+period]
                if df['high'].iloc[i] == window.max():
                    resistance_levels.append(df['high'].iloc[i])
            
            # Calcola i minimi locali (minimi che sono più bassi dei punti adiacenti)
            for i in range(period, len(df) - period):
                # Un punto è un minimo locale se è più basso di tutti i punti nel periodo considerato
                window = df['low'].iloc[i-period:i+period]
                if df['low'].iloc[i] == window.min():
                    support_levels.append(df['low'].iloc[i])
        
        # 2. Clusterizzazione dei livelli simili (raggruppa livelli che sono entro un certo range)
        
        # Funzione per raggruppare livelli simili
        def cluster_levels(levels, threshold_pct=0.02):
            if not levels:
                return []
                
            # Ordina i livelli
            sorted_levels = sorted(levels)
            
            # Inizializza cluster
            clusters = [[sorted_levels[0]]]
            
            # Raggruppa livelli vicini
            for level in sorted_levels[1:]:
                last_cluster = clusters[-1]
                last_level = last_cluster[-1]
                
                # Se il livello è abbastanza vicino all'ultimo del cluster, aggiungilo al cluster
                if abs(level - last_level) / last_level < threshold_pct:
                    last_cluster.append(level)
                else:
                    # Altrimenti, crea un nuovo cluster
                    clusters.append([level])
            
            # Calcola la media di ogni cluster
            return [sum(cluster) / len(cluster) for cluster in clusters]
        
        # Clusterizza i livelli
        clustered_supports = cluster_levels(support_levels)
        clustered_resistances = cluster_levels(resistance_levels)
        
        # 3. Trova i livelli più rilevanti in base alla distanza dal prezzo attuale
        current_price = df['close'].iloc[-1]
        
        # Filtra supporti sotto il prezzo attuale e resistenze sopra
        active_supports = [s for s in clustered_supports if s < current_price]
        active_resistances = [r for r in clustered_resistances if r > current_price]
        
        # Ordina per distanza dal prezzo attuale
        active_supports = sorted(active_supports, key=lambda x: current_price - x)
        active_resistances = sorted(active_resistances, key=lambda x: x - current_price)
        
        # 4. Determina lo stato del trend a lungo termine
        
        # Trend basati su posizione rispetto alle medie mobili
        above_ma_slow = current_price > df['ma_slow'].iloc[-1]
        above_ma_longest = current_price > df['ma_longest'].iloc[-1] if 'ma_longest' in df.columns else above_ma_slow
        
        # Trend basato sulla direzione delle medie mobili
        ma_slow_rising = df['ma_slow'].iloc[-1] > df['ma_slow'].iloc[-min(20, len(df))]
        ma_longest_rising = df['ma_longest'].iloc[-1] > df['ma_longest'].iloc[-min(20, len(df))] if 'ma_longest' in df.columns else ma_slow_rising
        
        # Definisci il trend principale
        if above_ma_slow and above_ma_longest and ma_slow_rising and ma_longest_rising:
            long_term_trend = 'strong_bullish'  # Trend rialzista forte
        elif above_ma_slow and above_ma_longest:
            long_term_trend = 'bullish'  # Trend rialzista
        elif not above_ma_slow and not above_ma_longest and not ma_slow_rising and not ma_longest_rising:
            long_term_trend = 'strong_bearish'  # Trend ribassista forte
        elif not above_ma_slow and not above_ma_longest:
            long_term_trend = 'bearish'  # Trend ribassista
        else:
            long_term_trend = 'neutral'  # Trend laterale o misto
        
        # 5. Inizializza colonna dei segnali
        df['signal'] = 'neutral'
        df['signal_strength'] = 0
        
        # 6. Calcola i segnali in base alle condizioni
        
        # Calcola la distanza dai livelli di supporto e resistenza più vicini
        nearest_support = active_supports[0] if active_supports else df['low'].min() * 0.9
        nearest_resistance = active_resistances[0] if active_resistances else df['high'].max() * 1.1
        
        # Calcola le distanze percentuali
        distance_to_support = (current_price - nearest_support) / current_price * 100
        distance_to_resistance = (nearest_resistance - current_price) / current_price * 100
        
        # Calcola la larghezza del range tra supporto e resistenza
        range_width = (nearest_resistance - nearest_support) / current_price * 100
        
        # Determina la posizione nel range (0 = al supporto, 100 = alla resistenza)
        if nearest_resistance > nearest_support:  # Evita divisione per zero
            position_in_range = (current_price - nearest_support) / (nearest_resistance - nearest_support) * 100
        else:
            position_in_range = 50
        
        # 7. Genera segnali basati sul trend e posizione nel range
        
        # Logica di generazione dei segnali
        
        # Nel trend rialzista (buy):
        # - Acquista vicino ai supporti se il trend è rialzista
        if (long_term_trend in ['bullish', 'strong_bullish']) and (position_in_range < 30):
            df.iloc[-1, df.columns.get_loc('signal')] = 'buy'
            
            # Calcola la forza del segnale
            strength = 60  # Base
            
            # Migliora forza se il trend è molto forte
            if long_term_trend == 'strong_bullish':
                strength += 20
            
            # Migliora forza se molto vicino al supporto
            if position_in_range < 15:
                strength += 15
            
            # Imposta la forza del segnale
            df.iloc[-1, df.columns.get_loc('signal_strength')] = min(100, strength)
        
        # Nel trend ribassista (sell):
        # - Vendi vicino alle resistenze se il trend è ribassista
        elif (long_term_trend in ['bearish', 'strong_bearish']) and (position_in_range > 70):
            df.iloc[-1, df.columns.get_loc('signal')] = 'sell'
            
            # Calcola la forza del segnale
            strength = 60  # Base
            
            # Migliora forza se il trend è molto forte
            if long_term_trend == 'strong_bearish':
                strength += 20
            
            # Migliora forza se molto vicino alla resistenza
            if position_in_range > 85:
                strength += 15
            
            # Imposta la forza del segnale
            df.iloc[-1, df.columns.get_loc('signal_strength')] = min(100, strength)
        
        # 8. Calcola livelli ottimali di ingresso, stop loss e take profit
        
        # Inizializza colonne
        df['entry_point'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Calcola ATR multiplo per il risk management
        atr_value = df['atr'].iloc[-1]
        atr_multiple = params.get('atr_multiple', 3.0)
        
        # Per segnali di acquisto (buy)
        if df['signal'].iloc[-1] == 'buy':
            # Punto di ingresso: prezzo attuale o supporto + piccolo margine
            entry_price = min(current_price, nearest_support * 1.01)
            
            # Stop loss: sotto il supporto (basato su distanza ATR)
            stop_price = entry_price - (atr_value * atr_multiple)
            
            # Take profit: verso la resistenza, con rapporto risk/reward favorevole
            risk = entry_price - stop_price
            reward_multiple = params.get('reward_multiple', 3.0)
            take_profit_price = entry_price + (risk * reward_multiple)
            
            # Limita il take profit alla resistenza se troppo alto
            if take_profit_price > nearest_resistance:
                take_profit_price = nearest_resistance * 0.99
            
            # Imposta i valori
            df.iloc[-1, df.columns.get_loc('entry_point')] = entry_price
            df.iloc[-1, df.columns.get_loc('stop_loss')] = stop_price
            df.iloc[-1, df.columns.get_loc('take_profit')] = take_profit_price
            
        # Per segnali di vendita (sell)
        elif df['signal'].iloc[-1] == 'sell':
            # Punto di ingresso: prezzo attuale o resistenza - piccolo margine
            entry_price = max(current_price, nearest_resistance * 0.99)
            
            # Stop loss: sopra la resistenza (basato su distanza ATR)
            stop_price = entry_price + (atr_value * atr_multiple)
            
            # Take profit: verso il supporto, con rapporto risk/reward favorevole
            risk = stop_price - entry_price
            reward_multiple = params.get('reward_multiple', 3.0)
            take_profit_price = entry_price - (risk * reward_multiple)
            
            # Limita il take profit al supporto se troppo basso
            if take_profit_price < nearest_support:
                take_profit_price = nearest_support * 1.01
            
            # Imposta i valori
            df.iloc[-1, df.columns.get_loc('entry_point')] = entry_price
            df.iloc[-1, df.columns.get_loc('stop_loss')] = stop_price
            df.iloc[-1, df.columns.get_loc('take_profit')] = take_profit_price
        
        # 9. Aggiungi informazioni sui livelli di supporto/resistenza al DataFrame
        
        # Aggiungi le liste dei livelli (fino a 3 per ciascuna direzione)
        df['support_levels'] = np.nan
        df['resistance_levels'] = np.nan
        
        supports_str = ", ".join([f"{s:.2f}" for s in active_supports[:3]])
        resistances_str = ", ".join([f"{r:.2f}" for r in active_resistances[:3]])
        
        df.iloc[-1, df.columns.get_loc('support_levels')] = supports_str
        df.iloc[-1, df.columns.get_loc('resistance_levels')] = resistances_str
        
        # 10. Aggiungi informazioni sul trend
        df['long_term_trend'] = np.nan
        df.iloc[-1, df.columns.get_loc('long_term_trend')] = long_term_trend
        
        return df
        
    def ml_enhanced_trend_following(self, df, **params):
        """
        Strategia di trend following potenziata con analisi avanzata della forza del trend.
        Utilizza combinazioni di indicatori e filtri per confermare la direzione e la forza del trend.
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV e indicatori
        params: Parametri opzionali per personalizzare la strategia
        
        Returns:
        pd.DataFrame: DataFrame con i segnali generati
        """
        df = df.copy()
        
        # Aggiungi indicatori necessari se non presenti
        if 'adx' not in df.columns:
            df = technical_indicators.add_adx(df)
        
        if 'atr' not in df.columns:
            df = technical_indicators.add_atr(df)
            
        if 'supertrend' not in df.columns:
            period = params.get('supertrend_period', 10)
            multiplier = params.get('supertrend_multiplier', 3.0)
            df = self._add_supertrend(df, period, multiplier)
        
        # Inizializza colonna dei segnali
        df['signal'] = 'neutral'
        df['signal_strength'] = 0
        
        # 1. Analisi del trend tramite ADX
        strong_trend = df['adx'] > params.get('adx_threshold', 25)
        very_strong_trend = df['adx'] > params.get('strong_adx_threshold', 40)
        
        # 2. Direzione del trend basata su medie mobili
        uptrend = (df['ma_fast'] > df['ma_slow']) & (df['close'] > df['ma_fast'])
        downtrend = (df['ma_fast'] < df['ma_slow']) & (df['close'] < df['ma_fast'])
        
        # 3. Conferma del trend con Supertrend
        supertrend_buy = df['close'] > df['supertrend']
        supertrend_sell = df['close'] < df['supertrend']
        
        # 4. Conferma con MACD
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd_hist'] > 0)
        macd_sell = (df['macd'] < df['macd_signal']) & (df['macd_hist'] < 0)
        
        # 5. Conferma di momentum con RSI
        rsi_buy = df['rsi'] > params.get('rsi_threshold', 50) 
        rsi_sell = df['rsi'] < params.get('rsi_threshold', 50)
        
        # 6. Calcolo dei livelli chiave per stop loss e take profit basati su ATR
        df['key_level'] = df['close'].rolling(window=20).mean()
        df['atr_multiple'] = df['atr'] * params.get('atr_multiple', 3.0)
        
        # Genera segnali di acquisto
        buy_condition = (
            strong_trend & 
            uptrend & 
            supertrend_buy & 
            (macd_buy | rsi_buy)
        )
        
        # Genera segnali di vendita
        sell_condition = (
            strong_trend & 
            downtrend & 
            supertrend_sell & 
            (macd_sell | rsi_sell)
        )
        
        # Calcola la forza del segnale
        df.loc[buy_condition, 'signal'] = 'buy'
        df.loc[sell_condition, 'signal'] = 'sell'
        
        # Calcola la forza del segnale (0-100)
        df.loc[buy_condition, 'signal_strength'] = (
            (strong_trend.astype(int) * 20) +
            (very_strong_trend.astype(int) * 20) +
            (uptrend.astype(int) * 15) +
            (supertrend_buy.astype(int) * 15) +
            (macd_buy.astype(int) * 15) +
            (rsi_buy.astype(int) * 15)
        ).clip(0, 100)
        
        df.loc[sell_condition, 'signal_strength'] = (
            (strong_trend.astype(int) * 20) +
            (very_strong_trend.astype(int) * 20) +
            (downtrend.astype(int) * 15) +
            (supertrend_sell.astype(int) * 15) +
            (macd_sell.astype(int) * 15) +
            (rsi_sell.astype(int) * 15)
        ).clip(0, 100)
        
        # Aggiungi livelli di entrata, stop loss e take profit
        df['entry_point'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Segnali di acquisto
        df.loc[df['signal'] == 'buy', 'entry_point'] = df['close']
        df.loc[df['signal'] == 'buy', 'stop_loss'] = df['close'] - df['atr_multiple']
        df.loc[df['signal'] == 'buy', 'take_profit'] = df['close'] + (df['atr_multiple'] * 2.5)
        
        # Segnali di vendita
        df.loc[df['signal'] == 'sell', 'entry_point'] = df['close']
        df.loc[df['signal'] == 'sell', 'stop_loss'] = df['close'] + df['atr_multiple']
        df.loc[df['signal'] == 'sell', 'take_profit'] = df['close'] - (df['atr_multiple'] * 2.5)
        
        return df
    
    def volatility_breakout(self, df, **params):
        """
        Strategia di breakout basata sulla volatilità che identifica rotture di importanti livelli di prezzo.
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV e indicatori
        params: Parametri opzionali per personalizzare la strategia
        
        Returns:
        pd.DataFrame: DataFrame con i segnali generati
        """
        df = df.copy()
        
        # Aggiungi ATR se non presente
        if 'atr' not in df.columns:
            df = technical_indicators.add_atr(df)
        
        # Calcola livelli di massimo e minimo significativi
        lookback = params.get('lookback_period', 14)
        
        # Calcola massimi e minimi rilevanti
        df['significant_high'] = df['high'].rolling(window=lookback).max()
        df['significant_low'] = df['low'].rolling(window=lookback).min()
        
        # Calcola la distanza percentuale dai massimi/minimi per identificare la vicinanza a livelli chiave
        df['dist_to_high'] = (df['significant_high'] - df['close']) / df['close'] * 100
        df['dist_to_low'] = (df['close'] - df['significant_low']) / df['close'] * 100
        
        # Determina la volatilità di mercato basata su ATR
        volatility_multiplier = params.get('volatility_multiplier', 2.0)
        df['volatility'] = df['atr'] / df['close'] * 100
        
        # Filtra breakout genuini in base alla volatilità attuale
        df['volatility_threshold'] = df['volatility'] * volatility_multiplier
        
        # Inizializza colonna dei segnali
        df['signal'] = 'neutral'
        df['signal_strength'] = 0
        
        # Identifica breakout verso l'alto (segnale di acquisto)
        breakout_up = (
            (df['close'] > df['significant_high'].shift(1)) & 
            (df['close'] > df['open']) &  # Candela rialzista
            (df['high'] - df['low'] > df['atr'] * 0.8)  # Range significativo
        )
        
        # Identifica breakout verso il basso (segnale di vendita)
        breakout_down = (
            (df['close'] < df['significant_low'].shift(1)) &
            (df['close'] < df['open']) &  # Candela ribassista
            (df['high'] - df['low'] > df['atr'] * 0.8)  # Range significativo
        )
        
        # Filtra breakout in base al volume
        if 'volume' in df.columns:
            # Calcola volume medio
            df['avg_volume'] = df['volume'].rolling(window=lookback).mean()
            
            # Breakout con volume alto sono più significativi
            breakout_up = breakout_up & (df['volume'] > df['avg_volume'] * 1.5)
            breakout_down = breakout_down & (df['volume'] > df['avg_volume'] * 1.5)
        
        # Genera segnali
        df.loc[breakout_up, 'signal'] = 'buy'
        df.loc[breakout_down, 'signal'] = 'sell'
        
        # Calcola forza del segnale basata su:
        # 1. Entità del breakout rispetto all'ATR
        # 2. Volume relativo
        # 3. Trend attuale
        
        # Per i segnali di acquisto
        df.loc[breakout_up, 'signal_strength'] = 70  # Base
        
        # Per i segnali di vendita
        df.loc[breakout_down, 'signal_strength'] = 70  # Base
        
        # Aggiungi boost alla forza del segnale in base alla rottura di resistenza/supporto di lungo periodo
        long_lookback = params.get('long_lookback', 50)
        df['long_high'] = df['high'].rolling(window=long_lookback).max()
        df['long_low'] = df['low'].rolling(window=long_lookback).min()
        
        # Migliora forza del segnale se il breakout è anche su livelli di lungo periodo
        df.loc[(breakout_up) & (df['close'] > df['long_high'].shift(1)), 'signal_strength'] += 20
        df.loc[(breakout_down) & (df['close'] < df['long_low'].shift(1)), 'signal_strength'] += 20
        
        # Aggiungi livelli di entrata, stop loss e take profit
        df['entry_point'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Imposta i livelli per segnali di acquisto
        df.loc[df['signal'] == 'buy', 'entry_point'] = df['close']
        df.loc[df['signal'] == 'buy', 'stop_loss'] = df['close'] - (df['atr'] * params.get('sl_multiple', 1.5))
        df.loc[df['signal'] == 'buy', 'take_profit'] = df['close'] + (df['atr'] * params.get('tp_multiple', 3.0))
        
        # Imposta i livelli per segnali di vendita
        df.loc[df['signal'] == 'sell', 'entry_point'] = df['close']
        df.loc[df['signal'] == 'sell', 'stop_loss'] = df['close'] + (df['atr'] * params.get('sl_multiple', 1.5))
        df.loc[df['signal'] == 'sell', 'take_profit'] = df['close'] - (df['atr'] * params.get('tp_multiple', 3.0))
        
        return df
    
    def mean_reversion(self, df, **params):
        """
        Strategia di ritorno alla media che identifica segnali quando il prezzo si discosta 
        significativamente dalla sua media.
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV e indicatori
        params: Parametri opzionali per personalizzare la strategia
        
        Returns:
        pd.DataFrame: DataFrame con i segnali generati
        """
        df = df.copy()
        
        # Assicurati che le bande di Bollinger siano presenti
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            period = params.get('bollinger_period', 20)
            std_dev = params.get('bollinger_std', 2.0)
            df = technical_indicators.add_bollinger_bands(df, period=period, std_dev=std_dev)
        
        # Aggiungi RSI se non presente
        if 'rsi' not in df.columns:
            df = technical_indicators.add_rsi(df)
        
        # Inizializza colonna dei segnali
        df['signal'] = 'neutral'
        df['signal_strength'] = 0
        
        # Parametri configurabili
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        
        # Calcola indicatori di mean reversion
        
        # 1. Calcola distanza dalle bande di Bollinger come percentuale
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_ma'] * 100
        df['percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
        
        # 2. Identifica condizioni di ipercomprato/ipervenduto
        oversold = (
            (df['close'] < df['bb_lower']) |  # Prezzo sotto la banda inferiore
            (df['percent_b'] < 10) |  # Prezzo molto vicino alla banda inferiore
            (df['rsi'] < rsi_oversold)  # RSI in zona di ipervenduto
        )
        
        overbought = (
            (df['close'] > df['bb_upper']) |  # Prezzo sopra la banda superiore
            (df['percent_b'] > 90) |  # Prezzo molto vicino alla banda superiore
            (df['rsi'] > rsi_overbought)  # RSI in zona di ipercomprato
        )
        
        # 3. Identifica inversioni
        # Inversione da condizione di ipervenduto = segnale di acquisto
        mean_reversion_buy = (
            oversold & 
            (df['close'] > df['close'].shift(1)) &  # Prezzo in aumento
            (df['rsi'] > df['rsi'].shift(1))  # RSI in aumento
        )
        
        # Inversione da condizione di ipercomprato = segnale di vendita
        mean_reversion_sell = (
            overbought & 
            (df['close'] < df['close'].shift(1)) &  # Prezzo in diminuzione
            (df['rsi'] < df['rsi'].shift(1))  # RSI in diminuzione
        )
        
        # 4. Filtra segnali in base alla volatilità
        # In periodi di alta volatilità, i segnali di mean reversion sono meno affidabili
        if 'atr' not in df.columns:
            df = technical_indicators.add_atr(df)
        
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window=100).mean()
        high_volatility = df['volatility_ratio'] > 1.5
        
        # Riduce la forza dei segnali durante alta volatilità
        mean_reversion_buy = mean_reversion_buy & ~high_volatility
        mean_reversion_sell = mean_reversion_sell & ~high_volatility
        
        # Genera segnali
        df.loc[mean_reversion_buy, 'signal'] = 'buy'
        df.loc[mean_reversion_sell, 'signal'] = 'sell'
        
        # Calcola la forza del segnale (0-100)
        
        # Componenti per la forza del segnale di acquisto
        rsi_buy_strength = (rsi_oversold - df['rsi']) / rsi_oversold * 100  # Più basso è l'RSI, più forte il segnale
        bb_buy_strength = (1 - df['percent_b'] / 100) * 100  # Più vicino alla banda inferiore, più forte il segnale
        
        # Componenti per la forza del segnale di vendita
        rsi_sell_strength = (df['rsi'] - rsi_overbought) / (100 - rsi_overbought) * 100
        bb_sell_strength = (df['percent_b'] / 100) * 100
        
        # Calcola punteggio finale
        for i in df.index[df['signal'] == 'buy']:
            rsi_component = min(100, max(0, rsi_buy_strength.loc[i]))
            bb_component = min(100, max(0, bb_buy_strength.loc[i]))
            df.loc[i, 'signal_strength'] = (rsi_component * 0.5 + bb_component * 0.5)
        
        for i in df.index[df['signal'] == 'sell']:
            rsi_component = min(100, max(0, rsi_sell_strength.loc[i]))
            bb_component = min(100, max(0, bb_sell_strength.loc[i]))
            df.loc[i, 'signal_strength'] = (rsi_component * 0.5 + bb_component * 0.5)
        
        # Aggiungi livelli di entrata, stop loss e take profit
        df['entry_point'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Imposta i livelli per segnali di acquisto
        df.loc[df['signal'] == 'buy', 'entry_point'] = df['close']
        df.loc[df['signal'] == 'buy', 'stop_loss'] = df['bb_lower'] * 0.99  # Appena sotto la banda inferiore
        df.loc[df['signal'] == 'buy', 'take_profit'] = df['bb_ma']  # Target = media mobile
        
        # Imposta i livelli per segnali di vendita
        df.loc[df['signal'] == 'sell', 'entry_point'] = df['close']
        df.loc[df['signal'] == 'sell', 'stop_loss'] = df['bb_upper'] * 1.01  # Appena sopra la banda superiore
        df.loc[df['signal'] == 'sell', 'take_profit'] = df['bb_ma']  # Target = media mobile
        
        return df
    
    def pattern_recognition(self, df, **params):
        """
        Strategia basata su pattern di candele e formazioni grafiche avanzate.
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV e indicatori
        params: Parametri opzionali per personalizzare la strategia
        
        Returns:
        pd.DataFrame: DataFrame con i segnali generati
        """
        df = df.copy()
        
        # Inizializza colonna dei segnali
        df['signal'] = 'neutral'
        df['signal_strength'] = 0
        
        # Calcola elementi necessari per il riconoscimento di pattern
        df['body_size'] = abs(df['close'] - df['open'])
        df['total_range'] = df['high'] - df['low']
        df['body_percent'] = (df['body_size'] / df['total_range'] * 100).fillna(0)
        df['upper_shadow'] = (df['high'] - np.maximum(df['close'], df['open']))
        df['lower_shadow'] = (np.minimum(df['close'], df['open']) - df['low'])
        
        # Identifica pattern bullish (per segnali di acquisto)
        
        # 1. Hammer (martello) - corpo piccolo in alto con lunga ombra inferiore
        hammer = (
            (df['body_percent'] < 30) &  # Corpo piccolo
            (df['upper_shadow'] < df['body_size'] * 0.5) &  # Ombra superiore corta
            (df['lower_shadow'] > df['body_size'] * 2) &  # Ombra inferiore lunga (almeno 2x il corpo)
            (df['close'] > df['open'])  # Candela rialzista
        )
        
        # 2. Bullish Engulfing - candela rialzista che ingloba interamente la precedente
        bullish_engulfing = (
            (df['close'] > df['open']) &  # Candela corrente rialzista
            (df['open'].shift(1) > df['close'].shift(1)) &  # Candela precedente ribassista
            (df['open'] < df['close'].shift(1)) &  # Apre sotto la chiusura precedente
            (df['close'] > df['open'].shift(1))  # Chiude sopra l'apertura precedente
        )
        
        # 3. Morning Star - pattern a tre candele, con la centrale piccola
        morning_star = (
            (df['open'].shift(2) > df['close'].shift(2)) &  # Prima candela ribassista
            (df['body_size'].shift(1) < df['body_size'].shift(2) * 0.5) &  # Seconda candela piccola
            (df['close'] > df['open']) &  # Terza candela rialzista
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Chiude oltre metà della prima candela
        )
        
        # 4. Piercing Pattern - due candele dove la seconda apre sotto il minimo della prima ma chiude sopra metà
        piercing_pattern = (
            (df['open'].shift(1) > df['close'].shift(1)) &  # Prima candela ribassista
            (df['open'] < df['low'].shift(1)) &  # Apre sotto il minimo precedente
            (df['close'] > df['open']) &  # Candela corrente rialzista
            (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2)  # Chiude oltre metà della prima candela
        )
        
        # 5. Bullish Harami - candela piccola contenuta nella precedente grande ribassista
        bullish_harami = (
            (df['open'].shift(1) > df['close'].shift(1)) &  # Prima candela ribassista
            (df['body_size'].shift(1) > df['body_size'].shift(2) * 1.5) &  # Prima candela grande
            (df['close'] > df['open']) &  # Seconda candela rialzista
            (df['open'] > df['close'].shift(1)) &  # Apre sopra chiusura precedente
            (df['close'] < df['open'].shift(1))  # Chiude sotto apertura precedente
        )
        
        # Identifica pattern bearish (per segnali di vendita)
        
        # 1. Shooting Star - corpo piccolo in basso con lunga ombra superiore
        shooting_star = (
            (df['body_percent'] < 30) &  # Corpo piccolo
            (df['lower_shadow'] < df['body_size'] * 0.5) &  # Ombra inferiore corta
            (df['upper_shadow'] > df['body_size'] * 2) &  # Ombra superiore lunga (almeno 2x il corpo)
            (df['close'] < df['open'])  # Candela ribassista
        )
        
        # 2. Bearish Engulfing - candela ribassista che ingloba interamente la precedente
        bearish_engulfing = (
            (df['close'] < df['open']) &  # Candela corrente ribassista
            (df['open'].shift(1) < df['close'].shift(1)) &  # Candela precedente rialzista
            (df['open'] > df['close'].shift(1)) &  # Apre sopra la chiusura precedente
            (df['close'] < df['open'].shift(1))  # Chiude sotto l'apertura precedente
        )
        
        # 3. Evening Star - pattern a tre candele, con la centrale piccola
        evening_star = (
            (df['open'].shift(2) < df['close'].shift(2)) &  # Prima candela rialzista
            (df['body_size'].shift(1) < df['body_size'].shift(2) * 0.5) &  # Seconda candela piccola
            (df['close'] < df['open']) &  # Terza candela ribassista
            (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Chiude sotto metà della prima candela
        )
        
        # 4. Dark Cloud Cover - due candele dove la seconda apre sopra il massimo della prima ma chiude sotto metà
        dark_cloud_cover = (
            (df['open'].shift(1) < df['close'].shift(1)) &  # Prima candela rialzista
            (df['open'] > df['high'].shift(1)) &  # Apre sopra il massimo precedente
            (df['close'] < df['open']) &  # Candela corrente ribassista
            (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2)  # Chiude sotto metà della prima candela
        )
        
        # 5. Bearish Harami - candela piccola contenuta nella precedente grande rialzista
        bearish_harami = (
            (df['open'].shift(1) < df['close'].shift(1)) &  # Prima candela rialzista
            (df['body_size'].shift(1) > df['body_size'].shift(2) * 1.5) &  # Prima candela grande
            (df['close'] < df['open']) &  # Seconda candela ribassista
            (df['open'] < df['close'].shift(1)) &  # Apre sotto chiusura precedente
            (df['close'] > df['open'].shift(1))  # Chiude sopra apertura precedente
        )
        
        # Combina pattern bullish
        bullish_patterns = hammer | bullish_engulfing | morning_star | piercing_pattern | bullish_harami
        
        # Combina pattern bearish
        bearish_patterns = shooting_star | bearish_engulfing | evening_star | dark_cloud_cover | bearish_harami
        
        # Calcola importanza del pattern in base al contesto di mercato
        
        # Aggiungi indicatori contestuali se non presenti
        if 'rsi' not in df.columns:
            df = technical_indicators.add_rsi(df)
        
        if 'ma_fast' not in df.columns:
            df = technical_indicators.add_moving_averages(df)
        
        # Valuta l'efficacia del pattern in base a:
        # 1. Posizione nel trend (contrarian vs trend following)
        # 2. Livelli di supporto/resistenza nelle vicinanze
        # 3. Volume della candela del pattern
        
        # Identifica trend
        uptrend = df['ma_fast'] > df['ma_slow']
        downtrend = df['ma_fast'] < df['ma_slow']
        
        # Identifica zone di ipercomprato/ipervenduto
        oversold = df['rsi'] < 30
        overbought = df['rsi'] > 70
        
        # Pattern bullish sono più efficaci in:
        # - Downtrend (per inversione)
        # - Zone di ipervenduto
        bullish_strength = 50.0  # Base
        bullish_strength += downtrend.astype(int) * 20  # +20 in downtrend
        bullish_strength += oversold.astype(int) * 20  # +20 in ipervenduto
        
        # Pattern bearish sono più efficaci in:
        # - Uptrend (per inversione)
        # - Zone di ipercomprato
        bearish_strength = 50.0  # Base
        bearish_strength += uptrend.astype(int) * 20  # +20 in uptrend
        bearish_strength += overbought.astype(int) * 20  # +20 in ipercomprato
        
        # Aggiungi bonus per volume se disponibile
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            bullish_strength += (df['volume_ratio'] > 1.5).astype(int) * 10  # +10 per volume alto
            bearish_strength += (df['volume_ratio'] > 1.5).astype(int) * 10  # +10 per volume alto
        
        # Genera segnali
        df.loc[bullish_patterns, 'signal'] = 'buy'
        df.loc[bearish_patterns, 'signal'] = 'sell'
        
        # Imposta forza del segnale
        for i in df.index[bullish_patterns]:
            df.loc[i, 'signal_strength'] = min(100, bullish_strength.loc[i])
            
        for i in df.index[bearish_patterns]:
            df.loc[i, 'signal_strength'] = min(100, bearish_strength.loc[i])
        
        # Aggiungi livelli di entrata, stop loss e take profit
        df['entry_point'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Per pattern di acquisto
        for i in df.index[df['signal'] == 'buy']:
            prev_idx = max(0, df.index.get_loc(i) - 1)
            prev2_idx = max(0, df.index.get_loc(i) - 2)
            
            entry = df.loc[i, 'close']
            
            # Stop loss sotto il minimo recente
            stop = min(df.loc[prev_idx, 'low'], df.loc[prev2_idx, 'low'], df.loc[i, 'low']) * 0.995
            
            # Target basato sul range recente
            price_range = df.loc[i, 'high'] - stop
            target = entry + (price_range * 2)  # 2:1 risk-reward
            
            df.loc[i, 'entry_point'] = entry
            df.loc[i, 'stop_loss'] = stop
            df.loc[i, 'take_profit'] = target
        
        # Per pattern di vendita
        for i in df.index[df['signal'] == 'sell']:
            prev_idx = max(0, df.index.get_loc(i) - 1)
            prev2_idx = max(0, df.index.get_loc(i) - 2)
            
            entry = df.loc[i, 'close']
            
            # Stop loss sopra il massimo recente
            stop = max(df.loc[prev_idx, 'high'], df.loc[prev2_idx, 'high'], df.loc[i, 'high']) * 1.005
            
            # Target basato sul range recente
            price_range = stop - df.loc[i, 'low']
            target = entry - (price_range * 2)  # 2:1 risk-reward
            
            df.loc[i, 'entry_point'] = entry
            df.loc[i, 'stop_loss'] = stop
            df.loc[i, 'take_profit'] = target
        
        return df
    
    def channel_trading(self, df, **params):
        """
        Strategia basata su canali di prezzo che identifica opportunità di trading 
        ai bordi del canale.
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV e indicatori
        params: Parametri opzionali per personalizzare la strategia
        
        Returns:
        pd.DataFrame: DataFrame con i segnali generati
        """
        df = df.copy()
        
        # Calcola i canali di prezzo
        channel_period = params.get('channel_period', 20)
        
        # Upper channel = massimi del periodo
        df['upper_channel'] = df['high'].rolling(window=channel_period).max()
        
        # Lower channel = minimi del periodo
        df['lower_channel'] = df['low'].rolling(window=channel_period).min()
        
        # Middle channel = media del canale
        df['middle_channel'] = (df['upper_channel'] + df['lower_channel']) / 2
        
        # Channel width come percentuale
        df['channel_width'] = (df['upper_channel'] - df['lower_channel']) / df['middle_channel'] * 100
        
        # Posizione del prezzo all'interno del canale (0-100%)
        df['channel_position'] = (df['close'] - df['lower_channel']) / (df['upper_channel'] - df['lower_channel']) * 100
        df['channel_position'] = df['channel_position'].clip(0, 100)
        
        # Inizializza colonna dei segnali
        df['signal'] = 'neutral'
        df['signal_strength'] = 0
        
        # Identifica condizioni di trading
        
        # Condizioni per segnali di acquisto
        lower_channel_touch = (
            (df['low'] <= df['lower_channel']) &  # Tocca o attraversa il canale inferiore
            (df['close'] > df['lower_channel']) &  # Ma chiude sopra
            (df['channel_position'] < 20)  # Prezzo nella parte bassa del canale
        )
        
        # Condizioni per segnali di vendita
        upper_channel_touch = (
            (df['high'] >= df['upper_channel']) &  # Tocca o attraversa il canale superiore
            (df['close'] < df['upper_channel']) &  # Ma chiude sotto
            (df['channel_position'] > 80)  # Prezzo nella parte alta del canale
        )
        
        # Condizioni aggiuntive per filtrare i segnali
        
        # Conferma da indicatori
        if 'rsi' not in df.columns:
            df = technical_indicators.add_rsi(df)
        
        # Conferma dal trend
        if 'ma_fast' not in df.columns or 'ma_slow' not in df.columns:
            df = technical_indicators.add_moving_averages(df)
        
        # Trend identificato dalle medie mobili
        uptrend = df['ma_fast'] > df['ma_slow']
        downtrend = df['ma_fast'] < df['ma_slow']
        
        # Conferma con RSI
        rsi_oversold = df['rsi'] < 40
        rsi_overbought = df['rsi'] > 60
        
        # Migliora segnali
        # - Buy in uptrend o RSI oversold è più forte
        # - Sell in downtrend o RSI overbought è più forte
        buy_strength = 60.0  # Base
        buy_strength += (uptrend | rsi_oversold).astype(int) * 20  # +20 per trend favorevole
        
        sell_strength = 60.0  # Base
        sell_strength += (downtrend | rsi_overbought).astype(int) * 20  # +20 per trend favorevole
        
        # Genera segnali
        df.loc[lower_channel_touch, 'signal'] = 'buy'
        df.loc[upper_channel_touch, 'signal'] = 'sell'
        
        # Imposta forza dei segnali
        for i in df.index[lower_channel_touch]:
            df.loc[i, 'signal_strength'] = min(100, buy_strength.loc[i])
            
        for i in df.index[upper_channel_touch]:
            df.loc[i, 'signal_strength'] = min(100, sell_strength.loc[i])
        
        # Aggiungi livelli di entrata, stop loss e take profit
        df['entry_point'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Per segnali di acquisto
        df.loc[df['signal'] == 'buy', 'entry_point'] = df['close']
        df.loc[df['signal'] == 'buy', 'stop_loss'] = df['lower_channel'] * 0.99  # Appena sotto il canale
        df.loc[df['signal'] == 'buy', 'take_profit'] = df['middle_channel']  # Target = centro del canale
        
        # Per segnali di vendita
        df.loc[df['signal'] == 'sell', 'entry_point'] = df['close']
        df.loc[df['signal'] == 'sell', 'stop_loss'] = df['upper_channel'] * 1.01  # Appena sopra il canale
        df.loc[df['signal'] == 'sell', 'take_profit'] = df['middle_channel']  # Target = centro del canale
        
        return df
    
    def _add_supertrend(self, df, period=10, multiplier=3.0):
        """
        Aggiunge l'indicatore Supertrend al DataFrame
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV
        period (int): Periodo per il calcolo dell'ATR
        multiplier (float): Moltiplicatore per il calcolo delle bande
        
        Returns:
        pd.DataFrame: DataFrame con l'indicatore Supertrend
        """
        df = df.copy()
        
        # Calcola ATR se non presente
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            df['atr'] = true_range.rolling(period).mean()
        
        # Calcola bande di base
        df['basic_upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
        df['basic_lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])
        
        # Calcola le bande finali con logica di trend
        df['final_upper_band'] = 0.0
        df['final_lower_band'] = 0.0
        
        for i in range(period, len(df)):
            if i == period:
                df.loc[df.index[i], 'final_upper_band'] = df.loc[df.index[i], 'basic_upper_band']
                df.loc[df.index[i], 'final_lower_band'] = df.loc[df.index[i], 'basic_lower_band']
            else:
                # Aggiorna la banda superiore
                if (df.loc[df.index[i], 'basic_upper_band'] < df.loc[df.index[i-1], 'final_upper_band']) or \
                   (df.loc[df.index[i-1], 'close'] > df.loc[df.index[i-1], 'final_upper_band']):
                    df.loc[df.index[i], 'final_upper_band'] = df.loc[df.index[i], 'basic_upper_band']
                else:
                    df.loc[df.index[i], 'final_upper_band'] = df.loc[df.index[i-1], 'final_upper_band']
                
                # Aggiorna la banda inferiore
                if (df.loc[df.index[i], 'basic_lower_band'] > df.loc[df.index[i-1], 'final_lower_band']) or \
                   (df.loc[df.index[i-1], 'close'] < df.loc[df.index[i-1], 'final_lower_band']):
                    df.loc[df.index[i], 'final_lower_band'] = df.loc[df.index[i], 'basic_lower_band']
                else:
                    df.loc[df.index[i], 'final_lower_band'] = df.loc[df.index[i-1], 'final_lower_band']
        
        # Calcola il Supertrend
        df['supertrend'] = 0.0
        df['supertrend_direction'] = 0  # 1: bullish, -1: bearish
        
        for i in range(period, len(df)):
            if i == period:
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_lower_band']
                df.loc[df.index[i], 'supertrend_direction'] = 1
            else:
                previous_supertrend = df.loc[df.index[i-1], 'supertrend']
                current_close = df.loc[df.index[i], 'close']
                upper_band = df.loc[df.index[i], 'final_upper_band']
                lower_band = df.loc[df.index[i], 'final_lower_band']
                
                # Logica del Supertrend
                if previous_supertrend == df.loc[df.index[i-1], 'final_upper_band']:
                    if current_close <= upper_band:
                        df.loc[df.index[i], 'supertrend'] = upper_band
                        df.loc[df.index[i], 'supertrend_direction'] = -1
                    else:
                        df.loc[df.index[i], 'supertrend'] = lower_band
                        df.loc[df.index[i], 'supertrend_direction'] = 1
                elif previous_supertrend == df.loc[df.index[i-1], 'final_lower_band']:
                    if current_close >= lower_band:
                        df.loc[df.index[i], 'supertrend'] = lower_band
                        df.loc[df.index[i], 'supertrend_direction'] = 1
                    else:
                        df.loc[df.index[i], 'supertrend'] = upper_band
                        df.loc[df.index[i], 'supertrend_direction'] = -1
        
        # Rimuovi colonne temporanee
        df = df.drop(['basic_upper_band', 'basic_lower_band', 'final_upper_band', 'final_lower_band'], axis=1)
        
        return df
    
    def apply_strategy(self, df, strategy, **params):
        """
        Applica una strategia di trading selezionata al DataFrame
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV
        strategy (str): Nome della strategia da applicare
        params: Parametri opzionali per la strategia
        
        Returns:
        pd.DataFrame: DataFrame con i segnali generati
        """
        # Assicurati che il DataFrame contenga i dati di base necessari
        if df is None or df.empty:
            raise ValueError("DataFrame non valido")
        
        # Verifica che il DataFrame contenga le colonne necessarie
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Colonna richiesta mancante: {col}")
        
        # Aggiungi indicatori tecnici di base se non presenti
        df = self._ensure_base_indicators(df)
        
        # Seleziona e applica la strategia
        strategy_method = getattr(self, strategy, None)
        if strategy_method is None:
            raise ValueError(f"Strategia non valida: {strategy}")
        
        df_with_signals = strategy_method(df, **params)
        
        return df_with_signals
    
    def _ensure_base_indicators(self, df):
        """
        Assicura che il DataFrame contenga gli indicatori tecnici di base necessari
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV
        
        Returns:
        pd.DataFrame: DataFrame con gli indicatori tecnici aggiunti
        """
        # Aggiungi gli indicatori di base se non presenti
        if 'rsi' not in df.columns:
            df = technical_indicators.add_rsi(df)
        
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            df = technical_indicators.add_macd(df)
        
        if 'ma_fast' not in df.columns or 'ma_slow' not in df.columns:
            df = technical_indicators.add_moving_averages(df)
        
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            df = technical_indicators.add_bollinger_bands(df)
        
        return df
    
    def get_best_strategy(self, df, timeframe):
        """
        Determina la migliore strategia di trading da applicare in base alle condizioni di mercato attuali
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV
        timeframe (str): Timeframe dei dati
        
        Returns:
        str: Nome della strategia migliore
        dict: Parametri ottimali per la strategia
        """
        df = self._ensure_base_indicators(df)
        
        # Calcola metriche sul mercato attuale
        
        # 1. Trend Analysis
        ma_fast = df['ma_fast'].iloc[-1]
        ma_slow = df['ma_slow'].iloc[-1]
        trend_direction = 1 if ma_fast > ma_slow else -1 if ma_fast < ma_slow else 0
        
        # 2. Volatility Analysis
        if 'atr' not in df.columns:
            df = technical_indicators.add_atr(df)
        
        # Calcola volatilità relativa
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].rolling(window=50).mean().iloc[-1]
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # 3. Momentum Analysis
        current_rsi = df['rsi'].iloc[-1]
        
        # 4. Volume Analysis (se disponibile)
        volume_spike = False
        if 'volume' in df.columns:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            volume_spike = volume_ratio > 1.5
        
        # Determina la strategia migliore in base alle condizioni attuali
        
        # Condizioni di mercato: alta volatilità, breakout recente
        if volatility_ratio > 1.3 or volume_spike:
            # In caso di alta volatilità o spike di volume, la strategia di volatility breakout è efficace
            return 'volatility_breakout', {
                'lookback_period': 20 if timeframe in ['1m', '5m', '15m'] else 14,
                'volatility_multiplier': 2.0,
                'sl_multiple': 1.5,
                'tp_multiple': 3.0
            }
        
        # Condizioni di mercato: trend forte
        elif (trend_direction != 0) and (abs(ma_fast/ma_slow - 1) * 100 > 2):
            # In caso di trend forte, la strategia di trend following è efficace
            return 'ml_enhanced_trend_following', {
                'adx_threshold': 25,
                'strong_adx_threshold': 40,
                'rsi_threshold': 50,
                'supertrend_period': 10,
                'supertrend_multiplier': 3.0,
                'atr_multiple': 2.5
            }
        
        # Condizioni di mercato: mercato laterale o range-bound
        elif volatility_ratio < 0.8:
            # In caso di bassa volatilità, la strategia di mean reversion è efficace
            return 'mean_reversion', {
                'bollinger_period': 20,
                'bollinger_std': 2.0,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
        
        # Condizioni di mercato: ipercomprato/ipervenduto
        elif current_rsi < 30 or current_rsi > 70:
            # In caso di RSI estremo, la strategia di pattern recognition può trovare inversioni
            return 'pattern_recognition', {}
        
        # Condizioni di mercato: trend in sviluppo
        else:
            # In altri casi, la strategia di channel trading è un buon compromesso
            return 'channel_trading', {
                'channel_period': 20
            }
    
    def get_multiple_strategy_signals(self, df, timeframe, strategies=None, max_results=5):
        """
        Applica multiple strategie e restituisce i migliori segnali
        
        Parameters:
        df (pd.DataFrame): DataFrame con i dati OHLCV
        timeframe (str): Timeframe dei dati
        strategies (list): Lista di strategie da applicare (se None, usa tutte)
        max_results (int): Numero massimo di segnali da restituire
        
        Returns:
        list: Lista di segnali dalle varie strategie
        """
        df = self._ensure_base_indicators(df)
        
        # Se non sono specificate strategie, usa tutte quelle disponibili
        if strategies is None:
            strategies = self.available_strategies
        
        all_signals = []
        
        # Applica ogni strategia
        for strategy in strategies:
            try:
                # Ottieni parametri ottimali per la strategia
                _, params = self.get_best_strategy(df, timeframe)
                
                # Applica la strategia
                df_with_signals = self.apply_strategy(df, strategy, **params)
                
                # Estrai i segnali recenti
                recent_signals = self._extract_recent_signals(df_with_signals, strategy, timeframe)
                
                # Aggiungi i segnali alla lista
                all_signals.extend(recent_signals)
            except Exception as e:
                print(f"Errore nell'applicare la strategia {strategy}: {e}")
        
        # Ordina i segnali per forza (discendente)
        all_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        
        # Limita il numero di risultati
        return all_signals[:max_results]
    
    def _extract_recent_signals(self, df, strategy, timeframe, lookback=5):
        """
        Estrae i segnali recenti dal DataFrame
        
        Parameters:
        df (pd.DataFrame): DataFrame con i segnali
        strategy (str): Nome della strategia usata
        timeframe (str): Timeframe dei dati
        lookback (int): Numero di barre da considerare (da quella più recente)
        
        Returns:
        list: Lista di segnali recenti
        """
        signals = []
        
        # Adatta il lookback in base al timeframe
        if timeframe in ['1d', '4h']:
            lookback = min(3, lookback)  # Meno barre per timeframe più lunghi
        elif timeframe in ['1w', '1M']:
            lookback = 1  # Solo la barra più recente per timeframe molto lunghi
        
        # Estrai segnali dalle barre recenti
        for i in range(min(lookback, len(df))):
            idx = -i-1  # Indice dalla fine
            
            if df['signal'].iloc[idx] != 'neutral':
                signal_data = {
                    'signal_type': df['signal'].iloc[idx],
                    'price': df['close'].iloc[idx],
                    'timestamp': df.index[idx],
                    'timeframe': timeframe,
                    'strategy': strategy,
                    'signal_strength': df['signal_strength'].iloc[idx],
                    'entry_point': df['entry_point'].iloc[idx],
                    'stop_loss': df['stop_loss'].iloc[idx],
                    'take_profit': df['take_profit'].iloc[idx],
                    'bars_ago': i
                }
                
                # Calcola rapporto rischio/rendimento
                if signal_data['signal_type'] == 'buy':
                    risk = signal_data['entry_point'] - signal_data['stop_loss']
                    reward = signal_data['take_profit'] - signal_data['entry_point']
                else:  # 'sell'
                    risk = signal_data['stop_loss'] - signal_data['entry_point']
                    reward = signal_data['entry_point'] - signal_data['take_profit']
                
                # Evita divisione per zero
                if risk > 0:
                    signal_data['risk_reward_ratio'] = reward / risk
                else:
                    signal_data['risk_reward_ratio'] = 0
                
                signals.append(signal_data)
        
        return signals

# Funzioni di utility per l'accesso esterno
def analyze_crypto_with_multiple_algorithms(symbol, timeframe, min_signal_strength=60):
    """
    Analizza una criptovaluta con algoritmi avanzati di trading
    
    Parameters:
    symbol (str): Simbolo della criptovaluta (es. 'BTC/USDT')
    timeframe (str): Timeframe da analizzare
    min_signal_strength (int): Forza minima del segnale (0-100)
    
    Returns:
    list: Lista di segnali di trading
    """
    try:
        # Importa i moduli necessari
        import crypto_data
        import technical_indicators
        
        # Ottieni i dati OHLCV
        df = crypto_data.fetch_ohlcv_data(symbol, timeframe, limit=200)
        
        if df is None or df.empty:
            print(f"Nessun dato disponibile per {symbol} su timeframe {timeframe}")
            return []
        
        # Aggiungi indicatori tecnici
        df = technical_indicators.add_all_indicators(df)
        
        # Crea istanza dell'algoritmo avanzato
        algo = AdvancedTradingAlgorithm()
        
        # Seleziona la strategia migliore per le condizioni attuali di mercato
        best_strategy, params = algo.get_best_strategy(df, timeframe)
        
        # Ottieni segnali da multiple strategie
        signals = algo.get_multiple_strategy_signals(df, timeframe)
        
        # Filtra i segnali in base alla forza minima
        strong_signals = [s for s in signals if s['signal_strength'] >= min_signal_strength]
        
        return strong_signals
    except Exception as e:
        print(f"Errore nell'analisi di {symbol}: {e}")
        return []

def analyze_market_with_advanced_algorithms(symbols, timeframes=None, limit=10):
    """
    Analizza più criptovalute con algoritmi avanzati per trovare le migliori opportunità
    
    Parameters:
    symbols (list): Lista di simboli da analizzare
    timeframes (list): Lista di timeframe da analizzare (default: 1h, 4h, 1d)
    limit (int): Limite di opportunità da restituire
    
    Returns:
    dict: Opportunità di trading divise per tipo (buy/sell)
    """
    if timeframes is None:
        timeframes = ['1h', '4h', '1d']
    
    all_signals = []
    
    # Analizza ogni simbolo su ogni timeframe
    for symbol in symbols:
        for timeframe in timeframes:
            signals = analyze_crypto_with_multiple_algorithms(symbol, timeframe)
            all_signals.extend(signals)
    
    # Ordina i segnali per forza
    all_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
    
    # Dividi in segnali di acquisto e vendita
    buy_signals = [s for s in all_signals if s['signal_type'] == 'buy']
    sell_signals = [s for s in all_signals if s['signal_type'] == 'sell']
    
    # Limita il numero di risultati
    buy_signals = buy_signals[:limit]
    sell_signals = sell_signals[:limit]
    
    # Arricchisci i segnali con informazioni aggiuntive
    try:
        market_sent = market_sentiment.get_market_sentiment()
        sentiment_score = market_sent.get('score', 50) if market_sent else 50
        market_direction = market_sent.get('direction', 'Neutral') if market_sent else 'Neutral'
    except:
        sentiment_score = 50
        market_direction = 'Neutral'
    
    # Aggiungi informazioni di sentiment ai segnali
    for signal in buy_signals + sell_signals:
        signal['market_sentiment'] = sentiment_score
        signal['market_direction'] = market_direction
        
        # Determina allineamento col sentiment
        if (sentiment_score > 60 and signal['signal_type'] == 'buy') or \
           (sentiment_score < 40 and signal['signal_type'] == 'sell'):
            signal['sentiment_alignment'] = True
        else:
            signal['sentiment_alignment'] = False
    
    return {
        'buy': buy_signals,
        'sell': sell_signals,
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_sentiment': sentiment_score,
        'market_direction': market_direction
    }

def get_signal_explanation(signal):
    """
    Fornisce una spiegazione dettagliata di un segnale di trading
    
    Parameters:
    signal (dict): Segnale di trading
    
    Returns:
    str: Spiegazione dettagliata
    """
    # Determina il tipo di segnale
    signal_type = signal['signal_type']
    strategy = signal.get('strategy', 'unknown')
    strength = signal.get('signal_strength', 0)
    
    # Mappa delle strategie
    strategy_names = {
        'ml_enhanced_trend_following': 'Trend Following Avanzato',
        'volatility_breakout': 'Breakout di Volatilità',
        'mean_reversion': 'Ritorno alla Media',
        'smart_momentum': 'Momentum Intelligente',
        'channel_trading': 'Trading di Canale',
        'multi_timeframe_confluence': 'Confluenza Multi-Timeframe',
        'advanced_support_resistance': 'Supporti e Resistenze Avanzati',
        'volume_profile_analysis': 'Analisi del Profilo del Volume',
        'pattern_recognition': 'Riconoscimento Pattern'
    }
    
    # Mappa delle strategie alle spiegazioni
    strategy_explanations = {
        'ml_enhanced_trend_following': 'identificazione di un trend forte con conferma da multipli indicatori tecnici',
        'volatility_breakout': 'rottura di livelli significativi di prezzo con aumento di volume',
        'mean_reversion': 'prezzo che si allontana troppo dalla media e mostra segni di inversione',
        'smart_momentum': 'forte slancio del prezzo con conferma da oscillatori',
        'channel_trading': 'prezzo che tocca i limiti di un canale di prezzo ben definito',
        'multi_timeframe_confluence': 'allineamento di segnali su diversi timeframe',
        'advanced_support_resistance': 'reazione a livelli critici di supporto/resistenza',
        'volume_profile_analysis': 'reazione a livelli di prezzo con volume significativo',
        'pattern_recognition': 'formazione di pattern di candele affidabili'
    }
    
    # Costruisci la spiegazione
    explanation = f"Questo è un segnale di {'ACQUISTO' if signal_type == 'buy' else 'VENDITA'} "
    explanation += f"generato utilizzando la strategia di {strategy_names.get(strategy, strategy)}.\n\n"
    
    explanation += f"Il segnale ha una forza di {strength}/100 e si basa su "
    explanation += f"{strategy_explanations.get(strategy, 'analisi tecnica avanzata')}.\n\n"
    
    # Aggiungi informazioni sui livelli
    explanation += f"🎯 Punto di ingresso suggerito: {signal['entry_point']:.4f}\n"
    explanation += f"🛑 Stop Loss consigliato: {signal['stop_loss']:.4f}\n"
    explanation += f"💰 Take Profit suggerito: {signal['take_profit']:.4f}\n\n"
    
    # Calcola rapporto rischio/rendimento
    if signal_type == 'buy':
        risk = signal['entry_point'] - signal['stop_loss']
        reward = signal['take_profit'] - signal['entry_point']
        potential_gain = ((signal['take_profit'] / signal['entry_point']) - 1) * 100
        potential_loss = ((signal['stop_loss'] / signal['entry_point']) - 1) * 100
    else:  # 'sell'
        risk = signal['stop_loss'] - signal['entry_point']
        reward = signal['entry_point'] - signal['take_profit']
        potential_gain = ((signal['entry_point'] / signal['take_profit']) - 1) * 100
        potential_loss = ((signal['entry_point'] / signal['stop_loss']) - 1) * 100
    
    rr_ratio = reward / risk if risk > 0 else 0
    
    explanation += f"📊 Rapporto Rischio/Rendimento: 1:{rr_ratio:.2f}\n"
    explanation += f"📈 Potenziale guadagno: {abs(potential_gain):.2f}%\n"
    explanation += f"📉 Potenziale perdita: {abs(potential_loss):.2f}%\n\n"
    
    # Aggiungi suggerimento sul timeframe
    explanation += f"⏰ Timeframe: {signal['timeframe']}\n"
    explanation += f"🕑 Generato: {signal.get('bars_ago', 0)} barre fa\n\n"
    
    # Aggiungi allineamento col sentiment di mercato
    market_sentiment = signal.get('market_sentiment', 50)
    market_direction = signal.get('market_direction', 'Neutral')
    sentiment_aligned = signal.get('sentiment_alignment', False)
    
    explanation += f"🌎 Sentiment di mercato attuale: {market_direction} ({market_sentiment}/100)\n"
    if sentiment_aligned:
        explanation += "✅ Questo segnale è allineato con il sentiment di mercato generale\n"
    else:
        explanation += "⚠️ Questo segnale va contro il sentiment di mercato generale\n"
    
    return explanation