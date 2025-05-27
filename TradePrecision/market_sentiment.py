import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class MarketSentimentAnalyzer:
    """
    Analizzatore del sentiment di mercato in tempo reale per criptovalute.
    Combina più fonti di dati per generare un indicatore di sentiment completo.
    """
    
    def __init__(self, exchange_id='kraken', max_symbols=15):
        """
        Inizializza l'analizzatore di sentiment di mercato
        
        Parameters:
        exchange_id (str): ID dell'exchange da utilizzare (default: 'kraken')
        max_symbols (int): Numero massimo di simboli da analizzare
        """
        self.exchange_id = exchange_id
        self.exchange = ccxt.kraken({
            'enableRateLimit': True,
            'timeout': 30000  # 30 secondi timeout
        })
        self.max_symbols = max_symbols
        self.timeframes = ['1h', '4h', '1d']
        self.top_market_cap = []
        self.cached_sentiment = None
        self.cache_time = None
        self.cache_duration = 300  # 5 minuti - aumentato per ridurre le chiamate API
        
        # Default market cap data per maggiori criptovalute se l'API esterna fallisce
        self.default_markets = [
            {'symbol': 'BTC/USD', 'marketCap': 1000000000000, 'weight': 0.35},
            {'symbol': 'ETH/USD', 'marketCap': 300000000000, 'weight': 0.25},
            {'symbol': 'XRP/USD', 'marketCap': 50000000000, 'weight': 0.1},
            {'symbol': 'SOL/USD', 'marketCap': 40000000000, 'weight': 0.1},
            {'symbol': 'DOGE/USD', 'marketCap': 15000000000, 'weight': 0.05},
            {'symbol': 'DOT/USD', 'marketCap': 12000000000, 'weight': 0.05},
            {'symbol': 'ADA/USD', 'marketCap': 10000000000, 'weight': 0.05},
            {'symbol': 'LTC/USD', 'marketCap': 5000000000, 'weight': 0.05}
        ]
    
    def get_market_cap_data(self):
        """
        Ottiene i dati di market cap per le principali criptovalute
        
        Returns:
        list: Lista di dizionari con i dati di market cap
        """
        try:
            # Se non possiamo ottenere i dati, usa i default
            if not hasattr(self, 'default_markets'):
                self.default_markets = []
            
            if len(self.default_markets) > 0:
                # Utilizziamo i dati default se disponibili
                return self.default_markets
                
            # Utilizzo dei dati di mercato disponibili tramite l'API dell'exchange
            markets = self.exchange.fetch_tickers()
            markets_list = []
            
            # Carica i mercati se non è stato fatto
            if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
                self.exchange.load_markets()
            
            for symbol, ticker in markets.items():
                # Kraken usa principalmente USD invece di USDT
                if symbol.endswith('/USD'):
                    base_symbol = symbol.split('/')[0]
                    
                    # Verifica che il mercato sia attivo
                    is_active = self.exchange.markets.get(symbol, {}).get('active', False)
                    if not is_active:
                        continue
                    
                    # Per Kraken, usiamo baseVolume * price come approssimazione del volume in USD
                    price = ticker.get('last', 0)
                    base_volume = ticker.get('baseVolume', 0)
                    
                    if not price or not base_volume:
                        continue
                        
                    # Stima del volume in USD e market cap
                    usd_volume = base_volume * price
                    estimated_market_cap = usd_volume * 100  # Stima approssimativa
                    
                    # Attribuisce un peso maggiore alle principali criptovalute
                    weight = 0.05  # Default
                    if 'BTC' in symbol:
                        weight = 0.35
                        estimated_market_cap *= 10
                    elif 'ETH' in symbol:
                        weight = 0.25
                        estimated_market_cap *= 8
                    elif any(s in symbol for s in ['SOL', 'XRP']):
                        weight = 0.1
                        estimated_market_cap *= 5
                    elif any(s in symbol for s in ['DOGE', 'ADA', 'DOT']):
                        weight = 0.05
                        estimated_market_cap *= 3
                    
                    if estimated_market_cap > 0:
                        markets_list.append({
                            'symbol': symbol,
                            'price': price,
                            'volume': usd_volume,
                            'marketCap': estimated_market_cap,
                            'weight': weight
                        })
            
            # Ordina per market cap stimato
            markets_list.sort(key=lambda x: x['marketCap'], reverse=True)
            result = markets_list[:self.max_symbols]
            
            # Se abbiamo ottenuto dati, aggiorniamo i default
            if result:
                self.default_markets = result
                
            return result
            
        except Exception as e:
            print(f"Errore nel recupero dei dati di market cap: {e}")
            # Ritorna i dati default se l'API fallisce
            return self.default_markets

    def fetch_ohlcv_for_symbol(self, symbol, timeframe='1h', limit=100):
        """
        Recupera i dati OHLCV per un simbolo specifico, utilizzando il modulo crypto_data
        che include caching ottimizzato
        
        Parameters:
        symbol (str): Simbolo della criptovaluta
        timeframe (str): Timeframe dei dati
        limit (int): Numero di candele da recuperare
        
        Returns:
        pd.DataFrame: DataFrame con i dati OHLCV
        """
        try:
            # Usa il modulo crypto_data per sfruttare il caching
            import crypto_data
            df = crypto_data.fetch_ohlcv_data(symbol, timeframe, limit=limit, force_refresh=False, exchange_id=self.exchange_id)
            
            if df is None:
                return pd.DataFrame()
                
            return df
        except Exception as e:
            print(f"Errore nel recupero dei dati OHLCV per {symbol}: {e}")
            return pd.DataFrame()

    def calculate_technical_metrics(self, df):
        """
        Calcola metriche tecniche dal DataFrame OHLCV
        
        Parameters:
        df (pd.DataFrame): DataFrame con dati OHLCV
        
        Returns:
        dict: Dizionario con le metriche tecniche
        """
        if df.empty:
            return {
                'rsi': 50,
                'price_trend': 0,
                'volume_trend': 0,
                'volatility': 0
            }
        
        # Calcola RSI semplificato
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # Calcola tendenza del prezzo (% di variazione nelle ultime 24 candele)
        price_change = ((df['close'].iloc[-1] / df['close'].iloc[-24]) - 1) * 100 if len(df) >= 24 else 0
        
        # Calcola tendenza del volume
        volume_change = ((df['volume'].iloc[-1] / df['volume'].iloc[-24]) - 1) * 100 if len(df) >= 24 else 0
        
        # Calcola volatilità (deviazione standard dei rendimenti)
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * 100  # come percentuale
        
        return {
            'rsi': current_rsi,
            'price_trend': price_change,
            'volume_trend': volume_change,
            'volatility': volatility
        }

    def fetch_market_data_parallel(self, symbols, timeframe):
        """
        Recupera dati di mercato in parallelo per più simboli
        
        Parameters:
        symbols (list): Lista di simboli da analizzare
        timeframe (str): Timeframe dei dati
        
        Returns:
        list: Lista di dizionari con dati di mercato
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_ohlcv_for_symbol, symbol['symbol'], timeframe): symbol['symbol']
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        metrics = self.calculate_technical_metrics(df)
                        results.append({
                            'symbol': symbol,
                            'metrics': metrics,
                            'timeframe': timeframe
                        })
                except Exception as e:
                    print(f"Errore nell'elaborazione di {symbol}: {e}")
        
        return results

    def calculate_overall_sentiment(self, market_data):
        """
        Calcola il sentiment complessivo del mercato
        
        Parameters:
        market_data (list): Lista di dati di mercato
        
        Returns:
        dict: Dizionario con le metriche di sentiment
        """
        if not market_data:
            return {
                'bullish_percent': 0,
                'bearish_percent': 0,
                'neutral_percent': 100,
                'overall_score': 50,
                'market_direction': 'Neutral',
                'market_strength': 'Weak',
                'top_gainers': [],
                'top_losers': [],
                'high_volume': [],
                'timestamp': datetime.now().isoformat()
            }
        
        # Inizializza contatori
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_count = len(market_data)
        
        # Lista per tracciare le performance
        performance_list = []
        volume_list = []
        
        # Analizza ogni asset
        for data in market_data:
            metrics = data['metrics']
            symbol = data['symbol']
            
            # Determina il sentiment per questo asset
            rsi = metrics['rsi']
            price_trend = metrics['price_trend']
            
            # Aggiungi alla lista delle performance
            performance_list.append({
                'symbol': symbol,
                'price_change': price_trend,
                'rsi': rsi,
                'timeframe': data['timeframe']
            })
            
            # Aggiungi alla lista del volume
            volume_list.append({
                'symbol': symbol,
                'volume_change': metrics['volume_trend'],
                'timeframe': data['timeframe']
            })
            
            # Calcola sentiment basato su RSI e tendenza del prezzo
            if (rsi > 60 and price_trend > 1) or price_trend > 3:
                bullish_count += 1
            elif (rsi < 40 and price_trend < -1) or price_trend < -3:
                bearish_count += 1
            else:
                neutral_count += 1
        
        # Calcola percentuali
        bullish_percent = (bullish_count / total_count) * 100 if total_count > 0 else 0
        bearish_percent = (bearish_count / total_count) * 100 if total_count > 0 else 0
        neutral_percent = (neutral_count / total_count) * 100 if total_count > 0 else 0
        
        # Calcola punteggio complessivo (0-100, dove 50 è neutrale)
        overall_score = 50 + ((bullish_percent - bearish_percent) / 2)
        
        # Determina la direzione del mercato
        if overall_score >= 65:
            market_direction = "Bullish"
        elif overall_score >= 58:
            market_direction = "Moderately Bullish"
        elif overall_score >= 52:
            market_direction = "Slightly Bullish"
        elif overall_score >= 48:
            market_direction = "Neutral"
        elif overall_score >= 42:
            market_direction = "Slightly Bearish"
        elif overall_score >= 35:
            market_direction = "Moderately Bearish"
        else:
            market_direction = "Bearish"
        
        # Determina la forza del mercato (quanto è forte il sentiment)
        strength_score = abs(overall_score - 50)
        if strength_score >= 20:
            market_strength = "Very Strong"
        elif strength_score >= 12:
            market_strength = "Strong"
        elif strength_score >= 6:
            market_strength = "Moderate"
        else:
            market_strength = "Weak"
        
        # Identifica top gainers e losers
        performance_list.sort(key=lambda x: x['price_change'], reverse=True)
        top_gainers = performance_list[:3] if len(performance_list) >= 3 else performance_list
        top_losers = performance_list[-3:] if len(performance_list) >= 3 else []
        top_losers.reverse()
        
        # Identifica asset con alto volume
        volume_list.sort(key=lambda x: x['volume_change'], reverse=True)
        high_volume = volume_list[:3] if len(volume_list) >= 3 else volume_list
        
        return {
            'bullish_percent': bullish_percent,
            'bearish_percent': bearish_percent,
            'neutral_percent': neutral_percent,
            'overall_score': overall_score,
            'market_direction': market_direction,
            'market_strength': market_strength,
            'top_gainers': top_gainers,
            'top_losers': top_losers,
            'high_volume': high_volume,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_real_time_sentiment(self, force_refresh=False):
        """
        Ottiene il sentiment di mercato in tempo reale
        
        Parameters:
        force_refresh (bool): Se True, forza l'aggiornamento del sentiment
        
        Returns:
        dict: Dizionario con le metriche di sentiment
        """
        current_time = time.time()
        
        # Se i dati in cache sono validi e non è richiesto un aggiornamento forzato, utilizza la cache
        if not force_refresh and self.cached_sentiment and self.cache_time and (current_time - self.cache_time < self.cache_duration):
            return self.cached_sentiment
        
        # Altrimenti, ricalcola il sentiment
        # Ottieni le principali criptovalute per market cap
        top_coins = self.get_market_cap_data()
        
        if not top_coins:
            return {
                'error': 'Non è stato possibile recuperare i dati di market cap',
                'bullish_percent': 0,
                'bearish_percent': 0,
                'neutral_percent': 100,
                'overall_score': 50,
                'market_direction': 'Neutral',
                'market_strength': 'Weak',
                'timestamp': datetime.now().isoformat()
            }
        
        # Raccoglie dati per diversi timeframe per un'analisi più completa
        market_data = []
        for timeframe in self.timeframes:
            timeframe_data = self.fetch_market_data_parallel(top_coins, timeframe)
            market_data.extend(timeframe_data)
        
        # Calcola il sentiment complessivo
        sentiment = self.calculate_overall_sentiment(market_data)
        
        # Aggiorna la cache
        self.cached_sentiment = sentiment
        self.cache_time = current_time
        
        return sentiment
    
    def get_simplified_sentiment(self):
        """
        Ottiene una versione semplificata del sentiment per la visualizzazione
        
        Returns:
        dict: Dizionario con il sentiment semplificato
        """
        sentiment = self.get_real_time_sentiment()
        
        if 'error' in sentiment:
            return {
                'score': 50,
                'direction': 'Neutral',
                'strength': 'Weak',
                'updated': datetime.now().strftime('%H:%M:%S')
            }
        
        return {
            'score': round(sentiment['overall_score']),
            'direction': sentiment['market_direction'],
            'strength': sentiment['market_strength'],
            'bullish': round(sentiment['bullish_percent']),
            'bearish': round(sentiment['bearish_percent']),
            'neutral': round(sentiment['neutral_percent']),
            'top_gainers': [{'symbol': item['symbol'].split('/')[0], 'change': round(item['price_change'], 1)} 
                          for item in sentiment['top_gainers']],
            'top_losers': [{'symbol': item['symbol'].split('/')[0], 'change': round(item['price_change'], 1)} 
                         for item in sentiment['top_losers']],
            'updated': datetime.now().strftime('%H:%M:%S')
        }


# Funzione per ottenere il sentiment aggregato da usare nell'app
def get_market_sentiment():
    """
    Ottiene il sentiment di mercato corrente
    
    Returns:
    dict: Dizionario con il sentiment di mercato
    """
    analyzer = MarketSentimentAnalyzer()
    return analyzer.get_simplified_sentiment()


# Per debug e test
if __name__ == "__main__":
    analyzer = MarketSentimentAnalyzer()
    sentiment = analyzer.get_simplified_sentiment()
    print("Market Sentiment Summary:")
    print(f"Score: {sentiment['score']}/100")
    print(f"Direction: {sentiment['direction']}")
    print(f"Strength: {sentiment['strength']}")
    print(f"Bullish: {sentiment['bullish']}% | Bearish: {sentiment['bearish']}% | Neutral: {sentiment['neutral']}%")
    print("\nTop Gainers:")
    for coin in sentiment['top_gainers']:
        print(f"{coin['symbol']}: {coin['change']}%")
    print("\nTop Losers:")
    for coin in sentiment['top_losers']:
        print(f"{coin['symbol']}: {coin['change']}%")