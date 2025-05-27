import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import technical_indicators
import signal_generator
import signal_validator
import market_sentiment
import utils

class MarketAnalyzer:
    """
    Analizzatore automatico del mercato che identifica le migliori opportunità di trading
    tra tutte le criptovalute disponibili.
    """
    
    def __init__(self, exchange_id='kraken', max_symbols=50):
        """
        Inizializza l'analizzatore di mercato
        
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
        self.timeframes = ['15m', '1h', '4h', '1d', '1w']
        self.default_timeframe = '1h'
        self.opportunities = []
        self.last_analysis_time = None
        self.analysis_valid_period = 900  # 15 minuti in secondi
        
        # Cache dei simboli per migliorare le prestazioni
        self._symbols_cache = []
        self._cache_timestamp = 0
        self._cache_ttl = 3600  # 1 ora
    
    def get_tradable_symbols(self):
        """
        Ottiene una lista di simboli scambiabili con buona liquidità
        
        Returns:
        list: Lista di simboli scambiabili
        """
        # Controlla se possiamo usare la cache
        current_time = time.time()
        if self._symbols_cache and (current_time - self._cache_timestamp < self._cache_ttl):
            print("Usando la cache dei simboli scambiabili")
            return self._symbols_cache
            
        try:
            # Carica i mercati se necessario
            self.exchange.load_markets()
            
            # Recupera i ticker dall'exchange
            tickers = self.exchange.fetch_tickers()
            
            # Prepara la lista dei simboli con il volume giornaliero
            symbols_with_volume = []
            
            for symbol, ticker in tickers.items():
                # Kraken ha principalmente coppie con USD invece di USDT
                if symbol.endswith('/USD') and 'active' in self.exchange.markets[symbol] and self.exchange.markets[symbol]['active']:
                    # Per Kraken, usiamo baseVolume * price come approssimazione del volume in USD
                    base_volume = ticker.get('baseVolume', 0)
                    price = ticker.get('last', 0)
                    
                    # Stima del volume in USD
                    usd_volume = base_volume * price if base_volume and price else 0
                    
                    # Controllo liquidità minima
                    if usd_volume > 50000:  # Più basso per Kraken che ha meno coppie di Binance
                        # Escludiamo simboli problematici e stablecoin vs stablecoin
                        if not (symbol.startswith('.') or 
                               ('USD/USD' in symbol) or 
                               ('DAI/USD' in symbol) or 
                               ('USDC/USD' in symbol) or
                               ('USDT/USD' in symbol)):
                            symbols_with_volume.append({
                                'symbol': symbol,
                                'volume': usd_volume,
                                'price': price
                            })
            
            # Se abbiamo troppo pochi simboli, includiamo anche quelli con volume più basso
            if len(symbols_with_volume) < 10:
                for symbol, ticker in tickers.items():
                    if symbol not in [s['symbol'] for s in symbols_with_volume] and symbol.endswith('/USD'):
                        base_volume = ticker.get('baseVolume', 0)
                        price = ticker.get('last', 0)
                        usd_volume = base_volume * price if base_volume and price else 0
                        
                        if usd_volume > 10000:  # Soglia più bassa per avere più simboli
                            symbols_with_volume.append({
                                'symbol': symbol,
                                'volume': usd_volume,
                                'price': price
                            })
                        
                        # Se abbiamo abbastanza simboli, interrompiamo
                        if len(symbols_with_volume) >= self.max_symbols:
                            break
            
            # Ordina per volume decrescente
            symbols_with_volume.sort(key=lambda x: x['volume'], reverse=True)
            
            # Prendiamo i primi max_symbols simboli
            result = symbols_with_volume[:self.max_symbols]
            
            # Memorizza nella cache
            self._symbols_cache = result
            self._cache_timestamp = current_time
            
            # Aggiunge simboli di default importanti anche se non presenti
            default_symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD']
            for def_sym in default_symbols:
                if not any(s['symbol'] == def_sym for s in result) and def_sym in self.exchange.markets:
                    # Aggiungi il simbolo predefinito con volume fittizio alto
                    price = self.exchange.fetch_ticker(def_sym).get('last', 0) if def_sym in self.exchange.markets else 0
                    result.append({
                        'symbol': def_sym,
                        'volume': 1000000,  # Alto per essere prioritario
                        'price': price
                    })
            
            return result
            
        except Exception as e:
            print(f"Errore nel recupero dei simboli scambiabili: {e}")
            
            # Fallback a lista default
            default_list = [
                {'symbol': 'BTC/USD', 'volume': 1000000000, 'price': 60000},
                {'symbol': 'ETH/USD', 'volume': 500000000, 'price': 3000},
                {'symbol': 'XRP/USD', 'volume': 300000000, 'price': 0.5},
                {'symbol': 'DOT/USD', 'volume': 200000000, 'price': 20},
                {'symbol': 'ADA/USD', 'volume': 200000000, 'price': 0.4},
                {'symbol': 'SOL/USD', 'volume': 150000000, 'price': 120},
                {'symbol': 'DOGE/USD', 'volume': 100000000, 'price': 0.1},
                {'symbol': 'LTC/USD', 'volume': 80000000, 'price': 80}
            ]
            return default_list
    
    def fetch_ohlcv_data(self, symbol, timeframe, limit=100):
        """
        Recupera i dati OHLCV per un simbolo usando il modulo crypto_data ottimizzato
        
        Parameters:
        symbol (str): Simbolo della criptovaluta
        timeframe (str): Timeframe dei dati
        limit (int): Numero di candele da recuperare
        
        Returns:
        pd.DataFrame: DataFrame con i dati OHLCV
        """
        try:
            import crypto_data
            # Usa il modulo ottimizzato con caching
            df = crypto_data.fetch_ohlcv_data(symbol, timeframe, limit=limit, exchange_id=self.exchange_id)
            
            if df is None:
                print(f"Nessun dato disponibile per {symbol} ({timeframe})")
                return pd.DataFrame()
                
            return df
        except Exception as e:
            print(f"Errore nel recupero dei dati OHLCV per {symbol} ({timeframe}): {e}")
            return pd.DataFrame()
    
    def analyze_symbol(self, symbol_data, timeframe):
        """
        Analizza un simbolo per identificare opportunità di trading
        
        Parameters:
        symbol_data (dict): Dizionario con informazioni sul simbolo
        timeframe (str): Timeframe da analizzare
        
        Returns:
        dict or None: Opportunità di trading se trovata, altrimenti None
        """
        symbol = symbol_data['symbol']
        
        try:
            # Recupera i dati OHLCV
            df = self.fetch_ohlcv_data(symbol, timeframe, limit=200)
            
            if df.empty:
                return None
            
            # Aggiungi indicatori tecnici
            df = technical_indicators.add_all_indicators(df)
            
            # Genera segnali
            df = signal_generator.generate_signals(df, timeframe=timeframe)
            
            # Valida i segnali
            df = signal_validator.validate_signals(df, quality_threshold=60)
            
            # Ottieni il segnale più recente
            recent_signals = []
            if not df.empty and 'signal' in df.columns:
                # Considera solo le ultime 3 barre per i segnali recenti
                look_back = 3
                
                # Per timeframe più lunghi, considera più barre
                if timeframe == '4h':
                    look_back = 6
                elif timeframe == '1d':
                    look_back = 10
                
                for i in range(min(look_back, len(df))):
                    idx = -i-1  # Indice dalla fine
                    if df['signal'].iloc[idx] in ['buy', 'sell']:
                        signal_type = df['signal'].iloc[idx]
                        
                        # Calcola forza del segnale (0-100)
                        signal_strength = df['signal_strength'].iloc[idx] if 'signal_strength' in df.columns else 50
                        
                        # Crea dizionario con i dettagli del segnale
                        signal_data = {
                            'symbol': symbol,
                            'price': df['close'].iloc[idx],
                            'signal_type': signal_type,
                            'timestamp': df.index[idx],
                            'timeframe': timeframe,
                            'signal_strength': signal_strength,
                            'entry_point': df['entry_point'].iloc[idx] if 'entry_point' in df.columns else df['close'].iloc[idx],
                            'stop_loss': df['stop_loss'].iloc[idx] if 'stop_loss' in df.columns else None,
                            'take_profit': df['take_profit'].iloc[idx] if 'take_profit' in df.columns else None,
                            'volume': symbol_data['volume'],
                            'bars_ago': i
                        }
                        
                        # Calcola metrics aggiuntive
                        if signal_type == 'buy':
                            signal_data['risk_reward_ratio'] = (signal_data['take_profit'] - signal_data['entry_point']) / (signal_data['entry_point'] - signal_data['stop_loss']) if signal_data['stop_loss'] and signal_data['take_profit'] else None
                        else:  # 'sell'
                            signal_data['risk_reward_ratio'] = (signal_data['entry_point'] - signal_data['take_profit']) / (signal_data['stop_loss'] - signal_data['entry_point']) if signal_data['stop_loss'] and signal_data['take_profit'] else None
                        
                        recent_signals.append(signal_data)
                
            return recent_signals
        
        except Exception as e:
            print(f"Errore nell'analisi di {symbol} ({timeframe}): {e}")
            return None
    
    def analyze_market_parallel(self, timeframe=None):
        """
        Analizza il mercato in parallelo per identificare opportunità di trading
        
        Parameters:
        timeframe (str, optional): Timeframe da analizzare. Se None, analizza tutti i timeframe configurati.
        
        Returns:
        list: Lista di opportunità di trading
        """
        # Recupera simboli scambiabili
        tradable_symbols = self.get_tradable_symbols()
        
        if not tradable_symbols:
            print("Nessun simbolo scambiabile trovato")
            return []
        
        # Determina i timeframe da analizzare
        timeframes_to_analyze = [timeframe] if timeframe else self.timeframes
        
        all_opportunities = []
        
        for tf in timeframes_to_analyze:
            opportunities = []
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Avvia analisi parallela per ogni simbolo
                future_to_symbol = {
                    executor.submit(self.analyze_symbol, symbol_data, tf): symbol_data
                    for symbol_data in tradable_symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol_data = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result:
                            opportunities.extend(result)
                    except Exception as e:
                        print(f"Eccezione nell'analizzare {symbol_data['symbol']}: {e}")
            
            # Aggiungi le opportunità trovate
            all_opportunities.extend(opportunities)
        
        return all_opportunities
    
    def rank_opportunities(self, opportunities):
        """
        Classifica le opportunità di trading in base a vari criteri
        
        Parameters:
        opportunities (list): Lista di opportunità di trading
        
        Returns:
        list: Lista di opportunità classificate
        """
        if not opportunities:
            return []
        
        # Aggiungi punteggio di sentiment di mercato
        try:
            market_sent = market_sentiment.get_market_sentiment()
            sentiment_score = market_sent.get('score', 50) if market_sent else 50
            sentiment_direction = market_sent.get('direction', 'Neutral') if market_sent else 'Neutral'
            
            # Calcola il punteggio di sentiment per ogni opportunità
            for opp in opportunities:
                # Aumenta il punteggio per segnali allineati con il sentiment
                if (sentiment_score > 60 and opp['signal_type'] == 'buy') or \
                   (sentiment_score < 40 and opp['signal_type'] == 'sell'):
                    opp['sentiment_alignment'] = 1.2  # Bonus 20%
                # Penalizza segnali contrastanti con il sentiment
                elif (sentiment_score < 40 and opp['signal_type'] == 'buy') or \
                     (sentiment_score > 60 and opp['signal_type'] == 'sell'):
                    opp['sentiment_alignment'] = 0.8  # Penalità 20%
                else:
                    opp['sentiment_alignment'] = 1.0  # Neutrale
                
                # Aggiungi il punteggio di sentiment
                opp['market_sentiment'] = sentiment_score
                opp['market_direction'] = sentiment_direction
                
        except Exception as e:
            print(f"Errore nell'ottenere il sentiment di mercato: {e}")
            # Valori default se il sentiment non è disponibile
            for opp in opportunities:
                opp['sentiment_alignment'] = 1.0
                opp['market_sentiment'] = 50
                opp['market_direction'] = 'Neutral'
        
        # Calcola il punteggio complessivo per ogni opportunità
        for opp in opportunities:
            # Forza del segnale (0-100)
            signal_score = opp.get('signal_strength', 50)
            
            # Freschezza del segnale (più recente = migliore)
            recency_score = 100 - (opp.get('bars_ago', 0) * 20)  # Decrementa 20 punti per ogni barra di distanza
            recency_score = max(0, recency_score)  # Non scendere sotto 0
            
            # Punteggio di volume (più alto = migliore, ma con scala logaritmica)
            volume_score = min(100, 10 * np.log10(opp.get('volume', 10000) / 10000))
            
            # Punteggio di risk/reward (più alto = migliore)
            rr_ratio = opp.get('risk_reward_ratio', 1)
            rr_score = min(100, rr_ratio * 50) if rr_ratio else 50
            
            # Timeframe (maggiore peso ai timeframe più lunghi per segnali più affidabili)
            timeframe_weights = {'15m': 0.8, '1h': 1.0, '4h': 1.2, '1d': 1.4}
            timeframe_weight = timeframe_weights.get(opp['timeframe'], 1.0)
            
            # Calcola punteggio finale
            final_score = (
                signal_score * 0.4 +  # 40% forza segnale
                recency_score * 0.2 +  # 20% freschezza
                volume_score * 0.1 +   # 10% volume
                rr_score * 0.3         # 30% rischio/rendimento
            ) * timeframe_weight * opp['sentiment_alignment']
            
            opp['score'] = final_score
        
        # Ordina per punteggio (decrescente)
        ranked_opportunities = sorted(opportunities, key=lambda x: x['score'], reverse=True)
        
        return ranked_opportunities
    
    def get_opportunities(self, timeframe=None, force_refresh=False, max_results=20):
        """
        Ottiene opportunità di trading
        
        Parameters:
        timeframe (str, optional): Timeframe da analizzare
        force_refresh (bool): Se True, forza un'analisi fresca
        max_results (int): Numero massimo di risultati da restituire
        
        Returns:
        list: Lista di opportunità di trading
        """
        current_time = time.time()
        
        # Controlla se è necessario aggiornare l'analisi
        if (force_refresh or 
            not self.last_analysis_time or 
            current_time - self.last_analysis_time > self.analysis_valid_period):
            
            # Esegui analisi di mercato
            opportunities = self.analyze_market_parallel(timeframe)
            
            # Classifica opportunità
            self.opportunities = self.rank_opportunities(opportunities)
            
            # Aggiorna timestamp
            self.last_analysis_time = current_time
        
        # Filtra opportunità per timeframe, se specificato
        if timeframe and self.opportunities:
            filtered_opportunities = [opp for opp in self.opportunities if opp['timeframe'] == timeframe]
        else:
            filtered_opportunities = self.opportunities
        
        # Limita il numero di risultati
        return filtered_opportunities[:max_results]
    
    def get_best_opportunities(self, buy_count=5, sell_count=5):
        """
        Ottiene le migliori opportunità di trading divise per tipo (buy/sell)
        
        Parameters:
        buy_count (int): Numero di opportunità di acquisto da restituire
        sell_count (int): Numero di opportunità di vendita da restituire
        
        Returns:
        dict: Dizionario con le migliori opportunità di acquisto e vendita
        """
        # Ottieni tutte le opportunità
        all_opportunities = self.get_opportunities(max_results=100)
        
        # Divide per tipo di segnale
        buy_opportunities = [opp for opp in all_opportunities if opp['signal_type'] == 'buy']
        sell_opportunities = [opp for opp in all_opportunities if opp['signal_type'] == 'sell']
        
        # Prendi i migliori N per ogni tipo
        top_buys = buy_opportunities[:buy_count]
        top_sells = sell_opportunities[:sell_count]
        
        return {
            'buy': top_buys,
            'sell': top_sells,
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_sentiment': self.opportunities[0]['market_sentiment'] if self.opportunities else 50,
            'market_direction': self.opportunities[0]['market_direction'] if self.opportunities else 'Neutral'
        }


# Funzioni di utilità per l'accesso dall'app
def get_market_opportunities(buy_count=5, sell_count=5, force_refresh=False):
    """
    Funzione di utilità per ottenere le migliori opportunità di trading
    
    Parameters:
    buy_count (int): Numero di opportunità di acquisto
    sell_count (int): Numero di opportunità di vendita
    force_refresh (bool): Se True, forza un'analisi fresca
    
    Returns:
    dict: Dizionario con le migliori opportunità
    """
    analyzer = MarketAnalyzer()
    return analyzer.get_best_opportunities(buy_count, sell_count)


# Per test e debug
if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    opportunities = analyzer.get_opportunities(timeframe='1h')
    
    print(f"Trovate {len(opportunities)} opportunità di trading")
    
    top_opportunities = analyzer.get_best_opportunities(3, 3)
    
    print("\nMigliori opportunità di acquisto:")
    for opp in top_opportunities['buy']:
        print(f"{opp['symbol']} - Punteggio: {opp['score']:.1f} - Prezzo: {opp['price']}")
        print(f"  Entry: {opp['entry_point']:.4f} - Stop: {opp['stop_loss']:.4f} - Target: {opp['take_profit']:.4f}")
    
    print("\nMigliori opportunità di vendita:")
    for opp in top_opportunities['sell']:
        print(f"{opp['symbol']} - Punteggio: {opp['score']:.1f} - Prezzo: {opp['price']}")
        print(f"  Entry: {opp['entry_point']:.4f} - Stop: {opp['stop_loss']:.4f} - Target: {opp['take_profit']:.4f}")