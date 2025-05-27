import ccxt
import pandas as pd
import streamlit as st
from datetime import datetime
import time

# Initialize exchange
def get_exchange():
    """
    Initialize and return the CCXT exchange object
    """
    try:
        # Use Kraken instead of Binance to avoid geo-restrictions
        exchange = ccxt.kraken({
            'enableRateLimit': True,
        })
        return exchange
    except Exception as e:
        st.error(f"Error initializing exchange: {str(e)}")
        return None

# Fetch OHLCV data for a cryptocurrency
def fetch_ohlcv_data(symbol, timeframe, limit=100):
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for a cryptocurrency
    
    Parameters:
    symbol (str): Cryptocurrency symbol (e.g., 'BTC/USDT')
    timeframe (str): Timeframe for data (e.g., '1h', '4h', '1d')
    limit (int): Number of candles to fetch
    
    Returns:
    pd.DataFrame: DataFrame with OHLCV data
    """
    try:
        exchange = get_exchange()
        if not exchange:
            return None
        
        # Verifica se il timeframe è supportato
        if timeframe not in exchange.timeframes:
            st.error(f"Timeframe {timeframe} non supportato. Timeframe supportati: {', '.join(exchange.timeframes.keys())}")
            return None
            
        # Verifica se il simbolo esiste e se è attivo
        exchange.load_markets()
        if symbol not in exchange.markets:
            st.error(f"Simbolo {symbol} non trovato.")
            return None
            
        if not exchange.markets[symbol]['active']:
            st.error(f"Simbolo {symbol} non attivo.")
            return None
            
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv or len(ohlcv) == 0:
            st.error(f"Nessun dato disponibile per {symbol} nel timeframe {timeframe}.")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Ensure all numeric columns are float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    except ccxt.NetworkError as e:
        st.error(f"Errore di rete: {str(e)}")
        return None
    except ccxt.ExchangeError as e:
        # Gestione specifica per errori comuni
        error_str = str(e)
        if "Invalid arguments" in error_str:
            st.error(f"Errore API per {symbol}: parametri non validi o coppia non supportata per questo timeframe.")
        else:
            st.error(f"Errore dell'exchange: {error_str}")
        return None
    except ccxt.BaseError as e:
        st.error(f"Errore CCXT per {symbol}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Errore durante il recupero dei dati per {symbol}: {str(e)}")
        return None

# Get current price for a cryptocurrency
def get_current_price(symbol):
    """
    Get the current price for a cryptocurrency
    
    Parameters:
    symbol (str): Cryptocurrency symbol (e.g., 'BTC/USDT')
    
    Returns:
    float: Current price
    """
    try:
        exchange = get_exchange()
        if not exchange:
            return None
        
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    
    except Exception as e:
        st.error(f"Error getting current price for {symbol}: {str(e)}")
        return None

# Get available markets
def get_available_markets():
    """
    Get a list of available cryptocurrency markets
    
    Returns:
    list: List of available markets
    """
    try:
        exchange = get_exchange()
        if not exchange:
            return []
        
        markets = exchange.load_markets()
        
        # Filtra solo le criptovalute attive con coppia USD
        active_markets = [
            market for market in markets.keys() 
            if '/USD' in market and markets[market]['active']
        ]
        
        return sorted(active_markets)
    
    except Exception as e:
        st.error(f"Errore nel recupero dei mercati disponibili: {str(e)}")
        return []
