import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import crypto_data
import technical_indicators
import signal_generator
import visualization
import performance_tracker
import signal_validator
import utils
import market_sentiment
import market_analyzer
import advanced_trading_algorithms

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Signals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Cryptocurrency Trading Signals")
st.markdown("""
This dashboard provides advanced trading signals for cryptocurrencies based on technical indicators.
The signals include optimized entry points, precise stop-loss, and take-profit recommendations.

### 🆕 New Feature: Real-time Market Sentiment Analysis
Our trading signals are now enhanced with real-time market sentiment data. Look for the ⬆️ and ⬇️ indicators 
in the signal table showing how sentiment is affecting each signal.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Get all available cryptocurrencies from Kraken
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_all_cryptos():
    # Predefined list of special tokens that may not be returned by the API
    special_tokens = [
        'VIRTUAL/USD', 'TAO/USD', 'TRUMP/USD', 'PEPE/USD', 'SUI/USD', 'SHIB/USD',
        'BONK/USD', 'AUCTION/USD', 'DYDX/USD', 'PYTH/USD', 'NEAR/USD', 'OP/USD'
    ]
    
    try:
        # Get available markets from the API
        markets = crypto_data.get_available_markets()
        
        # Add special tokens that aren't in the standard list
        for token in special_tokens:
            if token not in markets:
                markets.append(token)
        
        # Sort the list
        return sorted(markets)
    except Exception as e:
        st.error(f"Error fetching available cryptocurrencies: {str(e)}")
        # Fallback to a predefined list in case of error
        return [
            'BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'ADA/USD',
            'LINK/USD', 'DOT/USD', 'AVAX/USD', 'DOGE/USD', 'MATIC/USD',
            'FET/USD', 'TAO/USD', 'VIRTUAL/USD', 'TRUMP/USD', 'PEPE/USD',
            'SUI/USD', 'SHIB/USD', 'BONK/USD', 'APT/USD', 'ARB/USD'
        ]

# Cryptocurrency selection - with search function
available_cryptos = get_all_cryptos()

# Aggiungiamo token speciali all'inizio della lista che DEVONO essere disponibili
special_tokens = [
    'VIRTUAL/USDT', 'TAO/USDT', 'TRUMP/USDT', 'PEPE/USDT', 'MEME/USDT', 'SUI/USDT', 'BONK/USDT',
    'AI/USDT', 'SPACE/USDT', 'VR/USDT', 'GME/USDT', 'AMC/USDT', 'DOGE/USDT', 'SHIB/USDT',
    'META/USDT', 'APPLE/USDT', 'GOOGLE/USDT', 'AMAZON/USDT', 'TESLA/USDT'
]
for special in special_tokens:
    if special not in available_cryptos:
        available_cryptos.insert(0, special)

# Add a search feature
search_query = st.sidebar.text_input("Cerca criptovaluta", "")
filtered_cryptos = available_cryptos
if search_query:
    search_query = search_query.upper()
    filtered_cryptos = [crypto for crypto in available_cryptos if search_query in crypto.upper()]

# Show all available cryptocurrencies in a scrollable interface
st.sidebar.markdown("### Tutte le criptovalute disponibili:")

# Display in a more compact way
selected_cryptos = st.sidebar.multiselect(
    "Seleziona Criptovalute",
    options=filtered_cryptos,
    default=[] if search_query else [],  # No default selection
    format_func=lambda x: x.replace("/USDT", "").replace("/USD", "")  # Show more compact names
)

# Timeframe selection
timeframe_options = {
    '1m': '1 minuto',
    '5m': '5 minuti',
    '15m': '15 minuti',
    '30m': '30 minuti',
    '1h': '1 ora',
    '4h': '4 ore',
    '1d': '1 giorno',
    '1w': '1 settimana',
    '2w': '2 settimane',
    '1M': '1 mese'
}
selected_timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    options=list(timeframe_options.keys()),
    format_func=lambda x: timeframe_options[x],
    index=2  # Default to 15m
)

# Analysis mode
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Standard", "Conservative", "Aggressive", "Custom"],
    index=0
)

# Technical indicators parameters
st.sidebar.header("Indicator Parameters")

# Only show custom parameters if in custom mode
if analysis_mode == "Custom":
    rsi_period = st.sidebar.slider("RSI Period", 7, 25, 14)
    rsi_overbought = st.sidebar.slider("RSI Overbought Level", 65, 85, 70)
    rsi_oversold = st.sidebar.slider("RSI Oversold Level", 15, 35, 30)
    
    macd_fast = st.sidebar.slider("MACD Fast Period", 8, 24, 12)
    macd_slow = st.sidebar.slider("MACD Slow Period", 21, 52, 26)
    macd_signal = st.sidebar.slider("MACD Signal Period", 5, 15, 9)
    
    ma_fast_period = st.sidebar.slider("Fast MA Period", 5, 50, 20)
    ma_slow_period = st.sidebar.slider("Slow MA Period", 50, 200, 50)
    
    bollinger_period = st.sidebar.slider("Bollinger Bands Period", 10, 30, 20)
    bollinger_std = st.sidebar.slider("Bollinger Bands Standard Deviation", 1.5, 3.0, 2.0, 0.1)
    
    signal_quality_threshold = st.sidebar.slider("Signal Quality Threshold", 40, 90, 60)
    min_confirmation_indicators = st.sidebar.slider("Minimum Confirmation Indicators", 1, 5, 2)
    
else:
    # Preset parameters based on mode
    indicator_params = utils.get_indicator_params(analysis_mode, selected_timeframe)
    
    rsi_period = indicator_params["rsi_period"]
    rsi_overbought = indicator_params["rsi_overbought"]
    rsi_oversold = indicator_params["rsi_oversold"]
    
    macd_fast = indicator_params["macd_fast"]
    macd_slow = indicator_params["macd_slow"]
    macd_signal = indicator_params["macd_signal"]
    
    ma_fast_period = indicator_params["ma_fast_period"]
    ma_slow_period = indicator_params["ma_slow_period"]
    
    bollinger_period = indicator_params["bollinger_period"]
    bollinger_std = indicator_params["bollinger_std"]
    
    signal_quality_threshold = indicator_params["signal_quality_threshold"]
    min_confirmation_indicators = indicator_params["min_confirmation_indicators"]

# Risk management settings
st.sidebar.header("Risk Management")
stop_loss_percent = st.sidebar.slider("Stop Loss %", 1.0, 10.0, 3.0, 0.5)
take_profit_percent = st.sidebar.slider("Take Profit %", 2.0, 15.0, 6.0, 0.5)
risk_reward_ratio = st.sidebar.text(f"Risk/Reward Ratio: 1:{take_profit_percent/stop_loss_percent:.2f}")

# Advanced options expander
with st.sidebar.expander("Advanced Options"):
    backtest_period = st.slider("Backtest Period (days)", 7, 90, 30)
    show_signals_on_chart = st.checkbox("Show Signals on Chart", value=True)
    show_indicators_on_chart = st.checkbox("Show Indicators on Chart", value=True)
    show_historical_signals = st.checkbox("Show Historical Signals", value=True)
    auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)

# Function to load and process data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(crypto, timeframe, limit=100):
    try:
        df = crypto_data.fetch_ohlcv_data(crypto, timeframe, limit)
        if df is not None and not df.empty:
            # Optimize parameters based on timeframe and market conditions
            params = utils.optimize_parameters_for_market(
                df, 
                timeframe, 
                rsi_period, 
                rsi_overbought, 
                rsi_oversold,
                macd_fast, 
                macd_slow, 
                macd_signal,
                ma_fast_period, 
                ma_slow_period,
                bollinger_period,
                bollinger_std
            )
            
            # Calculate indicators with optimized parameters
            df = technical_indicators.add_all_indicators(
                df, 
                rsi_period=params["rsi_period"], 
                macd_fast=params["macd_fast"], 
                macd_slow=params["macd_slow"], 
                macd_signal=params["macd_signal"],
                ma_fast_period=params["ma_fast_period"],
                ma_slow_period=params["ma_slow_period"],
                bollinger_period=params["bollinger_period"],
                bollinger_std=params["bollinger_std"]
            )
            
            # Generate signals with optimized parameters
            df = signal_generator.generate_signals(
                df, 
                rsi_overbought=params["rsi_overbought"], 
                rsi_oversold=params["rsi_oversold"],
                timeframe=timeframe
            )
            
            # Validate signals for quality
            df = signal_validator.validate_signals(
                df, 
                quality_threshold=signal_quality_threshold,
                min_confirmation=min_confirmation_indicators
            )
            
            # Calculate optimized stops and targets
            df = utils.calculate_advanced_risk_levels(
                df, 
                stop_loss_percent=stop_loss_percent, 
                take_profit_percent=take_profit_percent
            )
            
            return df
        else:
            st.error(f"Failed to fetch data for {crypto}")
            return None
    except Exception as e:
        st.error(f"Error loading data for {crypto}: {str(e)}")
        return None

# Main content
if selected_cryptos:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Current Signals", 
        "Advanced Algorithms",
        "Market Scanner",
        "Market Sentiment",
        "Charts", 
        "Signal Performance", 
        "Performance Dashboard",
        "Trading Journal"
    ])
    
    with tab1:
        st.header("Current Trading Signals")
        
        # Signal quality filter
        signal_filter = st.select_slider(
            "Signal Quality Filter",
            options=["All Signals", "High Quality Only", "Highest Quality Only"],
            value="All Signals"  # Default to All Signals per vedere tutti i segnali
        )
        
        all_signals = []
        
        # Get market sentiment to integrate with signal generation
        try:
            market_sent = market_sentiment.get_market_sentiment()
            sentiment_score = market_sent.get('score', 50) if market_sent else 50
            market_direction = market_sent.get('direction', 'Neutral') if market_sent else 'Neutral'
        except Exception as e:
            sentiment_score = 50
            market_direction = 'Neutral'
            st.warning(f"Market sentiment data not available. Using neutral sentiment for signal generation.")
        
        # Determine sentiment bias for signal generation
        sentiment_bias = 'neutral'
        if sentiment_score >= 65:
            sentiment_bias = 'bullish'
        elif sentiment_score <= 35:
            sentiment_bias = 'bearish'
        
        for crypto in selected_cryptos:
            df = load_data(crypto, selected_timeframe)
            if df is not None and not df.empty:
                # Generate signals based on indicators with market sentiment adjustment
                with_signals = signal_generator.generate_signals(df, 
                                                               rsi_overbought=rsi_overbought, 
                                                               rsi_oversold=rsi_oversold,
                                                               timeframe=selected_timeframe)
                
                # Get the most recent signals with timeframe for adaptation
                recent_signals = signal_generator.get_recent_signals(with_signals, crypto, timeframe=selected_timeframe)
                
                # Apply sentiment bias to adjust signal strength
                for signal in recent_signals:
                    # Enhance buy signals in bullish market
                    if sentiment_bias == 'bullish' and signal['signal_type'] == 'buy':
                        signal['signal_strength'] = min(100, signal['signal_strength'] * 1.15)
                        signal['sentiment_boost'] = True
                    
                    # Enhance sell signals in bearish market
                    elif sentiment_bias == 'bearish' and signal['signal_type'] == 'sell':
                        signal['signal_strength'] = min(100, signal['signal_strength'] * 1.15)
                        signal['sentiment_boost'] = True
                    
                    # Reduce conflicting signals
                    elif (sentiment_bias == 'bullish' and signal['signal_type'] == 'sell') or \
                         (sentiment_bias == 'bearish' and signal['signal_type'] == 'buy'):
                        signal['signal_strength'] = max(0, signal['signal_strength'] * 0.85)
                        signal['sentiment_penalty'] = True
                    
                    # Add sentiment information to signal
                    signal['market_sentiment'] = market_direction
                    signal['sentiment_score'] = sentiment_score
                
                # Apply quality filter
                if signal_filter == "High Quality Only":
                    recent_signals = [s for s in recent_signals if s.get('signal_strength', 0) >= 40]
                elif signal_filter == "Highest Quality Only":
                    recent_signals = [s for s in recent_signals if s.get('signal_strength', 0) >= 60]
                
                if recent_signals:
                    all_signals.extend(recent_signals)
        
        if all_signals:
            # Create a DataFrame for the signals
            signals_df = pd.DataFrame(all_signals)
            
            # Add validation status and calculated risk metrics
            signals_df = signal_validator.enhance_signal_dataframe(signals_df)
            
            # Mostro i segnali in formato tabella
            st.subheader("📊 Tabella Segnali di Trading")
            
            # Definisco quali colonne visualizzare nella tabella
            # Verifichiamo quali colonne sono effettivamente disponibili
            available_columns = signals_df.columns.tolist()
            
            # Colonne base che dovrebbero sempre essere presenti
            base_columns = ['symbol', 'signal_type', 'price']
            
            # Colonne opzionali che potrebbero esserci
            optional_columns = {
                'entry_point': 'Punto di Ingresso',
                'stop_loss': 'Stop Loss',
                'take_profit': 'Take Profit',
                'signal_quality': 'Qualità',
                'signal_strength': 'Forza Segnale',
                'timeframe': 'Timeframe'
            }
            
            # Inizializzo il mapping per la rinomina delle colonne
            rename_map = {
                'symbol': 'Simbolo',
                'signal_type': 'Tipo Segnale',
                'price': 'Prezzo Attuale'
            }
            
            # Aggiungo le colonne opzionali se presenti
            display_columns = base_columns.copy()
            for col, name in optional_columns.items():
                if col in available_columns:
                    display_columns.append(col)
                    rename_map[col] = name
            
            # Se mancano entry_point, stop_loss o take_profit, li calcoliamo
            if 'price' in available_columns:
                if 'entry_point' not in available_columns:
                    signals_df['entry_point'] = signals_df['price']
                    display_columns.append('entry_point')
                    rename_map['entry_point'] = 'Punto di Ingresso'
                
                if 'stop_loss' not in available_columns and 'entry_point' in available_columns:
                    # Calcola stop loss come 3% sotto o sopra il punto di ingresso
                    signals_df['stop_loss'] = signals_df.apply(
                        lambda row: row['entry_point'] * 0.97 if row['signal_type'] == 'buy' else row['entry_point'] * 1.03, 
                        axis=1
                    )
                    display_columns.append('stop_loss')
                    rename_map['stop_loss'] = 'Stop Loss'
                
                if 'take_profit' not in available_columns and 'entry_point' in available_columns:
                    # Calcola take profit come 6% sopra o sotto il punto di ingresso
                    signals_df['take_profit'] = signals_df.apply(
                        lambda row: row['entry_point'] * 1.06 if row['signal_type'] == 'buy' else row['entry_point'] * 0.94,
                        axis=1
                    )
                    display_columns.append('take_profit')
                    rename_map['take_profit'] = 'Take Profit'
                
                if 'signal_strength' not in available_columns and 'signal_quality' not in available_columns:
                    # Aggiungiamo un valore di forza di segnale standard
                    signals_df['signal_strength'] = 50
                    display_columns.append('signal_strength')
                    rename_map['signal_strength'] = 'Forza Segnale'
            
            # Formato la tabella per la visualizzazione
            display_df = signals_df[display_columns].copy()
            
            # Aggiungiamo l'informazione sul sentiment alla tabella se disponibile
            if 'sentiment_boost' in signals_df.columns or 'sentiment_penalty' in signals_df.columns:
                display_df['sentiment_impact'] = ''
                
                # Marca i segnali potenziati dal sentiment
                if 'sentiment_boost' in signals_df.columns:
                    display_df.loc[signals_df['sentiment_boost'] == True, 'sentiment_impact'] = '⬆️'
                
                # Marca i segnali penalizzati dal sentiment
                if 'sentiment_penalty' in signals_df.columns:
                    display_df.loc[signals_df['sentiment_penalty'] == True, 'sentiment_impact'] = '⬇️'
                
                # Aggiungi colonna per l'impatto del sentiment
                display_columns = list(display_columns) + ['sentiment_impact']
                rename_map['sentiment_impact'] = 'Sentiment'
            
            display_df = display_df.rename(columns=rename_map)
            
            # Aggiunge icone per i tipi di segnale
            display_df['Tipo Segnale'] = display_df['Tipo Segnale'].apply(
                lambda x: "🟢 BUY" if x == 'buy' else ("🔴 SELL" if x == 'sell' else "⚪ NEUTRAL")
            )
            
            # Mostro la tabella
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Creo tre colonne per visualizzare i diversi tipi di segnale in dettaglio
            with st.expander("Visualizza dettagli segnali", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                # Buy signals
                with col1:
                    st.subheader("💹 Buy Signals")
                    buy_signals = signals_df[signals_df['signal_type'] == 'buy']
                    if not buy_signals.empty:
                        for _, signal in buy_signals.iterrows():
                            with st.container(border=True):
                                quality_field = 'signal_quality' if 'signal_quality' in signal else 'signal_strength'
                                quality_value = signal[quality_field] if pd.notna(signal[quality_field]) else 0
                                
                                st.write(f"**{signal['symbol']}** - Quality: {quality_value:.0f}%")
                                col1a, col2a = st.columns(2)
                                with col1a:
                                    price_change = signal.get('price_change', 0)
                                    st.metric("Entry", f"{signal['entry_point']:.4f}", f"{price_change:.2f}%")
                                with col2a:
                                    profit_potential = signal.get('profit_potential', ((signal['take_profit']/signal['entry_point'])-1)*100)
                                    st.metric("Take Profit", f"{signal['take_profit']:.4f}", f"{profit_potential:.2f}%")
                                st.progress(quality_value/100, text=f"Stop: {signal['stop_loss']:.4f}")
                    else:
                        st.info("No buy signals at the moment.")
                
                # Sell signals
                with col2:
                    st.subheader("📉 Sell Signals")
                    sell_signals = signals_df[signals_df['signal_type'] == 'sell']
                    if not sell_signals.empty:
                        for _, signal in sell_signals.iterrows():
                            with st.container(border=True):
                                quality_field = 'signal_quality' if 'signal_quality' in signal else 'signal_strength'
                                quality_value = signal[quality_field] if pd.notna(signal[quality_field]) else 0
                                
                                st.write(f"**{signal['symbol']}** - Quality: {quality_value:.0f}%")
                                col1a, col2a = st.columns(2)
                                with col1a:
                                    price_change = signal.get('price_change', 0)
                                    st.metric("Entry", f"{signal['entry_point']:.4f}", f"{price_change:.2f}%")
                                with col2a:
                                    profit_potential = signal.get('profit_potential', ((signal['entry_point']/signal['take_profit'])-1)*100)
                                    st.metric("Take Profit", f"{signal['take_profit']:.4f}", f"{profit_potential:.2f}%")
                                st.progress(quality_value/100, text=f"Stop: {signal['stop_loss']:.4f}")
                    else:
                        st.info("No sell signals at the moment.")
                
                # Neutral or hold
                with col3:
                    st.subheader("⏸️ Neutral/Hold")
                    neutral_signals = signals_df[signals_df['signal_type'] == 'neutral']
                    if not neutral_signals.empty:
                        st.dataframe(neutral_signals[['symbol', 'price', 'timestamp']], 
                                    use_container_width=True,
                                    hide_index=True)
                    else:
                        st.info("No neutral signals at the moment.")
                    
                # Show market sentiment overview
                st.subheader("Market Sentiment")
                if not signals_df.empty:
                    sentiment = utils.calculate_market_sentiment(signals_df)
                    st.write(f"Overall Market Direction: **{sentiment['overall_direction']}**")
                    st.progress(sentiment['sentiment_score']/100, 
                                text=f"Bullish: {sentiment['bullish_percent']:.0f}% | Bearish: {sentiment['bearish_percent']:.0f}%")
        else:
            st.info("No signals generated for the selected cryptocurrencies and timeframe.")
            
    with tab2:
        st.header("🧠 Algoritmi di Trading Avanzati")
        
        st.markdown("""
        Gli algoritmi di trading avanzati utilizzano strategie sofisticate di machine learning e analisi quantitativa per generare 
        segnali di trading ad alta precisione. Questi algoritmi superano le limitazioni delle API di exchange e possono identificare 
        opportunità che i metodi tradizionali potrebbero non rilevare.
        
        Ogni algoritmo è specializzato in una specifica condizione di mercato e utilizza una combinazione di indicatori tecnici, 
        pattern di prezzo e analisi di volatilità per generare segnali di trading ottimizzati.
        """)
        
        # Seleziona la criptovaluta da analizzare
        selected_crypto_advanced = st.selectbox(
            "Seleziona Criptovaluta",
            options=available_cryptos,
            index=0  # Default a BTC/USD
        )
        
        # Seleziona il timeframe
        selected_timeframe_advanced = st.selectbox(
            "Seleziona Timeframe",
            options=list(timeframe_options.keys()),
            format_func=lambda x: timeframe_options[x],
            index=4  # Default a 1h
        )
        
        # Seleziona gli algoritmi da utilizzare
        algo = advanced_trading_algorithms.AdvancedTradingAlgorithm()
        available_algorithms = algo.available_strategies
        
        # Crea una mappa con descrizioni per la visualizzazione
        algorithm_descriptions = algo.strategy_descriptions
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Selezione multipla degli algoritmi
            selected_algorithms = st.multiselect(
                "Seleziona Algoritmi di Trading",
                options=available_algorithms,
                default=['ml_enhanced_trend_following', 'volatility_breakout', 'mean_reversion'],
                format_func=lambda x: algorithm_descriptions.get(x, x)
            )
        
        with col2:
            # Filtro per qualità minima del segnale
            min_signal_strength = st.slider(
                "Qualità Minima Segnale",
                min_value=0,
                max_value=100,
                value=60,
                step=5
            )
            
            # Opzione per forzare analisi fresca
            force_refresh = st.checkbox("Forza Ricalcolo", value=False)
        
        # Pulsante per avviare l'analisi
        if st.button("Analizza con Algoritmi Avanzati", type="primary"):
            with st.spinner("Analisi in corso con algoritmi avanzati..."):
                try:
                    # Usa advanced_trading_algorithms per analizzare la criptovaluta
                    signals = advanced_trading_algorithms.analyze_crypto_with_multiple_algorithms(
                        selected_crypto_advanced, 
                        selected_timeframe_advanced,
                        min_signal_strength=min_signal_strength
                    )
                    
                    if signals:
                        st.success(f"Trovati {len(signals)} segnali di trading per {selected_crypto_advanced}")
                        
                        # Crea DataFrame per visualizzazione
                        signals_df = pd.DataFrame([
                            {
                                'Tipo': signal['signal_type'],
                                'Prezzo': signal['price'],
                                'Ingresso': signal['entry_point'],
                                'Stop Loss': signal['stop_loss'],
                                'Take Profit': signal['take_profit'],
                                'Timeframe': signal['timeframe'],
                                'Strategia': signal['strategy'],
                                'Qualità': signal['signal_strength'],
                                'Risk/Reward': signal.get('risk_reward_ratio', 0)
                            }
                            for signal in signals
                        ])
                        
                        # Visualizza tabella dei segnali
                        st.subheader("Tabella Segnali")
                        
                        # Aggiungi icone per i tipi di segnale
                        signals_df['Tipo'] = signals_df['Tipo'].apply(
                            lambda x: "🟢 BUY" if x == 'buy' else ("🔴 SELL" if x == 'sell' else "⚪ NEUTRAL")
                        )
                        
                        # Mostra la tabella
                        st.dataframe(signals_df, use_container_width=True, hide_index=True)
                        
                        # Visualizzazione dettagliata dei segnali
                        st.subheader("Dettagli Segnali")
                        
                        for i, signal in enumerate(signals):
                            with st.expander(f"{signal['signal_type'].upper()} - {signal['strategy']} - Qualità: {signal['signal_strength']:.0f}%", expanded=i==0):
                                # Ottieni spiegazione dettagliata del segnale
                                explanation = advanced_trading_algorithms.get_signal_explanation(signal)
                                
                                # Layout in colonne
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # Mostra spiegazione
                                    st.markdown(explanation)
                                
                                with col2:
                                    # Mostra metriche chiave
                                    st.metric("Prezzo Attuale", f"{signal['price']:.4f}")
                                    
                                    # Calcola potenziale profitto/perdita
                                    if signal['signal_type'] == 'buy':
                                        profit_potential = ((signal['take_profit']/signal['entry_point'])-1)*100
                                        loss_potential = ((signal['stop_loss']/signal['entry_point'])-1)*100
                                    else:  # 'sell'
                                        profit_potential = ((signal['entry_point']/signal['take_profit'])-1)*100
                                        loss_potential = ((signal['entry_point']/signal['stop_loss'])-1)*100
                                    
                                    col2a, col2b = st.columns(2)
                                    with col2a:
                                        st.metric("Potenziale Profitto", f"{abs(profit_potential):.2f}%")
                                    with col2b:
                                        st.metric("Potenziale Perdita", f"{abs(loss_potential):.2f}%")
                                    
                                    # Mostra barra di qualità
                                    st.progress(signal['signal_strength']/100, text=f"Forza: {signal['signal_strength']:.0f}/100")
                    else:
                        st.info(f"Nessun segnale di trading trovato per {selected_crypto_advanced} sul timeframe {selected_timeframe_advanced} che soddisfi i criteri di qualità minima.")
                        
                        # Suggerimenti per migliorare i risultati
                        st.markdown("""
                        ### Suggerimenti per trovare più segnali:
                        
                        1. Riduci la qualità minima del segnale
                        2. Prova timeframe diversi (4h o 1d per segnali più stabili)
                        3. Seleziona più algoritmi di trading
                        4. Prova con criptovalute più volatili
                        """)
                except Exception as e:
                    st.error(f"Errore durante l'analisi: {str(e)}")
                    
                    # Suggerimenti in caso di errore
                    st.warning("""
                    Gli algoritmi avanzati richiedono accesso ai dati di mercato. Se stai riscontrando errori, potrebbe essere 
                    dovuto a restrizioni API o problemi di connessione. Prova con un'altra criptovaluta o timeframe.
                    """)
        else:
            # Informazioni sugli algoritmi quando nessun'analisi è in corso
            st.subheader("Informazioni sugli Algoritmi Disponibili")
            
            # Crea tabella informativa sugli algoritmi
            algorithms_info = pd.DataFrame([
                {
                    'Algoritmo': algo_name,
                    'Descrizione': algorithm_descriptions[algo_name],
                    'Ideale Per': 'Trend forti' if 'trend' in algo_name.lower() else
                              'Breakout' if 'breakout' in algo_name.lower() else
                              'Mercati laterali' if 'mean_reversion' in algo_name.lower() or 'channel' in algo_name.lower() else
                              'Analisi candele' if 'pattern' in algo_name.lower() else
                              'Analisi multi-timeframe' if 'confluence' in algo_name.lower() else 'Varie condizioni'
                }
                for algo_name in available_algorithms
            ])
            
            st.dataframe(algorithms_info, use_container_width=True, hide_index=True)
            
            # Esempio di analisi combinata
            st.subheader("Potenza degli Algoritmi Combinati")
            st.markdown("""
            La vera potenza di questo sistema risiede nella combinazione di più algoritmi specializzati.
            Ogni algoritmo è ottimizzato per specifiche condizioni di mercato, e quando più algoritmi
            generano segnali concordanti, la probabilità di successo aumenta significativamente.
            
            ### Come vengono calcolati i livelli di ingresso, stop loss e take profit?
            
            A differenza dei metodi tradizionali basati su percentuali fisse, gli algoritmi avanzati calcolano
            livelli di rischio dinamici basati su:
            
            - **Volatilità attuale** (ATR)
            - **Livelli chiave** di supporto e resistenza
            - **Struttura del mercato** e pattern di prezzo
            - **Condizioni specifiche** di ogni strategia
            
            Questo approccio rende i segnali molto più precisi e adattati alle condizioni attuali del mercato.
            """)
        
    with tab3:
        st.header("🔍 Market Scanner - Trading Opportunities")
        
        st.markdown("""
        Lo scanner di mercato analizza automaticamente centinaia di criptovalute per identificare 
        le migliori opportunità di trading in questo momento. Il sistema utilizza un algoritmo 
        di ranking che considera molteplici fattori tra cui:
        
        - Forza dei segnali tecnici
        - Allineamento con il sentiment di mercato
        - Rapporto rischio/rendimento
        - Volume di trading e liquidità
        - Timeframe multipli
        """)
        
        # Configurazione dello scanner
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            buy_count = st.number_input("Numero di segnali di acquisto", min_value=1, max_value=20, value=5)
        with col2:
            sell_count = st.number_input("Numero di segnali di vendita", min_value=1, max_value=20, value=5)
        with col3:
            force_refresh = st.checkbox("Forza aggiornamento analisi", value=False)
        
        # Pulsante per avviare l'analisi
        if st.button("Scansiona il Mercato", type="primary"):
            with st.spinner("Analisi del mercato in corso... Questa operazione potrebbe richiedere alcuni minuti."):
                try:
                    # Ottieni le migliori opportunità di trading
                    market_opps = market_analyzer.get_market_opportunities(
                        buy_count=buy_count, 
                        sell_count=sell_count, 
                        force_refresh=force_refresh
                    )
                    
                    if market_opps:
                        # Mostra il sentiment di mercato corrente
                        sentiment_score = market_opps.get('market_sentiment', 50)
                        sentiment_direction = market_opps.get('market_direction', 'Neutral')
                        
                        st.info(f"Analisi completata il {market_opps.get('updated', 'N/A')}. " +
                               f"Sentiment di mercato attuale: **{sentiment_direction}** ({sentiment_score}/100)")
                        
                        # Mostra le migliori opportunità di acquisto
                        st.subheader("🟢 Migliori Opportunità di Acquisto")
                        buy_opportunities = market_opps.get('buy', [])
                        
                        if buy_opportunities:
                            # Crea DataFrame per visualizzazione
                            buy_df = pd.DataFrame([
                                {
                                    'Simbolo': opp['symbol'],
                                    'Prezzo Attuale': opp['price'],
                                    'Punto di Ingresso': opp['entry_point'],
                                    'Stop Loss': opp['stop_loss'],
                                    'Take Profit': opp['take_profit'],
                                    'Timeframe': opp['timeframe'],
                                    'Qualità': f"{opp['score']:.1f}",
                                    'Risk/Reward': f"{opp.get('risk_reward_ratio', 0):.2f}"
                                } 
                                for opp in buy_opportunities
                            ])
                            
                            # Display table
                            st.dataframe(buy_df, use_container_width=True, hide_index=True)
                            
                            # Visualizzazione dettagliata
                            with st.expander("Dettagli Opportunità di Acquisto", expanded=False):
                                for i, opp in enumerate(buy_opportunities):
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                    with col1:
                                        st.subheader(f"{opp['symbol']} - {opp['timeframe']}")
                                        profit_potential = ((opp['take_profit']/opp['entry_point'])-1)*100
                                        stop_risk = ((opp['entry_point']-opp['stop_loss'])/opp['entry_point'])*100
                                        
                                        st.write(f"**Prezzo Attuale:** {opp['price']:.4f}")
                                        st.write(f"**Punto di Ingresso:** {opp['entry_point']:.4f}")
                                        st.write(f"**Stop Loss:** {opp['stop_loss']:.4f} ({stop_risk:.1f}%)")
                                        st.write(f"**Take Profit:** {opp['take_profit']:.4f} (+{profit_potential:.1f}%)")
                                        
                                    with col2:
                                        # Metrics
                                        st.metric("Qualità", f"{opp['score']:.1f}/100")
                                        st.metric("Risk/Reward", f"1:{opp.get('risk_reward_ratio', 0):.2f}")
                                        
                                    with col3:
                                        # More metrics
                                        bars_ago = opp.get('bars_ago', 0)
                                        freshness = "Recente" if bars_ago <= 1 else f"{bars_ago} barre fa"
                                        st.metric("Freschezza", freshness)
                                        
                                        # Alignment with market sentiment
                                        sentiment_alignment = opp.get('sentiment_alignment', 1.0)
                                        alignment_text = "Alto" if sentiment_alignment > 1.0 else "Basso" if sentiment_alignment < 1.0 else "Neutro"
                                        st.metric("Allineamento Sentiment", alignment_text)
                                    
                                    st.divider()
                        else:
                            st.info("Nessuna opportunità di acquisto rilevante trovata al momento.")
                        
                        # Mostra le migliori opportunità di vendita
                        st.subheader("🔴 Migliori Opportunità di Vendita")
                        sell_opportunities = market_opps.get('sell', [])
                        
                        if sell_opportunities:
                            # Crea DataFrame per visualizzazione
                            sell_df = pd.DataFrame([
                                {
                                    'Simbolo': opp['symbol'],
                                    'Prezzo Attuale': opp['price'],
                                    'Punto di Ingresso': opp['entry_point'],
                                    'Stop Loss': opp['stop_loss'],
                                    'Take Profit': opp['take_profit'],
                                    'Timeframe': opp['timeframe'],
                                    'Qualità': f"{opp['score']:.1f}",
                                    'Risk/Reward': f"{opp.get('risk_reward_ratio', 0):.2f}"
                                } 
                                for opp in sell_opportunities
                            ])
                            
                            # Display table
                            st.dataframe(sell_df, use_container_width=True, hide_index=True)
                            
                            # Visualizzazione dettagliata
                            with st.expander("Dettagli Opportunità di Vendita", expanded=False):
                                for i, opp in enumerate(sell_opportunities):
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                    with col1:
                                        st.subheader(f"{opp['symbol']} - {opp['timeframe']}")
                                        profit_potential = ((opp['entry_point']-opp['take_profit'])/opp['entry_point'])*100
                                        stop_risk = ((opp['stop_loss']-opp['entry_point'])/opp['entry_point'])*100
                                        
                                        st.write(f"**Prezzo Attuale:** {opp['price']:.4f}")
                                        st.write(f"**Punto di Ingresso:** {opp['entry_point']:.4f}")
                                        st.write(f"**Stop Loss:** {opp['stop_loss']:.4f} (+{stop_risk:.1f}%)")
                                        st.write(f"**Take Profit:** {opp['take_profit']:.4f} (-{profit_potential:.1f}%)")
                                        
                                    with col2:
                                        # Metrics
                                        st.metric("Qualità", f"{opp['score']:.1f}/100")
                                        st.metric("Risk/Reward", f"1:{opp.get('risk_reward_ratio', 0):.2f}")
                                        
                                    with col3:
                                        # More metrics
                                        bars_ago = opp.get('bars_ago', 0)
                                        freshness = "Recente" if bars_ago <= 1 else f"{bars_ago} barre fa"
                                        st.metric("Freschezza", freshness)
                                        
                                        # Alignment with market sentiment
                                        sentiment_alignment = opp.get('sentiment_alignment', 1.0)
                                        alignment_text = "Alto" if sentiment_alignment > 1.0 else "Basso" if sentiment_alignment < 1.0 else "Neutro"
                                        st.metric("Allineamento Sentiment", alignment_text)
                                    
                                    st.divider()
                        else:
                            st.info("Nessuna opportunità di vendita rilevante trovata al momento.")
                        
                        # Suggerimenti per i trader
                        with st.expander("Suggerimenti per i Trader"):
                            st.markdown("""
                            ### Come utilizzare al meglio lo Scanner di Mercato
                            
                            1. **Analisi Multi-Timeframe:** Le opportunità contrassegnate con timeframe più lunghi (4h, 1d) 
                               tendono ad essere più affidabili rispetto a quelle sui timeframe più brevi.
                               
                            2. **Qualità del Segnale:** Priorizza i segnali con un punteggio di qualità superiore a 70 
                               per aumentare le probabilità di successo.
                               
                            3. **Rapporto Rischio/Rendimento:** Un buon trade dovrebbe avere un rapporto rischio/rendimento 
                               di almeno 1:2, il che significa che il potenziale guadagno è il doppio del rischio.
                               
                            4. **Allineamento col Sentiment:** I segnali che sono in linea con il sentiment di mercato generale 
                               hanno maggiori probabilità di successo.
                               
                            5. **Freschezza:** I segnali più recenti sono generalmente più rilevanti. Considera di dare 
                               priorità ai segnali generati nelle ultime barre.
                            
                            6. **Verifica Tecnica:** Prima di tradare, esamina sempre il grafico e gli indicatori nella 
                               scheda "Charts" per confermare personalmente l'opportunità.
                            """)
                    else:
                        st.error("Non è stato possibile recuperare le opportunità di trading. Riprova più tardi.")
                
                except Exception as e:
                    st.error(f"Errore durante l'analisi del mercato: {str(e)}")
                    st.warning("L'analisi automatica del mercato richiede l'accesso ai dati in tempo reale. "
                              "Se vedi questo errore, potrebbe essere dovuto a limiti di API o connessione.")
        else:
            # Mostra un prompt iniziale
            st.info("Clicca 'Scansiona il Mercato' per analizzare tutte le criptovalute disponibili e "
                   "identificare le migliori opportunità di trading in questo momento.")
            
            # Suggerimenti sul funzionamento dello scanner
            st.markdown("""
            ### Come funziona lo scanner di mercato
            
            1. Analizza automaticamente centinaia di criptovalute attraverso più timeframe (15m, 1h, 4h, 1d)
            2. Calcola segnali di trading utilizzando indicatori tecnici avanzati
            3. Integra l'analisi del sentiment di mercato per migliorare l'efficacia
            4. Classifica le opportunità in base a molteplici fattori di qualità
            5. Suggerisce i livelli ottimali di ingresso, stop loss e take profit
            
            L'analisi può richiedere alcuni minuti, in quanto vengono elaborati migliaia di dati in tempo reale.
            """)
            
    with tab3:
        st.header("🌍 Market Sentiment Analysis")
        
        # Call market sentiment analyzer
        try:
            with st.spinner("Analyzing market sentiment..."):
                sentiment_data = market_sentiment.get_market_sentiment()
            
            if sentiment_data:
                # Create columns for the main sentiment indicators
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Create a sentiment score gauge
                    sentiment_score = sentiment_data.get('score', 50)
                    sentiment_color = "green" if sentiment_score > 60 else "red" if sentiment_score < 40 else "orange"
                    
                    st.metric(
                        "Market Sentiment Score", 
                        f"{sentiment_score}/100",
                        delta=None,
                    )
                    
                    # Progress bar as gauge
                    st.progress(sentiment_score/100, 
                               text=f"Sentiment Score: {sentiment_score}/100")
                
                with col2:
                    # Display sentiment direction
                    direction = sentiment_data.get('direction', 'Neutral')
                    direction_icon = "🐂" if "Bullish" in direction else "🐻" if "Bearish" in direction else "⚖️"
                    
                    st.metric(
                        "Market Direction", 
                        f"{direction_icon} {direction}",
                        delta=None,
                    )
                    
                    # Display sentiment breakdown
                    st.write(f"Bullish: **{sentiment_data.get('bullish', 0)}%**")
                    st.write(f"Bearish: **{sentiment_data.get('bearish', 0)}%**")
                    st.write(f"Neutral: **{sentiment_data.get('neutral', 0)}%**")
                
                with col3:
                    # Display sentiment strength and update time
                    strength = sentiment_data.get('strength', 'Weak')
                    updated_time = sentiment_data.get('updated', 'N/A')
                    
                    st.metric(
                        "Sentiment Strength", 
                        strength,
                        delta=None,
                    )
                    
                    st.info(f"Last updated: {updated_time}")
                
                # Create columns for gainers and losers
                st.subheader("Market Movers")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### 🚀 Top Gainers")
                    
                    top_gainers = sentiment_data.get('top_gainers', [])
                    if top_gainers:
                        # Create a dataframe for better display
                        gainers_df = pd.DataFrame(top_gainers)
                        gainers_df.columns = ['Asset', 'Change (%)']
                        
                        # Format for display
                        gainers_df['Change (%)'] = gainers_df['Change (%)'].apply(
                            lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
                        )
                        
                        st.dataframe(gainers_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No top gainers data available")
                
                with col2:
                    st.write("### 📉 Top Losers")
                    
                    top_losers = sentiment_data.get('top_losers', [])
                    if top_losers:
                        # Create a dataframe for better display
                        losers_df = pd.DataFrame(top_losers)
                        losers_df.columns = ['Asset', 'Change (%)']
                        
                        # Format for display
                        losers_df['Change (%)'] = losers_df['Change (%)'].apply(
                            lambda x: f"{x:.1f}%"
                        )
                        
                        st.dataframe(losers_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No top losers data available")
                
                # Market Sentiment Indicators Explanation
                with st.expander("Understanding Market Sentiment Indicators"):
                    st.markdown("""
                    ### 📊 Come interpretare i nostri indicatori di sentiment
                    
                    - **Score di Sentiment (0-100)**: Un valore sopra 60 indica un mercato generalmente bullish (rialzista), 
                      mentre un valore sotto 40 indica un mercato bearish (ribassista). Valori tra 40-60 indicano 
                      un mercato neutrale o incerto.
                      
                    - **Direzione di Mercato**: Indica la tendenza generale del mercato basata su molteplici fattori 
                      tecnici e di volume. Può essere Bullish (rialzista), Bearish (ribassista) o una variazione intermedia.
                      
                    - **Forza del Sentiment**: Indica quanto è forte e coerente il sentiment rilevato. Un sentiment 
                      "Strong" o "Very Strong" suggerisce un consenso significativo nella direzione indicata.
                      
                    - **Percentuali Bullish/Bearish/Neutral**: Mostrano la distribuzione del sentiment tra i principali 
                      asset del mercato crypto. Una percentuale elevata di asset bullish (> 60%) suggerisce un mercato 
                      complessivamente positivo.
                      
                    - **Top Gainers/Losers**: Gli asset con le migliori e peggiori performance nel periodo recente, 
                      utili per identificare potenziali opportunità o asset da evitare.
                      
                    ### ⚠️ Nota Importante
                    Il sentiment di mercato è solo uno degli strumenti da considerare nelle decisioni di trading. 
                    Combinalo sempre con l'analisi tecnica e fondamentale per decisioni più informate.
                    """)
                
                # Add a refresh button
                if st.button("🔄 Refresh Market Sentiment"):
                    st.rerun()
            else:
                st.error("Unable to retrieve market sentiment data. Please try again later.")
        
        except Exception as e:
            st.error(f"Error analyzing market sentiment: {str(e)}")
            st.info("The Market Sentiment feature requires access to market data. Please ensure you have a stable internet connection.")
            
    with tab3:
        st.header("Price Charts & Indicators")
        
        for crypto in selected_cryptos:
            st.subheader(f"{crypto} - {timeframe_options[selected_timeframe]}")
            
            df = load_data(crypto, selected_timeframe)
            if df is not None and not df.empty:
                # Create price chart with indicators
                fig = visualization.create_chart(
                    df, crypto, 
                    show_signals=show_signals_on_chart
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators values
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rsi_value = df['rsi'].iloc[-1]
                    rsi_delta = df['rsi'].iloc[-1] - df['rsi'].iloc[-2]
                    rsi_color = "normal"
                    if rsi_value > rsi_overbought:
                        rsi_color = "inverse"  # overbought
                    elif rsi_value < rsi_oversold:
                        rsi_color = "off"  # oversold
                    
                    st.metric(
                        "RSI", 
                        f"{rsi_value:.2f}", 
                        delta=f"{rsi_delta:.2f}",
                        delta_color=rsi_color
                    )
                
                with col2:
                    macd_val = df['macd'].iloc[-1]
                    macd_signal_val = df['macd_signal'].iloc[-1]
                    macd_hist = df['macd_hist'].iloc[-1]
                    macd_delta_color = "normal" if macd_hist > 0 else "inverse"
                    
                    st.metric(
                        "MACD", 
                        f"{macd_val:.2f}", 
                        delta=f"Hist: {macd_hist:.2f}",
                        delta_color=macd_delta_color
                    )
                
                with col3:
                    ma_fast = df['ma_fast'].iloc[-1]
                    ma_slow = df['ma_slow'].iloc[-1]
                    ma_delta = ((ma_fast/ma_slow)-1)*100
                    ma_delta_color = "normal" if ma_delta > 0 else "inverse"
                    
                    st.metric(
                        f"MA {ma_fast_period}/{ma_slow_period}", 
                        f"{ma_fast:.2f}/{ma_slow:.2f}", 
                        delta=f"{ma_delta:.2f}%",
                        delta_color=ma_delta_color
                    )
                
                # Bollinger Bands
                with col4:
                    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                        bb_upper = df['bb_upper'].iloc[-1]
                        bb_lower = df['bb_lower'].iloc[-1]
                        bb_width = ((bb_upper - bb_lower) / df['bb_ma'].iloc[-1]) * 100
                        
                        current_price = df['close'].iloc[-1]
                        # Calculate position within bands (0-100%)
                        band_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
                        
                        # Set color based on position in bands
                        position_color = "normal"
                        if band_position > 80:
                            position_color = "inverse"  # near upper band
                        elif band_position < 20:
                            position_color = "off"  # near lower band
                        
                        st.metric(
                            "Bollinger Bands", 
                            f"Width: {bb_width:.2f}%", 
                            delta=f"Position: {band_position:.0f}%",
                            delta_color=position_color
                        )
                
                # Additional metrics in expandable section
                with st.expander("Additional Metrics"):
                    col1, col2, col3 = st.columns(3)
                    
                    # Volatility
                    with col1:
                        # Use the ATR value that's already calculated
                        atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                        volatility = (atr_value / df['close'].iloc[-1]) * 100
                        st.metric("Volatility (ATR %)", f"{volatility:.2f}%")
                    
                    # Volume analysis
                    with col2:
                        vol_avg = df['volume'].rolling(20).mean().iloc[-1]
                        vol_current = df['volume'].iloc[-1]
                        vol_ratio = (vol_current / vol_avg)
                        st.metric("Volume Ratio", f"{vol_ratio:.2f}x", f"{(vol_ratio-1)*100:.0f}%")
                    
                    # Trend strength
                    with col3:
                        adx_val = df['adx'].iloc[-1] if 'adx' in df.columns else 0
                        
                        trend_strength = "Weak"
                        if adx_val > 25:
                            trend_strength = "Moderate"
                        if adx_val > 50:
                            trend_strength = "Strong"
                        if adx_val > 75:
                            trend_strength = "Very Strong"
                            
                        st.metric("Trend Strength (ADX)", f"{adx_val:.1f}", trend_strength)
            else:
                st.error(f"Failed to load data for {crypto}")
    
    with tab3:
        st.header("Signal Performance Analysis")
        
        # Split screen for signal history and current validation
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Historical Signal Results")
            
            # Load signal history
            performance_data = performance_tracker.load_performance_data()
            signals_history = performance_data.get("signals_history", [])
            
            if signals_history:
                # Convert to DataFrame
                signals_df = pd.DataFrame(signals_history)
                
                # Calculate signal success rate metrics
                metrics = utils.calculate_signal_success_metrics(signals_history)
                
                # Performance metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Total Signals", metrics["total_signals"])
                
                with metric_col2:
                    success_color = "normal" if metrics["success_rate"] >= 50 else "inverse"
                    st.metric("Success Rate", f"{metrics['success_rate']:.1f}%", delta_color=success_color)
                
                with metric_col3:
                    profit_color = "normal" if metrics["avg_profit_pct"] > 0 else "inverse"
                    st.metric("Avg. Profit", f"{metrics['avg_profit_pct']:.2f}%", delta_color=profit_color)
                
                with metric_col4:
                    st.metric("Win/Loss Ratio", f"{metrics['win_loss_ratio']:.2f}")
                
                # Show historical signals
                if show_historical_signals:
                    # Filter and sort signals
                    filter_option = st.radio(
                        "Filter signals by:",
                        ["All Signals", "Executed Only", "Recent Only", "By Cryptocurrency"],
                        horizontal=True
                    )
                    
                    filtered_signals = signals_df
                    
                    if filter_option == "Executed Only":
                        filtered_signals = signals_df[signals_df["executed"] == True]
                    elif filter_option == "Recent Only":
                        # Get signals from last 7 days
                        filtered_signals = signals_df[pd.to_datetime(signals_df["timestamp"]) > (datetime.now() - timedelta(days=7))]
                    elif filter_option == "By Cryptocurrency":
                        crypto_filter = st.selectbox("Select Cryptocurrency", 
                                                   options=sorted(signals_df["symbol"].unique()))
                        filtered_signals = signals_df[signals_df["symbol"] == crypto_filter]
                    
                    # Convert timestamps and sort
                    filtered_signals["timestamp"] = pd.to_datetime(filtered_signals["timestamp"])
                    filtered_signals = filtered_signals.sort_values("timestamp", ascending=False)
                    
                    # Display results
                    if not filtered_signals.empty:
                        # Display columns based on what's useful
                        display_cols = ["symbol", "type", "price", "entry_point", "stop_loss", 
                                        "take_profit", "timestamp", "executed"]
                        
                        st.dataframe(
                            filtered_signals[display_cols],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No signals found with the selected filter.")
            else:
                st.info("No signal history available yet.")
        
        with col2:
            st.subheader("Signal Quality Analysis")
            
            # Get current signals for analysis
            current_signals = []
            for crypto in selected_cryptos:
                df = load_data(crypto, selected_timeframe)
                if df is not None and not df.empty:
                    signals = signal_generator.get_recent_signals(df, crypto, timeframe=selected_timeframe)
                    if signals:
                        current_signals.extend(signals)
                        
            if current_signals:
                # Display validation metrics for current signals
                for signal in current_signals:
                    with st.container(border=True):
                        st.write(f"**{signal['symbol']}** - {signal['signal_type'].upper()}")
                        
                        # Calculate validation metrics
                        validation_metrics = signal_validator.get_signal_validation_metrics(signal)
                        
                        # Signal quality gauge
                        st.progress(validation_metrics["quality_score"]/100, 
                                   text=f"Quality Score: {validation_metrics['quality_score']:.0f}%")
                        
                        # Confirmation metrics
                        st.write(f"Confirmations: {validation_metrics['confirmation_count']}/{validation_metrics['total_indicators']}")
                        
                        # Risk/reward assessment
                        st.write(f"Risk/Reward: 1:{validation_metrics['risk_reward_ratio']:.2f}")
                        
                        # Market condition assessment
                        st.write(f"Market Condition: {validation_metrics['market_condition']}")
            else:
                st.info("No current signals to analyze.")
    
    with tab4:
        st.header("Performance Dashboard")
        
        performance_data = performance_tracker.load_performance_data()
        
        # Portfolio summary
        portfolio = performance_data.get("portfolio", {})
        initial_value = portfolio.get("initial_value", 10000.0)
        current_value = portfolio.get("current_value", 10000.0)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_return = ((current_value / initial_value) - 1) * 100
            return_color = "normal" if total_return >= 0 else "inverse"
            st.metric(
                "Total Return", 
                f"{total_return:.2f}%", 
                delta=f"${current_value - initial_value:.2f}",
                delta_color=return_color
            )
        
        with col2:
            # Calculate win rate from trades
            trades = performance_data.get("trades", [])
            closed_trades = [t for t in trades if t.get("status") == "closed"]
            
            if closed_trades:
                profitable_trades = [t for t in closed_trades if t.get("profit_loss", 0) > 0]
                win_rate = (len(profitable_trades) / len(closed_trades)) * 100
                
                st.metric(
                    "Win Rate", 
                    f"{win_rate:.1f}%", 
                    f"{len(profitable_trades)}/{len(closed_trades)} trades"
                )
            else:
                st.metric("Win Rate", "N/A", "No closed trades")
        
        with col3:
            # Calculate average trade return
            if closed_trades:
                avg_return = np.mean([t.get("profit_loss_percent", 0) for t in closed_trades])
                
                st.metric(
                    "Avg Trade Return", 
                    f"{avg_return:.2f}%", 
                    delta_color="normal" if avg_return >= 0 else "inverse"
                )
            else:
                st.metric("Avg Trade Return", "N/A", "No closed trades")
        
        # Performance chart
        st.subheader("Portfolio Performance")
        
        performance_history = performance_data.get("performance_history", [])
        if performance_history:
            fig = performance_tracker.create_performance_dashboard()
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance history available yet.")
        
        # Trade history
        st.subheader("Trade History")
        
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Convert timestamps
            if "entry_time" in trades_df.columns:
                trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            if "exit_time" in trades_df.columns:
                trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
            
            # Sort by entry time, most recent first
            trades_df = trades_df.sort_values("entry_time", ascending=False)
            
            # Display columns
            display_cols = ["symbol", "type", "entry_price", "exit_price", 
                           "quantity", "profit_loss", "profit_loss_percent", 
                           "entry_time", "exit_time", "status"]
            
            st.dataframe(
                trades_df[display_cols], 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No trades recorded yet.")
    
    with tab5:
        st.header("Trading Journal")
        
        # Create a form to add trades manually
        st.subheader("Add Trade")
        
        with st.form("add_trade_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                trade_symbol = st.selectbox("Symbol", options=available_cryptos)
                trade_type = st.radio("Type", options=["buy", "sell"], horizontal=True)
                trade_price = st.number_input("Entry Price", min_value=0.0, step=0.0001)
                trade_quantity = st.number_input("Quantity", min_value=0.0, step=0.01)
            
            with col2:
                trade_stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.0001)
                trade_take_profit = st.number_input("Take Profit", min_value=0.0, step=0.0001)
                trade_notes = st.text_area("Notes", placeholder="Add trade notes here...")
            
            submit_trade = st.form_submit_button("Add Trade")
        
        if submit_trade:
            if trade_price <= 0 or trade_quantity <= 0:
                st.error("Please enter valid price and quantity values.")
            else:
                try:
                    # Add the trade to performance tracker
                    trade = performance_tracker.add_trade(
                        symbol=trade_symbol,
                        trade_type=trade_type,
                        entry_price=trade_price,
                        quantity=trade_quantity,
                        stop_loss=trade_stop_loss if trade_stop_loss > 0 else None,
                        take_profit=trade_take_profit if trade_take_profit > 0 else None
                    )
                    
                    if trade:
                        st.success(f"Trade added successfully: {trade_type.upper()} {trade_quantity} {trade_symbol} at {trade_price}")
                except Exception as e:
                    st.error(f"Error adding trade: {str(e)}")
        
        # Form to close trades
        st.subheader("Close Trade")
        
        # Get open trades
        performance_data = performance_tracker.load_performance_data()
        open_trades = [t for t in performance_data.get("trades", []) if t.get("status") == "open"]
        
        if open_trades:
            with st.form("close_trade_form"):
                trade_id = st.selectbox(
                    "Select Trade to Close",
                    options=[t["id"] for t in open_trades],
                    format_func=lambda x: next((f"ID {x}: {t['type'].upper()} {t['quantity']} {t['symbol']} at {t['entry_price']}" 
                                               for t in open_trades if t["id"] == x), "")
                )
                
                # Get current price for the selected trade
                selected_trade = next((t for t in open_trades if t["id"] == trade_id), None)
                if selected_trade:
                    current_price = crypto_data.get_current_price(selected_trade["symbol"])
                    
                exit_price = st.number_input(
                    "Exit Price", 
                    min_value=0.0, 
                    value=current_price if current_price else 0.0,
                    step=0.0001
                )
                
                close_notes = st.text_area("Close Notes", placeholder="Add notes about the trade close...")
                
                submit_close = st.form_submit_button("Close Trade")
            
            if submit_close:
                if exit_price <= 0:
                    st.error("Please enter a valid exit price.")
                else:
                    try:
                        # Close the trade
                        closed_trade = performance_tracker.close_trade(trade_id, exit_price)
                        
                        if closed_trade:
                            profit_loss = closed_trade.get("profit_loss", 0)
                            profit_loss_percent = closed_trade.get("profit_loss_percent", 0)
                            
                            result_color = "green" if profit_loss > 0 else "red"
                            result_text = "PROFIT" if profit_loss > 0 else "LOSS"
                            
                            st.success(f"Trade closed successfully with {result_text}: {profit_loss:.2f} USD ({profit_loss_percent:.2f}%)")
                    except Exception as e:
                        st.error(f"Error closing trade: {str(e)}")
        else:
            st.info("No open trades to close.")
            
        # Update portfolio value manually
        st.subheader("Update Portfolio Value")
        
        with st.form("update_portfolio_form"):
            current_portfolio_value = performance_data.get("portfolio", {}).get("current_value", 10000.0)
            
            new_portfolio_value = st.number_input(
                "Current Portfolio Value (USD)",
                min_value=0.0,
                value=float(current_portfolio_value),
                step=100.0
            )
            
            update_notes = st.text_area("Update Notes", placeholder="Reason for manual update...")
            
            submit_update = st.form_submit_button("Update Portfolio Value")
        
        if submit_update:
            if new_portfolio_value <= 0:
                st.error("Please enter a valid portfolio value.")
            else:
                try:
                    # Update portfolio value
                    updated_portfolio = performance_tracker.update_portfolio_value(new_portfolio_value)
                    
                    if updated_portfolio:
                        st.success(f"Portfolio value updated to ${new_portfolio_value:.2f}")
                except Exception as e:
                    st.error(f"Error updating portfolio value: {str(e)}")

# Auto-refresh setup
if auto_refresh:
    time.sleep(30)
    st.rerun()
