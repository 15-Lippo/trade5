import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
import crypto_data
import technical_indicators
import signal_generator
import visualization
import performance_tracker

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
This dashboard provides trading signals for cryptocurrencies based on technical indicators.
The signals include entry points and stop-loss recommendations.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Get all available cryptocurrencies from Kraken
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_all_cryptos():
    try:
        return crypto_data.get_available_markets()
    except Exception as e:
        st.error(f"Error fetching available cryptocurrencies: {str(e)}")
        # Fallback to a predefined list in case of error
        return [
            'BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'ADA/USD',
            'LINK/USD', 'DOT/USD', 'AVAX/USD', 'DOGE/USD', 'MATIC/USD',
            'FET/USD', 'TAO/USD', 'VIRTUAL/USD', 'TRUMP/USD', 'PEPE/USD'
        ]

# Cryptocurrency selection
available_cryptos = get_all_cryptos()
selected_cryptos = st.sidebar.multiselect(
    "Select Cryptocurrencies",
    options=available_cryptos,
    default=['BTC/USD', 'ETH/USD', 'FET/USD', 'PEPE/USD', 'TRUMP/USD']
)

# Timeframe selection (solo quelli supportati da Kraken)
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

# Technical indicators parameters
st.sidebar.header("Indicator Parameters")

rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought Level", 70, 85, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold Level", 15, 30, 30)

macd_fast = st.sidebar.slider("MACD Fast Period", 8, 24, 12)
macd_slow = st.sidebar.slider("MACD Slow Period", 21, 52, 26)
macd_signal = st.sidebar.slider("MACD Signal Period", 5, 15, 9)

ma_fast_period = st.sidebar.slider("Fast MA Period", 5, 50, 20)
ma_slow_period = st.sidebar.slider("Slow MA Period", 50, 200, 50)

# Function to load and process data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(crypto, timeframe, limit=100):
    try:
        df = crypto_data.fetch_ohlcv_data(crypto, timeframe, limit)
        if df is not None and not df.empty:
            # Adatta parametri in base al timeframe per maggiore precisione
            if timeframe == '1M':
                # Per timeframe mensile, parametri molto più conservativi
                current_rsi_period = int(rsi_period * 2.5)     # RSI molto più lungo 
                current_macd_fast = int(macd_fast * 2.5)       # MACD fast molto più lungo
                current_macd_slow = int(macd_slow * 2.5)       # MACD slow molto più lungo
                current_macd_signal = int(macd_signal * 2.5)   # MACD signal molto più lungo
                current_ma_fast = int(ma_fast_period * 2.5)     # MA fast molto più lunga
                current_ma_slow = int(ma_slow_period * 2.5)     # MA slow molto più lunga
                current_rsi_overbought = min(90, rsi_overbought + 10)  # Soglia molto più alta
                current_rsi_oversold = max(10, rsi_oversold - 10)      # Soglia molto più bassa
            elif timeframe in ['4h', '1d', '1w', '2w']:
                # Per timeframe più lunghi, parametri più conservativi ma più sensibili di prima
                # Riduciamo la differenza per generare più segnali
                current_rsi_period = int(rsi_period * 1.2)       # RSI leggermente più lungo
                current_macd_fast = int(macd_fast * 1.2)         # MACD fast leggermente più lungo
                current_macd_slow = int(macd_slow * 1.2)         # MACD slow leggermente più lungo  
                current_macd_signal = int(macd_signal * 1.2)     # MACD signal leggermente più lungo
                current_ma_fast = int(ma_fast_period * 1.2)      # MA fast leggermente più lunga
                current_ma_slow = int(ma_slow_period * 1.2)      # MA slow leggermente più lunga
                current_rsi_overbought = min(80, rsi_overbought + 3)  # Soglia leggermente più alta
                current_rsi_oversold = max(20, rsi_oversold - 3)      # Soglia leggermente più bassa
            else:
                # Per timeframe più brevi, parametri standard dalla sidebar
                current_rsi_period = rsi_period
                current_macd_fast = macd_fast
                current_macd_slow = macd_slow  
                current_macd_signal = macd_signal
                current_ma_fast = ma_fast_period
                current_ma_slow = ma_slow_period
                current_rsi_overbought = rsi_overbought
                current_rsi_oversold = rsi_oversold
                
            # Calculate indicators con parametri adattati al timeframe
            df = technical_indicators.add_all_indicators(
                df, 
                rsi_period=current_rsi_period, 
                macd_fast=current_macd_fast, 
                macd_slow=current_macd_slow, 
                macd_signal=current_macd_signal,
                ma_fast_period=current_ma_fast,
                ma_slow_period=current_ma_slow
            )
            
            # Generate signals con parametri adattati al timeframe
            df = signal_generator.generate_signals(
                df, 
                rsi_overbought=current_rsi_overbought, 
                rsi_oversold=current_rsi_oversold,
                timeframe=timeframe  # Passiamo il timeframe per adattare la sensibilità
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
    tab1, tab2, tab3, tab4 = st.tabs(["Current Signals", "Charts", "Historical Performance", "Performance Dashboard"])
    
    with tab1:
        st.header("Current Trading Signals")
        
        all_signals = []
        for crypto in selected_cryptos:
            df = load_data(crypto, selected_timeframe)
            if df is not None and not df.empty:
                # Get the most recent signals con timeframe per adattare la ricerca
                recent_signals = signal_generator.get_recent_signals(df, crypto, timeframe=selected_timeframe)
                if recent_signals:
                    all_signals.extend(recent_signals)
        
        if all_signals:
            # Create a DataFrame for the signals
            signals_df = pd.DataFrame(all_signals)
            
            # Create three columns for different signal types
            col1, col2, col3 = st.columns(3)
            
            # Buy signals
            with col1:
                st.subheader("💹 Buy Signals")
                buy_signals = signals_df[signals_df['signal_type'] == 'buy']
                if not buy_signals.empty:
                    st.dataframe(buy_signals[['symbol', 'price', 'entry_point', 'stop_loss', 'take_profit', 'signal_strength', 'timestamp']], 
                                 use_container_width=True,
                                 hide_index=True)
                else:
                    st.info("No buy signals at the moment.")
            
            # Sell signals
            with col2:
                st.subheader("📉 Sell Signals")
                sell_signals = signals_df[signals_df['signal_type'] == 'sell']
                if not sell_signals.empty:
                    st.dataframe(sell_signals[['symbol', 'price', 'entry_point', 'stop_loss', 'take_profit', 'signal_strength', 'timestamp']], 
                                 use_container_width=True,
                                 hide_index=True)
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
        else:
            st.info("No signals generated for the selected cryptocurrencies and timeframe.")
            
    with tab2:
        st.header("Price Charts & Indicators")
        
        for crypto in selected_cryptos:
            st.subheader(f"{crypto} - {timeframe_options[selected_timeframe]}")
            
            df = load_data(crypto, selected_timeframe)
            if df is not None and not df.empty:
                # Create price chart with indicators
                fig = visualization.create_chart(df, crypto, show_signals=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators values
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "RSI", 
                        f"{df['rsi'].iloc[-1]:.2f}", 
                        delta=f"{df['rsi'].iloc[-1] - df['rsi'].iloc[-2]:.2f}"
                    )
                
                with col2:
                    macd_val = df['macd'].iloc[-1]
                    macd_signal_val = df['macd_signal'].iloc[-1]
                    macd_hist = df['macd_hist'].iloc[-1]
                    
                    st.metric(
                        "MACD", 
                        f"{macd_val:.2f}", 
                        delta=f"Hist: {macd_hist:.2f}"
                    )
                
                with col3:
                    ma_fast = df['ma_fast'].iloc[-1]
                    ma_slow = df['ma_slow'].iloc[-1]
                    
                    st.metric(
                        f"MA {ma_fast_period}/{ma_slow_period}", 
                        f"{ma_fast:.2f}/{ma_slow:.2f}", 
                        delta=f"{((ma_fast/ma_slow)-1)*100:.2f}%"
                    )
                
                # Bande di Bollinger (se disponibili)
                with col4:
                    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                        bb_upper = df['bb_upper'].iloc[-1]
                        bb_lower = df['bb_lower'].iloc[-1]
                        current_price = df['close'].iloc[-1]
                        
                        # Calcola la distanza percentuale dalle bande
                        distance_upper = ((bb_upper / current_price) - 1) * 100
                        distance_lower = ((current_price / bb_lower) - 1) * 100
                        
                        if distance_upper < distance_lower:
                            # Più vicino alla banda superiore
                            st.metric(
                                "Bollinger Bands", 
                                f"{bb_lower:.2f} - {bb_upper:.2f}", 
                                delta=f"{distance_upper:.2f}% dalla superiore"
                            )
                        else:
                            # Più vicino alla banda inferiore
                            st.metric(
                                "Bollinger Bands", 
                                f"{bb_lower:.2f} - {bb_upper:.2f}", 
                                delta=f"{distance_lower:.2f}% dalla inferiore"
                            )
            else:
                st.error(f"Failed to load data for {crypto}")
    
    with tab3:
        st.header("Historical Signal Performance")
        
        for crypto in selected_cryptos:
            st.subheader(f"{crypto} Signal History")
            
            df = load_data(crypto, selected_timeframe, limit=500)  # Get more historical data
            if df is not None and not df.empty:
                # Get historical signals
                historical_signals = signal_generator.get_historical_signals(df, crypto)
                
                if historical_signals:
                    # Convert to DataFrame
                    hist_df = pd.DataFrame(historical_signals)
                    
                    # Display performance stats
                    perf_metrics = signal_generator.calculate_performance(hist_df, df)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Win Rate", f"{perf_metrics['win_rate']:.2f}%")
                    
                    with col2:
                        st.metric("Avg Profit", f"{perf_metrics['avg_profit']:.2f}%")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{perf_metrics['max_drawdown']:.2f}%")
                    
                    with col4:
                        st.metric("Total Signals", perf_metrics['total_signals'])
                    
                    # Display historical signals
                    st.dataframe(
                        hist_df[['signal_type', 'price', 'entry_point', 'stop_loss', 'take_profit',
                                'exit_price', 'profit_loss', 'timestamp']], 
                        use_container_width=True
                    )
                    
                    # Performance chart
                    fig = visualization.create_performance_chart(hist_df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No historical signals for {crypto} in the selected timeframe.")
            else:
                st.error(f"Failed to load historical data for {crypto}")
    
    with tab4:
        # Aggiungiamo i segnali correnti al tracker di performance
        if all_signals:
            for signal in all_signals:
                # Aggiungiamo solo i segnali non-neutrali
                if signal['signal_type'] in ['buy', 'sell']:
                    performance_tracker.add_signal(
                        symbol=signal['symbol'],
                        signal_type=signal['signal_type'],
                        price=signal['price'],
                        timeframe=selected_timeframe,
                        entry_point=signal.get('entry_point'),
                        stop_loss=signal.get('stop_loss'),
                        take_profit=signal.get('take_profit')
                    )
        
        # Mostra il dashboard di performance
        performance_tracker.display_performance_dashboard()
        
        # Aggiungi interfaccia per l'esecuzione manuale dei segnali
        st.header("Esegui Segnale Manualmente")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox("Seleziona Criptovaluta", options=available_cryptos)
        
        with col2:
            trade_type = st.selectbox("Tipo di Trade", options=["buy", "sell"])
        
        with col3:
            quantity = st.number_input("Quantità", min_value=0.0, value=0.1, step=0.01)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            entry_price = st.number_input("Prezzo di Entrata", min_value=0.0, value=0.0, step=0.1)
        
        with col2:
            stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.1)
        
        with col3:
            take_profit = st.number_input("Take Profit", min_value=0.0, value=0.0, step=0.1)
        
        if st.button("Aggiungi Trade"):
            if symbol and entry_price > 0 and quantity > 0:
                trade = performance_tracker.add_trade(
                    symbol=symbol,
                    trade_type=trade_type,
                    entry_price=entry_price,
                    quantity=quantity,
                    stop_loss=stop_loss if stop_loss > 0 else None,
                    take_profit=take_profit if take_profit > 0 else None
                )
                
                if trade:
                    st.success(f"Trade aggiunto con successo: {trade_type.upper()} {quantity} {symbol} a ${entry_price}")
                    # Aggiorna il valore del portafoglio
                    performance_tracker.update_portfolio_value()
                    st.rerun()
            else:
                st.error("Inserire correttamente simbolo, prezzo e quantità")
        
        # Chiusura di un trade
        st.header("Chiudi Trade")
        
        data = performance_tracker.load_performance_data()
        open_trades = [t for t in data["trades"] if t["status"] == "open"]
        
        if open_trades:
            # Crea un dizionario per visualizzare i trade in modo leggibile
            trade_display = {
                f"{t['id']} - {t['type'].upper()} {t['quantity']} {t['symbol']} a ${t['entry_price']}": t['id'] 
                for t in open_trades
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_trade_id = st.selectbox(
                    "Seleziona Trade da Chiudere", 
                    options=list(trade_display.keys())
                )
                if selected_trade_id:  # Controlla che sia stato selezionato un trade
                    trade_id = trade_display[selected_trade_id]
                
                    with col2:
                        exit_price = st.number_input("Prezzo di Uscita", min_value=0.0, value=0.0, step=0.1)
                    
                    if st.button("Chiudi Trade"):
                        if exit_price > 0:
                            closed_trade = performance_tracker.close_trade(trade_id, exit_price)
                            if closed_trade:
                                profit_loss = closed_trade.get('profit_loss_percent', 0)
                                emoji = "🟢" if profit_loss > 0 else "🔴"
                                st.success(f"Trade chiuso con successo: {emoji} {profit_loss:.2f}%")
                                st.rerun()
                        else:
                            st.error("Inserire un prezzo di uscita valido")
        else:
            st.info("Nessun trade aperto da chiudere")

    # Add auto-refresh
    refresh_interval = st.sidebar.slider("Auto-refresh interval (seconds)", 30, 300, 60)
    st.sidebar.write(f"Next refresh in: {refresh_interval} seconds")
    
    if st.sidebar.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh countdown
    placeholder = st.sidebar.empty()
    
    # Display current time
    current_time = datetime.now().strftime("%H:%M:%S")
    st.sidebar.write(f"Last update: {current_time}")
    
    # Only auto-refresh when the app is in the foreground
    if st.sidebar.checkbox("Enable Auto-refresh", value=True):
        count = 0
        while count < refresh_interval:
            remaining = refresh_interval - count
            placeholder.write(f"Refreshing in {remaining} seconds...")
            time.sleep(1)
            count += 1
        
        placeholder.write("Refreshing data...")
        st.cache_data.clear()
        st.rerun()

else:
    st.warning("Please select at least one cryptocurrency in the sidebar.")
