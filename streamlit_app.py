import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Struttura di directory in Streamlit Cloud
TRADEPRECISION_DIR = os.path.join(os.path.dirname(__file__), "TradePrecision")
sys.path.append(TRADEPRECISION_DIR)

# Prova ad importare i moduli necessari direttamente
try:
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
    
    # Esegui il codice principale dell'app
    exec(open(os.path.join(TRADEPRECISION_DIR, "app.py"), encoding='utf-8').read())
except ImportError as e:
    st.error(f"Errore di importazione: {e}")
    st.write("Directory contenuto:")
    if os.path.exists(TRADEPRECISION_DIR):
        st.write(os.listdir(TRADEPRECISION_DIR))
    else:
        st.error(f"Directory {TRADEPRECISION_DIR} non trovata")
    
    st.write("Python path:")
    st.write(sys.path)
