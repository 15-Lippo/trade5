import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Nota: la configurazione della pagina (st.set_page_config) è stata spostata nel modulo TradePrecision.app
# per evitare il conflitto con chiamate multiple

# Messaggio di inizializzazione
st.write("Inizializzazione dell'applicazione...")

# Gestione dei percorsi per compatibilità tra ambiente locale e Streamlit Cloud
try:
    # Aggiungi il percorso base e il percorso TradePrecision al sys.path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tradeprecision_dir = os.path.join(base_dir, "TradePrecision")
    
    # Se non siamo in locale, prova il percorso relativo (per Streamlit Cloud)
    if not os.path.exists(tradeprecision_dir):
        tradeprecision_dir = "TradePrecision"
    
    # Debug info
    st.write(f"Directory TradePrecision: {tradeprecision_dir}")
    st.write(f"Esiste directory: {os.path.exists(tradeprecision_dir)}")
    
    # Aggiungi entrambi i percorsi a sys.path
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    if tradeprecision_dir not in sys.path:
        sys.path.insert(0, tradeprecision_dir)
    
    # Importa i moduli necessari
    try:
        # Importazione con namespace del pacchetto
        from TradePrecision import app as tp_app
        # Esegui l'applicazione principale
        tp_app.main()
    except ImportError as e:
        st.error(f"Errore nell'importazione del modulo TradePrecision: {e}")
        
        # Fallback: prova a importare i moduli direttamente
        try:
            # Mostra il contenuto delle directory per debugging
            st.write("Contenuto directory corrente:")
            st.write(os.listdir('.'))
            if os.path.exists(tradeprecision_dir):
                st.write(f"Contenuto directory {tradeprecision_dir}:")
                st.write(os.listdir(tradeprecision_dir))
            
            # Controlla se il file app.py esiste
            app_path = os.path.join(tradeprecision_dir, "app.py")
            if os.path.exists(app_path):
                # Importa direttamente i moduli necessari
                import sys
                sys.path.append('.')
                
                # Importa i moduli richiesti
                from TradePrecision.app import main
                main()
            else:
                st.error(f"File app.py non trovato in {tradeprecision_dir}")
        except Exception as inner_e:
            st.error(f"Fallimento anche nel metodo alternativo: {inner_e}")
            st.code(traceback.format_exc())

except Exception as e:
    import traceback
    st.error(f"Errore durante l'inizializzazione: {e}")
    st.code(traceback.format_exc())
    
    # Informazioni di debug
    st.write("Python path:")
    st.write(sys.path)
    st.write("Directory corrente:")
    st.write(os.listdir('.'))
