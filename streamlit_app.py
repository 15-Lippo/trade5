# TradePrecision Streamlit App - Entry Point
# Questo file è un semplice entry point che esegue il codice principale
# dalla directory TradePrecision/app.py direttamente

import os
import sys

# Questa linea è cruciale: esegue il file nella directory principale 
# prima di importare streamlit, per garantire che set_page_config sia
# chiamato correttamente una sola volta

# Percorso al modulo principale dell'app
current_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_dir, "TradePrecision", "app.py")

# Aggiungi le directory necessarie al Python path
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "TradePrecision"))

# Esegui il file app.py direttamente
with open(app_path, 'r', encoding='utf-8') as f:
    app_code = f.read()
    # Esegui il codice in un namespace separato per evitare conflitti
    exec(app_code, {"__file__": app_path})
