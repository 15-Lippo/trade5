import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# Definizione del percorso per il file JSON delle performance
PERFORMANCE_FILE = "performance_data.json"

def load_performance_data():
    """
    Carica i dati di performance dal file JSON
    
    Returns:
    dict: Dizionario con i dati di performance
    """
    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Errore nel caricamento dei dati di performance: {str(e)}")
            return create_empty_performance_data()
    else:
        return create_empty_performance_data()

def save_performance_data(data):
    """
    Salva i dati di performance nel file JSON
    
    Parameters:
    data (dict): Dizionario con i dati di performance
    """
    try:
        with open(PERFORMANCE_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.error(f"Errore nel salvataggio dei dati di performance: {str(e)}")

def create_empty_performance_data():
    """
    Crea una struttura dati vuota per le performance
    
    Returns:
    dict: Struttura dati vuota per le performance
    """
    return {
        "portfolio": {
            "initial_value": 10000.0,  # Valore iniziale del portafoglio (USD)
            "current_value": 10000.0,   # Valore corrente del portafoglio (USD)
            "last_updated": datetime.now().isoformat()
        },
        "trades": [],
        "positions": [],
        "performance_history": [],
        "signals_history": []
    }

def add_trade(symbol, trade_type, entry_price, quantity, stop_loss=None, take_profit=None):
    """
    Aggiunge un nuovo trade al tracker di performance
    
    Parameters:
    symbol (str): Simbolo della criptovaluta
    trade_type (str): Tipo di trade ('buy' o 'sell')
    entry_price (float): Prezzo di entrata
    quantity (float): Quantità acquistata/venduta
    stop_loss (float, optional): Livello di stop loss
    take_profit (float, optional): Livello di take profit
    
    Returns:
    dict: Dettagli del trade creato
    """
    data = load_performance_data()
    
    trade_id = len(data["trades"]) + 1
    timestamp = datetime.now().isoformat()
    
    trade = {
        "id": trade_id,
        "symbol": symbol,
        "type": trade_type,
        "entry_price": entry_price,
        "quantity": quantity,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "entry_time": timestamp,
        "exit_time": None,
        "exit_price": None,
        "profit_loss": None,
        "profit_loss_percent": None,
        "status": "open"
    }
    
    data["trades"].append(trade)
    
    # Aggiungi la posizione se non esiste già una posizione aperta per questo simbolo
    existing_position = next((p for p in data["positions"] if p["symbol"] == symbol and p["status"] == "open"), None)
    
    if existing_position:
        # Aggiorna la posizione esistente (media il prezzo di ingresso)
        existing_position["quantity"] += quantity if trade_type == "buy" else -quantity
        # Aggiorna status se la posizione diventa neutra
        if existing_position["quantity"] == 0:
            existing_position["status"] = "closed"
            existing_position["exit_time"] = timestamp
    else:
        # Crea una nuova posizione
        position = {
            "id": len(data["positions"]) + 1,
            "symbol": symbol,
            "entry_price": entry_price,
            "quantity": quantity if trade_type == "buy" else -quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": timestamp,
            "exit_time": None,
            "status": "open"
        }
        data["positions"].append(position)
    
    save_performance_data(data)
    return trade

def close_trade(trade_id, exit_price):
    """
    Chiude un trade esistente
    
    Parameters:
    trade_id (int): ID del trade da chiudere
    exit_price (float): Prezzo di uscita
    
    Returns:
    dict: Dettagli del trade aggiornato
    """
    data = load_performance_data()
    
    # Trova il trade da chiudere
    trade = next((t for t in data["trades"] if t["id"] == trade_id and t["status"] == "open"), None)
    
    if not trade:
        st.error(f"Trade ID {trade_id} non trovato o già chiuso.")
        return None
    
    # Aggiorna i dettagli del trade
    timestamp = datetime.now().isoformat()
    trade["exit_time"] = timestamp
    trade["exit_price"] = exit_price
    trade["status"] = "closed"
    
    # Calcola il profitto/perdita
    if trade["type"] == "buy":
        profit_loss = (exit_price - trade["entry_price"]) * trade["quantity"]
        profit_loss_percent = ((exit_price / trade["entry_price"]) - 1) * 100
    else:  # sell
        profit_loss = (trade["entry_price"] - exit_price) * trade["quantity"]
        profit_loss_percent = ((trade["entry_price"] / exit_price) - 1) * 100
    
    trade["profit_loss"] = profit_loss
    trade["profit_loss_percent"] = profit_loss_percent
    
    # Aggiorna il valore del portafoglio
    data["portfolio"]["current_value"] += profit_loss
    data["portfolio"]["last_updated"] = timestamp
    
    # Aggiungi alla cronologia delle performance
    performance_entry = {
        "timestamp": timestamp,
        "portfolio_value": data["portfolio"]["current_value"],
        "trade_id": trade_id,
        "symbol": trade["symbol"],
        "profit_loss": profit_loss,
        "profit_loss_percent": profit_loss_percent
    }
    data["performance_history"].append(performance_entry)
    
    # Chiudi la posizione se necessario
    position = next((p for p in data["positions"] if p["symbol"] == trade["symbol"] and p["status"] == "open"), None)
    if position:
        position_quantity = position["quantity"]
        trade_quantity = trade["quantity"] if trade["type"] == "buy" else -trade["quantity"]
        
        # Aggiorna la quantità della posizione
        position["quantity"] -= trade_quantity
        
        # Se la posizione è chiusa (quantità = 0), aggiorna lo stato
        if position["quantity"] == 0:
            position["status"] = "closed"
            position["exit_time"] = timestamp
    
    save_performance_data(data)
    return trade

def add_signal(symbol, signal_type, price, timeframe, entry_point=None, stop_loss=None, take_profit=None):
    """
    Aggiunge un segnale alla cronologia dei segnali
    
    Parameters:
    symbol (str): Simbolo della criptovaluta
    signal_type (str): Tipo di segnale ('buy', 'sell', 'neutral')
    price (float): Prezzo corrente
    timeframe (str): Timeframe del segnale
    entry_point (float, optional): Punto di ingresso suggerito
    stop_loss (float, optional): Livello di stop loss
    take_profit (float, optional): Livello di take profit
    
    Returns:
    dict: Dettagli del segnale aggiunto
    """
    data = load_performance_data()
    
    timestamp = datetime.now().isoformat()
    
    signal = {
        "id": len(data["signals_history"]) + 1,
        "symbol": symbol,
        "type": signal_type,
        "price": price,
        "timeframe": timeframe,
        "entry_point": entry_point,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "timestamp": timestamp,
        "executed": False
    }
    
    data["signals_history"].append(signal)
    save_performance_data(data)
    return signal

def execute_signal(signal_id, executed_price=None, quantity=None):
    """
    Segna un segnale come eseguito e crea il relativo trade
    
    Parameters:
    signal_id (int): ID del segnale da eseguire
    executed_price (float, optional): Prezzo effettivo di esecuzione
    quantity (float, optional): Quantità da tradare
    
    Returns:
    dict: Dettagli del trade creato
    """
    data = load_performance_data()
    
    # Trova il segnale da eseguire
    signal = next((s for s in data["signals_history"] if s["id"] == signal_id and not s["executed"]), None)
    
    if not signal:
        st.error(f"Segnale ID {signal_id} non trovato o già eseguito.")
        return None
    
    # Segna il segnale come eseguito
    signal["executed"] = True
    save_performance_data(data)
    
    # Se non è specificato il prezzo di esecuzione, usa il prezzo del segnale
    if executed_price is None:
        executed_price = signal["price"]
    
    # Se non è specificata la quantità, calcola un valore predefinito
    if quantity is None:
        # Usa il 5% del valore del portafoglio per default
        portfolio_value = data["portfolio"]["current_value"]
        quantity = (portfolio_value * 0.05) / executed_price
    
    # Crea il trade basato sul segnale
    trade = add_trade(
        symbol=signal["symbol"],
        trade_type=signal["type"],
        entry_price=executed_price,
        quantity=quantity,
        stop_loss=signal["stop_loss"],
        take_profit=signal["take_profit"]
    )
    
    return trade

def update_portfolio_value(current_value=None):
    """
    Aggiorna il valore del portafoglio
    
    Parameters:
    current_value (float, optional): Nuovo valore del portafoglio
    
    Returns:
    dict: Portafoglio aggiornato
    """
    data = load_performance_data()
    
    timestamp = datetime.now().isoformat()
    
    if current_value is not None:
        data["portfolio"]["current_value"] = current_value
    
    data["portfolio"]["last_updated"] = timestamp
    
    # Aggiungi alla cronologia delle performance
    performance_entry = {
        "timestamp": timestamp,
        "portfolio_value": data["portfolio"]["current_value"],
        "trade_id": None,
        "symbol": None,
        "profit_loss": None,
        "profit_loss_percent": None
    }
    data["performance_history"].append(performance_entry)
    
    save_performance_data(data)
    return data["portfolio"]

def reset_performance_data():
    """
    Resetta i dati di performance
    
    Returns:
    dict: Nuova struttura dati vuota
    """
    data = create_empty_performance_data()
    save_performance_data(data)
    return data

def create_performance_dashboard():
    """
    Crea un dashboard di performance
    
    Returns:
    plotly.graph_objects.Figure: Figura Plotly con il dashboard
    """
    data = load_performance_data()
    
    # Converti la cronologia delle performance in DataFrame
    if data["performance_history"]:
        perf_df = pd.DataFrame(data["performance_history"])
        perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])
        perf_df = perf_df.sort_values("timestamp")
    else:
        # Crea un DataFrame vuoto se non ci sono dati
        perf_df = pd.DataFrame({
            "timestamp": [datetime.now() - timedelta(days=1), datetime.now()],
            "portfolio_value": [data["portfolio"]["initial_value"], data["portfolio"]["current_value"]]
        })
    
    # Calcola la performance in percentuale
    initial_value = data["portfolio"]["initial_value"]
    perf_df["performance_pct"] = ((perf_df["portfolio_value"] / initial_value) - 1) * 100
    
    # Crea il grafico
    fig = go.Figure()
    
    # Aggiungi la linea di performance del portafoglio
    fig.add_trace(go.Scatter(
        x=perf_df["timestamp"],
        y=perf_df["portfolio_value"],
        mode="lines",
        name="Portfolio Value (USD)",
        line=dict(color="#2E86C1", width=2)
    ))
    
    # Aggiungi i marker per i trade chiusi
    closed_trades = [t for t in data["trades"] if t["status"] == "closed"]
    if closed_trades:
        trade_df = pd.DataFrame(closed_trades)
        trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"])
        
        # Filtra per trade con profitto positivo
        profit_trades = trade_df[trade_df["profit_loss"] > 0]
        if not profit_trades.empty:
            # Trova i valori di portafoglio corrispondenti
            profit_times = profit_trades["exit_time"].tolist()
            profit_values = []
            for t in profit_times:
                idx = perf_df[perf_df["timestamp"] >= t].index.min()
                if pd.notna(idx):
                    profit_values.append(perf_df.loc[idx, "portfolio_value"])
                else:
                    profit_values.append(None)
            
            fig.add_trace(go.Scatter(
                x=profit_trades["exit_time"],
                y=profit_values,
                mode="markers",
                name="Profitable Trades",
                marker=dict(color="green", size=10, symbol="circle"),
                text=profit_trades.apply(lambda x: f"{x['symbol']}: +{x['profit_loss_percent']:.2f}%", axis=1),
                hoverinfo="text"
            ))
        
        # Filtra per trade con perdita
        loss_trades = trade_df[trade_df["profit_loss"] < 0]
        if not loss_trades.empty:
            # Trova i valori di portafoglio corrispondenti
            loss_times = loss_trades["exit_time"].tolist()
            loss_values = []
            for t in loss_times:
                idx = perf_df[perf_df["timestamp"] >= t].index.min()
                if pd.notna(idx):
                    loss_values.append(perf_df.loc[idx, "portfolio_value"])
                else:
                    loss_values.append(None)
            
            fig.add_trace(go.Scatter(
                x=loss_trades["exit_time"],
                y=loss_values,
                mode="markers",
                name="Loss Trades",
                marker=dict(color="red", size=10, symbol="circle"),
                text=loss_trades.apply(lambda x: f"{x['symbol']}: {x['profit_loss_percent']:.2f}%", axis=1),
                hoverinfo="text"
            ))
    
    # Imposta il layout del grafico
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def create_performance_metrics():
    """
    Calcola le metriche di performance
    
    Returns:
    dict: Metriche di performance
    """
    data = load_performance_data()
    
    # Valori del portafoglio
    initial_value = data["portfolio"]["initial_value"]
    current_value = data["portfolio"]["current_value"]
    
    # Calcolo della performance totale
    total_return = current_value - initial_value
    total_return_pct = ((current_value / initial_value) - 1) * 100
    
    # Analisi dei trade
    closed_trades = [t for t in data["trades"] if t["status"] == "closed"]
    total_trades = len(closed_trades)
    
    if total_trades > 0:
        winning_trades = len([t for t in closed_trades if t["profit_loss"] > 0])
        losing_trades = len([t for t in closed_trades if t["profit_loss"] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calcola il profitto/perdita medio
        avg_profit = np.mean([t["profit_loss_percent"] for t in closed_trades if t["profit_loss"] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t["profit_loss_percent"] for t in closed_trades if t["profit_loss"] < 0]) if losing_trades > 0 else 0
        
        # Calcola il rapporto rischio/rendimento
        risk_reward_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        
        # Calcola il profitto/perdita massimo
        max_profit = max([t["profit_loss_percent"] for t in closed_trades]) if closed_trades else 0
        max_loss = min([t["profit_loss_percent"] for t in closed_trades]) if closed_trades else 0
    else:
        winning_trades = 0
        losing_trades = 0
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
        risk_reward_ratio = 0
        max_profit = 0
        max_loss = 0
    
    # Posizioni aperte
    open_positions = [p for p in data["positions"] if p["status"] == "open"]
    
    # Metriche dei segnali
    total_signals = len(data["signals_history"])
    executed_signals = len([s for s in data["signals_history"] if s["executed"]])
    signal_execution_rate = (executed_signals / total_signals) * 100 if total_signals > 0 else 0
    
    # Risultato finale
    metrics = {
        "portfolio": {
            "initial_value": initial_value,
            "current_value": current_value,
            "total_return": total_return,
            "total_return_pct": total_return_pct
        },
        "trades": {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "risk_reward_ratio": risk_reward_ratio,
            "max_profit": max_profit,
            "max_loss": max_loss
        },
        "positions": {
            "open_positions": len(open_positions)
        },
        "signals": {
            "total_signals": total_signals,
            "executed_signals": executed_signals,
            "execution_rate": signal_execution_rate
        }
    }
    
    return metrics

def display_performance_dashboard():
    """
    Visualizza il dashboard di performance in Streamlit
    """
    st.header("Dashboard di Performance in Tempo Reale")
    
    # Ottieni le metriche di performance
    metrics = create_performance_metrics()
    
    # Visualizza le metriche principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Valore Portafoglio",
            f"${metrics['portfolio']['current_value']:.2f}",
            f"{metrics['portfolio']['total_return_pct']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Win Rate",
            f"{metrics['trades']['win_rate']:.2f}%",
            f"{metrics['trades']['winning_trades']} su {metrics['trades']['total_trades']} trade"
        )
    
    with col3:
        st.metric(
            "Profitto Medio",
            f"{metrics['trades']['avg_profit']:.2f}%",
            f"Max: {metrics['trades']['max_profit']:.2f}%"
        )
    
    with col4:
        st.metric(
            "Risk/Reward",
            f"{metrics['trades']['risk_reward_ratio']:.2f}",
            f"Perdita Media: {metrics['trades']['avg_loss']:.2f}%"
        )
    
    # Visualizza il grafico di performance
    fig = create_performance_dashboard()
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualizza posizioni aperte
    st.subheader("Posizioni Aperte")
    data = load_performance_data()
    open_positions = [p for p in data["positions"] if p["status"] == "open"]
    
    if open_positions:
        positions_df = pd.DataFrame(open_positions)
        positions_df["entry_time"] = pd.to_datetime(positions_df["entry_time"])
        
        # Formatta il DataFrame per la visualizzazione
        display_positions = positions_df[["symbol", "quantity", "entry_price", "stop_loss", "take_profit", "entry_time"]]
        display_positions = display_positions.rename(columns={
            "symbol": "Simbolo",
            "quantity": "Quantità",
            "entry_price": "Prezzo Entrata",
            "stop_loss": "Stop Loss",
            "take_profit": "Take Profit",
            "entry_time": "Data Entrata"
        })
        
        st.dataframe(display_positions, use_container_width=True)
    else:
        st.info("Nessuna posizione aperta al momento.")
    
    # Visualizza ultimi segnali
    st.subheader("Ultimi Segnali")
    signals_history = data["signals_history"]
    
    if signals_history:
        # Prendi gli ultimi 10 segnali
        recent_signals = sorted(signals_history, key=lambda x: x["timestamp"], reverse=True)[:10]
        signals_df = pd.DataFrame(recent_signals)
        signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
        
        # Formatta il DataFrame per la visualizzazione
        display_signals = signals_df[["symbol", "type", "price", "entry_point", "stop_loss", "take_profit", "timeframe", "timestamp", "executed"]]
        display_signals = display_signals.rename(columns={
            "symbol": "Simbolo",
            "type": "Tipo",
            "price": "Prezzo",
            "entry_point": "Punto Entrata",
            "stop_loss": "Stop Loss",
            "take_profit": "Take Profit",
            "timeframe": "Timeframe",
            "timestamp": "Data",
            "executed": "Eseguito"
        })
        
        st.dataframe(display_signals, use_container_width=True)
    else:
        st.info("Nessun segnale registrato.")
    
    # Pulsanti di azione
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Aggiorna Dashboard"):
            st.rerun()
    
    with col2:
        if st.button("Resetta Dati (Debug)"):
            reset_performance_data()
            st.success("Dati di performance resettati.")
            st.rerun()