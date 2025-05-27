import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_chart(df, symbol, show_signals=True):
    """
    Create an interactive chart with price, indicators, and signals
    
    Parameters:
    df (pd.DataFrame): DataFrame with price and indicators data
    symbol (str): Cryptocurrency symbol
    show_signals (bool): Whether to show signals on the chart
    
    Returns:
    go.Figure: Plotly figure with the chart
    """
    # Create subplot figure with 3 rows
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol} Price", "MACD", "RSI")
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add Moving Averages
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ma_fast'],
            name=f"MA Fast",
            line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ma_slow'],
            name=f"MA Slow",
            line=dict(color='rgba(46, 139, 87, 0.8)', width=1.5)
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands if available
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_ma' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                name="BB Upper",
                line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                name="BB Lower",
                line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.2)',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_ma'],
                name="BB MA",
                line=dict(color='rgba(173, 216, 230, 0.8)', width=1, dash='dot'),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add MACD
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['macd'],
            name="MACD",
            line=dict(color='rgba(0, 0, 255, 0.8)', width=1.5)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['macd_signal'],
            name="Signal",
            line=dict(color='rgba(255, 0, 0, 0.8)', width=1.5)
        ),
        row=2, col=1
    )
    
    # Add MACD histogram
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['macd_hist'],
            name="Histogram",
            marker=dict(
                color=np.where(df['macd_hist'] >= 0, 'rgba(0, 255, 0, 0.7)', 'rgba(255, 0, 0, 0.7)')
            )
        ),
        row=2, col=1
    )
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['rsi'],
            name="RSI",
            line=dict(color='purple', width=1.5)
        ),
        row=3, col=1
    )
    
    # Add RSI levels
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[70] * len(df),
            name="Overbought",
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[30] * len(df),
            name="Oversold",
            line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dash')
        ),
        row=3, col=1
    )
    
    # Add signals if requested
    if show_signals:
        # Buy signals
        buy_signals = df[df['signal'] == 'buy']
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['low'] * 0.99,  # Place below the candle
                name="Buy",
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="green",
                    line=dict(width=1, color="darkgreen")
                )
            ),
            row=1, col=1
        )
        
        # Sell signals
        sell_signals = df[df['signal'] == 'sell']
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['high'] * 1.01,  # Place above the candle
                name="Sell",
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="red",
                    line=dict(width=1, color="darkred")
                )
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def create_performance_chart(hist_df):
    """
    Create a performance chart for historical signals
    
    Parameters:
    hist_df (pd.DataFrame): DataFrame with historical signals
    
    Returns:
    go.Figure: Plotly figure with the performance chart
    """
    # Filter signals with profit/loss data
    signals_with_pl = hist_df[hist_df['profit_loss'].notna()].copy()
    
    if signals_with_pl.empty:
        # Return empty chart if no data
        fig = go.Figure()
        fig.update_layout(
            title="No historical performance data available",
            xaxis_title="Date",
            yaxis_title="Profit/Loss (%)",
            template="plotly_dark"
        )
        return fig
    
    # Sort by timestamp
    signals_with_pl.sort_values('timestamp', inplace=True)
    
    # Calculate cumulative profit/loss
    signals_with_pl['cumulative_pl'] = signals_with_pl['profit_loss'].cumsum()
    
    # Create figure
    fig = go.Figure()
    
    # Add cumulative profit/loss line
    fig.add_trace(
        go.Scatter(
            x=signals_with_pl['timestamp'],
            y=signals_with_pl['cumulative_pl'],
            name="Cumulative P/L",
            line=dict(color='cyan', width=2)
        )
    )
    
    # Add individual trades as markers
    fig.add_trace(
        go.Scatter(
            x=signals_with_pl['timestamp'],
            y=signals_with_pl['profit_loss'],
            name="Individual Trades",
            mode="markers",
            marker=dict(
                size=8,
                color=np.where(signals_with_pl['profit_loss'] >= 0, 'green', 'red'),
                line=dict(width=1, color='darkgray')
            )
        )
    )
    
    # Add zero line
    fig.add_trace(
        go.Scatter(
            x=signals_with_pl['timestamp'],
            y=[0] * len(signals_with_pl),
            name="Break-even",
            line=dict(color='gray', width=1, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Historical Signal Performance",
        xaxis_title="Date",
        yaxis_title="Profit/Loss (%)",
        template="plotly_dark",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig
