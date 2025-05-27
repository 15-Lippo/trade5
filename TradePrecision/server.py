from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__,
    static_folder='static',
    template_folder='templates')

# Configure CORS to allow all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Ensure static files are cached properly
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/market-data')
def market_data():
    # Generate dates and prices for the chart
    dates = pd.date_range(start='2025-01-01', end='2025-06-23', freq='D')
    prices = pd.Series(np.random.normal(100, 10, len(dates)))
    
    # Convert dates to string format for JSON serialization
    dates_str = dates.strftime('%Y-%m-%d').tolist()
    prices_list = prices.round(2).tolist()
    
    # Complete response with all required data
    response = {
        'dates': dates_str,
        'prices': prices_list,
        'industryGroups': {
            'leaders': ['Software', 'Semiconductors', 'Healthcare Equipment'],
            'laggards': ['Airlines', 'Hotels', 'Energy'],
            'signals': {
                'Software': {'strength': 0.85, 'risk': 'LOW'},
                'Semiconductors': {'strength': 0.78, 'risk': 'MEDIUM'},
                'Healthcare': {'strength': 0.72, 'risk': 'LOW'}
            }
        },
        'sectorRotation': {
            'phase': 'Mid Recovery',
            'leaders': ['Technology', 'Materials', 'Energy'],
            'confidence': 0.82
        },
        'seasonal': {
            'currentPattern': 'Bullish',
            'upcomingEvents': [
                {'name': 'Earnings Season', 'days': 15},
                {'name': 'Fed Meeting', 'days': 22}
            ],
            'confidence': 0.75
        }
    }
    return jsonify(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
