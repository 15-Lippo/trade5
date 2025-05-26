import os
import sys

# Add the TradePrecision directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "TradePrecision"))

# Import and run the main app
from TradePrecision.app import *

# The main code from app.py will execute when this file is run
