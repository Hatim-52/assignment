"""
Entry point: Launch Streamlit Chat UI.
Run with: streamlit run app.py
"""
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Re-export the Streamlit app from src.ui.chat_app
# Streamlit runs this file directly, so we import and re-run
from src.ui.chat_app import *  # noqa: F401, F403
