import streamlit as st
import importlib
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os

# Add the current directory to the path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules - updated paths to ensure they work with streamlit.io deployment
from components.factory_components import FactoryComponents 
from components.live_metrics import render_live_metrics

# Import the interactive header
from components.interactive_header import AdvancedInteractiveHeader

# Import digital twin components properly with correct indentation
from components.digital_twin_components import (
    MachineConnector,
    DigitalTwinVisualizer,
    CameraManager,
    SensorProcessor
)

# Set page configuration at the very beginning - must be called first
st.set_page_config(
    page_title="HyperSyncDT Autonomous Agent Factory",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"  # Ensure sidebar is always expanded
)

# Create a fresh instance of the header 
if 'header' not in st.session_state:
    st.session_state.header = AdvancedInteractiveHeader()
    
# Initialize machine connector
if 'machine_connector' not in st.session_state:
    st.session_state.machine_connector = MachineConnector()
    
# Initialize factory components
if 'factory' not in st.session_state:
    st.session_state.factory = FactoryComponents()

# Main app container
main_container = st.container()
with main_container:
    # Render header in a container
    header_container = st.container()
    with header_container:
        st.session_state.header.render()
        
    # Add live metrics dashboard
    metrics_container = st.container()
    with metrics_container:
        render_live_metrics()
