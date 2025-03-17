import streamlit as st
import importlib
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from factory_components import FactoryComponents
from live_metrics import render_live_metrics  # Add this import

# Force reload the components module to pick up changes
# if 'components.interactive_header' in sys.modules:
#     importlib.reload(sys.modules['components.interactive_header'])

# Import AdvancedInteractiveHeader with exact path that works in Render
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from components.interactive_header import AdvancedInteractiveHeader

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

# Auto-refresh script
st.markdown("""
<script>
    const refreshInterval = 1000;  // 1 second
    
    function refreshVisualizations() {
        const elements = window.parent.document.querySelectorAll('.advanced-screen');
        elements.forEach(element => {
            const chart = element.querySelector('.js-plotly-plot');
            if (chart) {
                Plotly.update(chart);
            }
        });
    }
    
    // Set up the refresh interval
    setInterval(refreshVisualizations, refreshInterval);
    
    // Add advanced interaction handlers
    document.addEventListener('DOMContentLoaded', function() {
        const screens = document.querySelectorAll('.advanced-screen');
        screens.forEach(screen => {
            screen.addEventListener('mousemove', (e) => {
                const rect = screen.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                const xPercent = (x / rect.width - 0.5) * 20;
                const yPercent = (y / rect.height - 0.5) * 20;
                
                screen.style.transform = `
                    perspective(1000px)
                    rotateY(${xPercent}deg)
                    rotateX(${-yPercent}deg)
                    scale3d(1.02, 1.02, 1.02)
                    translateZ(20px)
                `;
            });
            
            screen.addEventListener('mouseleave', () => {
                screen.style.transform = 'none';
            });
        });
    });
</script>
""", unsafe_allow_html=True)

# All dashboard-related session state initializations have been removed

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import advanced_visualizations
from advanced_visualizations import MultiModalVisualizer
import time
import os
import random
from PIL import Image
import json
import importlib
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import StatevectorSampler
from advanced_visualization_page import render_advanced_visualization_page
from research_roadmap import render_research_roadmap
from tool_wear_analysis import render_tool_wear_analysis
from eda_workspace import render_eda_workspace, EDAWorkspace
from provider_management import render_provider_management
from model_performance import render_model_performance
from rag_assistant import render_rag_assistant
from factory_components import (
    render_factory_connect,
    render_factory_build,
    render_factory_analyze,
    render_factory_operate,
    render_sustainability,
    render_risk_mitigation,
    OperatorCoPilot,
    render_copilot,
    render_agent_factory,
    AgentFactory,
    SelfCalibratingShadow,
    FactoryComponents
)
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
import h5py
import tensorly as tl
from pathlib import Path
from digital_twin_components import (
    MachineConnector,
    DigitalTwinVisualizer,
    CameraManager,
    SensorProcessor
)
import asyncio
from rag_agent_creator import render_rag_agent_creator
from tool_condition_monitoring import render_tool_condition_monitoring
from wear_pattern_recognition import render_wear_pattern_recognition
from tool_life_prediction import render_tool_life_prediction
from virtual_testing import render_virtual_testing
from process_simulation import render_process_simulation
from what_if_analysis import render_what_if_analysis
from experiment_tracking import render_experiment_tracking
import requests
from research_roadmap import render_research_roadmap
# Import the live dashboard module

# Monkey patch for PyTorch custom class issue
import types
def _getattr_patch(self, attr):
    if attr == '__path__':
        return types.SimpleNamespace(_path=[])
    try:
        return self.__orig_getattr__(attr)
    except Exception as e:
        if 'Tried to instantiate class' in str(e):
            return None
        raise e

if hasattr(torch._classes, '__getattr__'):
    torch._classes.__orig_getattr__ = torch._classes.__getattr__
    torch._classes.__getattr__ = types.MethodType(_getattr_patch, torch._classes)

# Ensure we have a running event loop
def ensure_event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Explicitly handle torch._classes.__path__._path access
        # This prevents the error when Streamlit tries to inspect module paths
        if hasattr(torch, '_classes'):
            if not hasattr(torch._classes, '__path__'):
                class MockPath:
                    _path = []
                torch._classes.__path__ = MockPath()
    return loop

def generate_process_data(start_date: str, end_date: str, freq: str = '1h') -> pd.DataFrame:
    """Generate sample process data for the application.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        freq: Frequency of data points (default: '1h' for hourly)
    
    Returns:
        DataFrame containing generated process data
    """
    # Create date range - ensure we're using 'h' instead of 'H' for hours
    # Convert any legacy 'H' to 'h' to avoid FutureWarning
    if 'H' in freq:
        freq = freq.replace('H', 'h')
        
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_points = len(date_range)
    
    # Base signals
    base_temp = np.sin(np.linspace(0, 8*np.pi, n_points)) * 5 + 70  # Temperature around 70°C
    base_pressure = np.cos(np.linspace(0, 6*np.pi, n_points)) * 3 + 100  # Pressure around 100 PSI
    base_flow = np.sin(np.linspace(0, 4*np.pi, n_points)) * 2 + 75  # Flow rate around 75 L/min
    
    # Add noise and trends
    temperature = base_temp + np.random.normal(0, 0.5, n_points) + np.linspace(0, 2, n_points)
    pressure = base_pressure + np.random.normal(0, 0.3, n_points) - np.linspace(0, 1, n_points)
    flow_rate = base_flow + np.random.normal(0, 0.2, n_points)
    
    # Generate dependent variables
    vibration = 0.1 + 0.05 * np.sin(np.linspace(0, 10*np.pi, n_points)) + np.random.normal(0, 0.01, n_points)
    energy_consumption = 8 + temperature * 0.05 + np.random.normal(0, 0.2, n_points)
    tool_wear = np.cumsum(np.random.normal(0.001, 0.0002, n_points))
    quality_score = 98 - tool_wear * 10 + np.random.normal(0, 0.1, n_points)
    
    # Create DataFrame
    return pd.DataFrame({
        'timestamp': date_range,
        'temperature': temperature,
        'pressure': pressure,
        'flow_rate': flow_rate,
        'vibration': vibration,
        'energy_consumption': energy_consumption,
        'tool_wear': tool_wear,
        'quality_score': quality_score
    })

class HDF5QuantumMesh:
    """
    Quantum-enhanced HDF5 data management system optimized for M1 MAX.
    """
    def __init__(self):
        self.compression_ratio = 98
        self.quantum_bits = 3
        self.mesh_data = {}
        self.last_sync = datetime.now()
        
        # Initialize quantum backend if available
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.use_quantum = True
        else:
            self.device = torch.device("cpu")
            self.use_quantum = False
    
    def compress_data(self, data, compression_ratio=None, quantum_bits=None):
        """Compress data using quantum-enhanced tensor decomposition"""
        compression_ratio = compression_ratio or self.compression_ratio
        quantum_bits = quantum_bits or self.quantum_bits
        
        # Convert data to tensor
        if isinstance(data, np.ndarray):
            tensor_data = torch.from_numpy(data).to(self.device)
        else:
            tensor_data = torch.tensor(data).to(self.device)
        
        # Apply quantum optimization if available
        if self.use_quantum:
            tensor_data = self._apply_quantum_optimization(tensor_data)
        
        # Perform tensor decomposition
        core, factors = tl.decomposition.tucker(
            tensor_data.cpu().numpy(),
            rank=[int(s * compression_ratio/100) for s in tensor_data.shape],
            init='random'
        )
        
        return core, factors
    
    def _apply_quantum_optimization(self, tensor):
        """Apply quantum-inspired optimization"""
        # Simulate quantum advantage
        quantum_factor = 1 + (self.quantum_bits / 16)
        return tensor * quantum_factor
    
    def save_to_hdf5(self, data, filename):
        """Save compressed data to HDF5 file"""
        core, factors = self.compress_data(data)
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('core', data=core)
            for i, factor in enumerate(factors):
                f.create_dataset(f'factor_{i}', data=factor)
        
        return core, factors
    
    def load_from_hdf5(self, filename):
        """Load and decompress data from HDF5 file"""
        with h5py.File(filename, 'r') as f:
            core = f['core'][:]
            factors = [f[f'factor_{i}'][:] for i in range(len(f.keys())-1)]
        
        return tl.tucker_to_tensor((core, factors))
    
    def sync_mesh(self):
        """Synchronize quantum mesh data"""
        self.last_sync = datetime.now()
        return {
            'status': 'synchronized',
            'timestamp': self.last_sync,
            'device': str(self.device)
        }

class EnergyOptimization:
    """
    Energy optimization system with M1 acceleration support.
    """
    def __init__(self, quantum_mesh, threshold=10.0):
        self.quantum_mesh = quantum_mesh
        self.threshold = threshold
        self.history = []
        self.last_optimization = datetime.now()
        
        # Initialize optimization model
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Move model to MPS if available
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.model = self.model.to(self.device)
            self.model = torch.compile(self.model)
        else:
            self.device = torch.device("cpu")
    
    def optimize(self, current_consumption):
        """Generate energy optimization recommendations"""
        self.last_optimization = datetime.now()
        
        if current_consumption > self.threshold:
            message = f"""Energy consumption ({current_consumption:.1f} kW) exceeds threshold ({self.threshold:.1f} kW).
            Recommendations:
            1. Check for equipment inefficiencies
            2. Optimize process scheduling
            3. Consider load balancing
            """
        else:
            message = f"Energy consumption ({current_consumption:.1f} kW) within optimal range."
        
        # Record optimization attempt
        self.history.append({
            'timestamp': self.last_optimization,
            'consumption': current_consumption,
            'threshold': self.threshold
        })
        
        return message
    
    def predict_consumption(self, features):
        """Predict energy consumption based on process features"""
        with torch.no_grad():
            features = torch.tensor(features, dtype=torch.float32).to(self.device)
            prediction = self.model(features)
            return prediction.item()
    
    def get_optimization_history(self):
        """Get optimization history"""
        return pd.DataFrame(self.history)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.user_role = "Operator"
    st.session_state.selected_page = "Dashboard"
    st.session_state.dark_theme = True
    
    # Initialize compression-related state
    st.session_state.last_compressed_size = None
    st.session_state.last_compression_time = None
    st.session_state.quantum_mesh_results = None
    st.session_state.compression_history = []
    
    # Initialize digital twin components
    st.session_state.machine_connector = MachineConnector()
    st.session_state.digital_twin_viz = DigitalTwinVisualizer()
    st.session_state.camera_manager = CameraManager()
    st.session_state.sensor_processor = SensorProcessor()
    
    # Force reload the visualizer module to ensure latest changes
    importlib.reload(advanced_visualizations)
    
    # Initialize visualizer after reload
    st.session_state.visualizer = MultiModalVisualizer()
    st.session_state.current_role = "Operator"  # Default role
    st.session_state.energy_threshold = 10.0  # kW threshold for optimization
    
    # Generate process data using cached function
    start_date = '2024-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    st.session_state.process_data = generate_process_data(start_date, end_date)
    
    # Initialize Q-PIAGN model with M1 optimization
    class QPIAGN(nn.Module):
        def __init__(self):
            super(QPIAGN, self).__init__()
            self.gnn = nn.Sequential(
                nn.Linear(7, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.gnn(x)
    
    # Initialize model on CPU
    model = QPIAGN()
    st.session_state.qpiagn_model = model
    
    # Initialize other components with thread pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        # Initialize HDF5 Quantum Mesh
        quantum_mesh = HDF5QuantumMesh()
        futures.append(executor.submit(lambda: quantum_mesh))
        
        # Initialize Energy Optimization with quantum mesh
        energy_optimization = EnergyOptimization(quantum_mesh=quantum_mesh, threshold=st.session_state.energy_threshold)
        futures.append(executor.submit(lambda: energy_optimization))
        
        # Initialize AgentFactory
        agent_factory = AgentFactory()
        futures.append(executor.submit(lambda: agent_factory))
        
        # Initialize OperatorCoPilot with agent_factory and energy_optimization
        futures.append(executor.submit(lambda: OperatorCoPilot(agent_factory, energy_optimization)))
        
        # Initialize SelfCalibratingShadow
        shadow = SelfCalibratingShadow()
        futures.append(executor.submit(lambda: shadow))
        
        # Wait for all futures to complete
        for future in futures:
            result = future.result()
            if isinstance(result, HDF5QuantumMesh):
                st.session_state.quantum_mesh = result
            elif isinstance(result, EnergyOptimization):
                st.session_state.energy_optimization = result
            elif isinstance(result, AgentFactory):
                st.session_state.agent_factory = result
            elif isinstance(result, OperatorCoPilot):
                st.session_state.operator_copilot = result
            elif isinstance(result, SelfCalibratingShadow):
                st.session_state.shadow = result

# Cache expensive computations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_process_metrics():
    """Get cached process metrics"""
    return {
        "system_health": 98,
        "process_efficiency": 94,
        "quality_score": 96,
        "uptime": 99.9
    }

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_real_time_data():
    """Get cached real-time data"""
    if 'process_data' not in st.session_state:
        # Generate default process data if not available
        start_date = '2024-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        st.session_state.process_data = generate_process_data(start_date, end_date)
    
    # Return the last 50 rows or empty DataFrame if process_data is empty
    try:
        return st.session_state.process_data.tail(50)
    except Exception:
        # Return empty DataFrame with required columns if process_data is invalid
        return pd.DataFrame(columns=['timestamp', 'wear', 'temperature', 'vibration'])

# Add enhanced UI/UX styling
st.markdown("""
<style>
    /* Base Theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-green {
        background: rgba(100, 255, 100, 0.8);
        box-shadow: 0 0 10px rgba(100, 255, 100, 0.5);
    }
    
    .status-yellow {
        background: rgba(255, 255, 100, 0.8);
        box-shadow: 0 0 10px rgba(255, 255, 100, 0.5);
    }
    
    .status-red {
        background: rgba(255, 100, 100, 0.8);
        box-shadow: 0 0 10px rgba(255, 100, 100, 0.5);
    }
    
    /* Typography */
    h1, h2, h3, p {
        color: #ffffff;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(147, 51, 234, 0.1), rgba(79, 70, 229, 0.2));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        text-align: center;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Animated Tabs */
    .stTabs {
        animation: slideInRight 0.5s ease-out;
    }
    
    .stTab {
        transition: transform 0.3s ease, background-color 0.3s ease;
    }
    
    .stTab:hover {
        transform: translateY(-2px);
    }
    
    /* Enhanced Glass Cards */
    .glass-card {
        background: rgba(20, 30, 40, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100, 255, 200, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 1.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(100, 255, 200, 0.3);
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(100, 255, 200, 0.1),
            transparent
        );
        transition: 0.8s;
        z-index: 1;
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    /* Algorithm Flow Visualization */
    .algorithm-flow {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
        padding: 25px;
        background: rgba(10, 20, 30, 0.7);
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(100, 255, 200, 0.1);
        position: relative;
        overflow: hidden;
        transition: all 0.5s ease;
    }
    
    .algorithm-flow:hover {
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(100, 255, 200, 0.2);
    }
    
    .flow-step {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 18px;
        background: rgba(40, 50, 60, 0.8);
        border-radius: 10px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        flex: 1;
        min-width: 220px;
        position: relative;
        z-index: 1;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(100, 255, 200, 0.1);
        cursor: pointer;
    }
    
    .flow-step:hover {
        transform: translateY(-8px) scale(1.03);
        background: rgba(60, 70, 80, 0.9);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        border-color: rgba(100, 255, 200, 0.3);
        z-index: 2;
    }
    
    .flow-step:hover .step-number {
        background: rgba(0, 255, 150, 0.4);
        transform: scale(1.2);
        box-shadow: 0 0 15px rgba(0, 255, 150, 0.5);
    }
    
    .flow-step:hover .step-content h5 {
        color: rgba(100, 255, 200, 1);
    }
    
    .step-number {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background: rgba(0, 255, 150, 0.2);
        border-radius: 50%;
        font-weight: bold;
        color: rgba(100, 255, 200, 0.9);
        transition: all 0.4s ease;
        box-shadow: 0 0 10px rgba(0, 255, 150, 0.2);
    }
    
    .flow-arrow {
        color: rgba(100, 255, 200, 0.6);
        font-size: 24px;
        font-weight: bold;
        animation: pulseArrow 2s infinite;
        text-shadow: 0 0 10px rgba(0, 255, 150, 0.3);
    }
    
    /* Enhanced Architecture Layers */
    .architecture-layers {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .layer {
        background: rgba(30, 40, 50, 0.7);
        border-radius: 8px;
        padding: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(100, 255, 200, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .layer:hover {
        transform: translateX(10px);
        background: rgba(40, 50, 60, 0.8);
        border-color: rgba(100, 255, 200, 0.3);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
    }
    
    .layer-icon {
        font-size: 24px;
        margin-top: 10px;
        text-align: center;
    }
    
    .details-section {
        display: none;
        padding: 15px;
        background: rgba(20, 30, 40, 0.6);
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid rgba(100, 255, 200, 0.1);
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .details-section.active {
        display: block;
    }
    
    /* Enhanced Version Comparison Styles */
    .version-card {
        background: rgba(20, 30, 40, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid rgba(100, 255, 200, 0.1);
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
    }
    
    .version-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border-color: rgba(100, 255, 200, 0.3);
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        background: rgba(100, 255, 200, 0.05);
        padding-left: 5px;
    }
    
    .feature-icon {
        color: rgba(100, 255, 200, 0.8);
        font-size: 1.2rem;
    }
    
    .limitation-icon {
        color: rgba(255, 150, 100, 0.8);
        font-size: 1.2rem;
    }
    
    .metric-value {
        font-weight: bold;
        color: rgba(100, 255, 200, 0.9);
    }
    
    /* Enhanced Clock Display */
    .time-display {
        font-family: 'Courier New', monospace;
        font-size: 2rem;
        background: linear-gradient(90deg, #00ff99, #2bffd2, #00ffcc, #0fdab3);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 25px rgba(0, 255, 150, 0.6);
        animation: neonFlow 4.5s ease-in-out infinite; /* Further reduced speed */
        position: relative;
        white-space: nowrap;
        font-weight: bold;
    }
    
    .time-display::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ffc3, #09ffaa, transparent);
        background-size: 300% 100%;
        animation: scanline 4.5s linear infinite, neonFlow 4.5s ease-in-out infinite; /* Further reduced speed */
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 40, 50, 0.7);
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        border: 1px solid rgba(100, 255, 200, 0.1);
        border-bottom: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(40, 50, 60, 0.8);
        border-color: rgba(100, 255, 200, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(50, 60, 70, 0.9) !important;
        border-color: rgba(100, 255, 200, 0.5) !important;
    }
    
    /* Animations */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulseArrow {
        0% {
            opacity: 0.6;
            transform: scale(1);
        }
        50% {
            opacity: 1;
            transform: scale(1.2);
        }
        100% {
            opacity: 0.6;
            transform: scale(1);
        }
    }
    
    @keyframes neonFlow {
        0% {
            background-position: 0% 50%;
            text-shadow: 0 0 20px rgba(0, 255, 150, 0.6);
        }
        50% {
            background-position: 100% 50%;
            text-shadow: 0 0 30px rgba(0, 255, 150, 0.8);
        }
        100% {
            background-position: 0% 50%;
            text-shadow: 0 0 20px rgba(0, 255, 150, 0.6);
        }
    }
    
    @keyframes scanline {
        0% {
            background-position: 0% 50%;
            opacity: 0.5;
        }
        50% {
            background-position: 100% 50%;
            opacity: 1;
        }
        100% {
            background-position: 0% 50%;
            opacity: 0.5;
        }
    }
    
    @keyframes circuit-scan {
        0% { left: 0; }
        100% { left: 100%; }
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(30px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    /* Add animations to the radar chart */
    .plotly-graph-div .main-svg {
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Add hover effects to the optimization recommendations */
    .stMarkdownContainer .stMarkdown:hover {
        transform: scale(1.02);
        transition: transform 0.3s ease;
    }

    /* Add a glow effect to the buttons */
    .stButton>button {
        background-color: #1a1a2e;
        color: #ffffff;
        border: 1px solid #2ed573;
        box-shadow: 0 0 10px rgba(46, 213, 115, 0.5);
        transition: box-shadow 0.3s ease;
    }

    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(46, 213, 115, 0.8);
    }
    
    /* Highlight for selected page */
    div[data-testid="stSelectbox"] ul li:has(div[aria-selected="true"]) {
        background-color: rgba(72, 109, 209, 0.2) !important;
        color: #ffffff !important;
        border-left: 3px solid #486DD1 !important;
        padding-left: 10px !important;
    }
    
    /* Active button styling */
    button:active, button:focus, button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(72, 109, 209, 0.4) !important;
        color: white !important;
        border-color: #486DD1 !important;
        box-shadow: 0 0 5px rgba(72, 109, 209, 0.5) !important;
    }
    
    /* Button hover effect */
    button:hover {
        transform: translateY(-2px) !important;
        transition: transform 0.2s ease !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
</style>

<script>
    // Function to handle layer selection
    function selectLayer(layerId) {
        // Hide all detail sections
        document.querySelectorAll('.details-section').forEach(section => {
            section.classList.remove('active');
        });
        
        // Show selected section
        document.getElementById(layerId + '-details').classList.add('active');
        
        // Highlight selected layer
        document.querySelectorAll('.layer').forEach(layer => {
            layer.style.borderColor = 'rgba(100, 255, 200, 0.1)';
            layer.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.2)';
        });
        
        const selectedLayer = document.querySelector('.layer.' + layerId + '-layer');
        if (selectedLayer) {
            selectedLayer.style.borderColor = 'rgba(100, 255, 200, 0.5)';
            selectedLayer.style.boxShadow = '0 8px 30px rgba(0, 0, 0, 0.4)';
        }
    }
    
    // Add animation to algorithm flow steps
    document.addEventListener('DOMContentLoaded', function() {
        // Animate flow steps with delay
        const flowSteps = document.querySelectorAll('.flow-step');
        flowSteps.forEach((step, index) => {
            step.style.opacity = '0';
            step.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                step.style.transition = 'all 0.5s ease';
                step.style.opacity = '1';
                step.style.transform = 'translateY(0)';
            }, 100 + (index * 150));
        });
        
        // Initialize the first layer as active
        selectLayer('quantum');
        
        // Add hover effects to glass cards
        const cards = document.querySelectorAll('.glass-card');
        cards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.2}s`;
            
            // Add parallax effect on mouse move
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                const xPercent = (x / rect.width - 0.5) * 10;
                const yPercent = (y / rect.height - 0.5) * 10;
                
                card.style.transform = `
                    perspective(1000px)
                    rotateX(${yPercent}deg)
                    rotateY(${xPercent}deg)
                    translateZ(5px)
                `;
            });
            
            // Reset transform on mouse leave
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'none';
            });
        });
    });
</script>
""", unsafe_allow_html=True)

def render_process_monitoring():
    """Render the Process Monitoring page using our advanced visualizer"""
    st.header("Process Monitoring")
    
    tab1, tab2 = st.tabs(["Real-time Process", "3D Analysis"])
    
    with tab1:
        # Use our advanced visualizer for real-time monitoring
        fig = st.session_state.visualizer.render_uncertainty_visualization(
            x=st.session_state.process_data['timestamp'],
            mean=st.session_state.process_data['temperature'],
            std=np.ones_like(st.session_state.process_data['temperature']) * 0.5,
            title="Temperature Trend with Uncertainty"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current metrics
        col1, col2, col3, col4 = st.columns(4)

        # Helper function to safely get metric values and deltas
        def safe_metric_value(df, column, default="N/A"):
            if df is None or df.empty or column not in df.columns or len(df) < 1:
                return default, "0"
            
            current = df[column].iloc[-1]
            
            # For delta, we need at least 2 rows
            if len(df) < 2:
                delta = 0
            else:
                delta = current - df[column].iloc[-2]
            
            return current, delta

        with col1:
            temp_val, temp_delta = safe_metric_value(st.session_state.process_data, 'temperature')
            if temp_val != "N/A":
                st.metric("Temperature", f"{temp_val:.1f}°C", f"{temp_delta:.1f}°C")
            else:
                st.metric("Temperature", "N/A", "0")

        with col2:
            press_val, press_delta = safe_metric_value(st.session_state.process_data, 'pressure')
            if press_val != "N/A":
                st.metric("Pressure", f"{press_val:.1f} bar", f"{press_delta:.1f} bar")
            else:
                st.metric("Pressure", "N/A", "0")

        with col3:
            flow_val, flow_delta = safe_metric_value(st.session_state.process_data, 'flow_rate')
            if flow_val != "N/A":
                st.metric("Flow Rate", f"{flow_val:.1f} L/min", f"{flow_delta:.1f} L/min")
            else:
                st.metric("Flow Rate", "N/A", "0")

        with col4:
            vib_val, vib_delta = safe_metric_value(st.session_state.process_data, 'vibration')
            if vib_val != "N/A":
                st.metric("Vibration", f"{vib_val:.3f} mm/s", f"{vib_delta:.3f} mm/s")
            else:
                st.metric("Vibration", "N/A", "0")
    
    with tab2:
        # Use our 3D point cloud visualization
        points = np.column_stack((
            st.session_state.process_data['temperature'],
            st.session_state.process_data['pressure'],
            st.session_state.process_data['vibration']
        ))
        fig = st.session_state.visualizer.render_3d_point_cloud(
            points=points,
            colors=st.session_state.process_data['flow_rate'].values.reshape(-1, 1),
            title="Process Parameters in 3D Space"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_quality_control():
    """Render the Quality Control page using our advanced visualizer"""
    st.header("Quality Control")
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Quality Metrics", "Process Capability"])
    
    with tab1:
        # Use our uncertainty visualization for quality metrics
        quality_data = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=100, freq='1h'),
            'quality_score': 95 + np.cumsum(np.random.normal(0, 0.1, 100)),
            'defect_rate': 2 + np.cumsum(np.random.normal(0, 0.05, 100))
        })
        
        fig = st.session_state.visualizer.render_uncertainty_visualization(
            x=quality_data['timestamp'],
            mean=quality_data['quality_score'],
            std=np.ones_like(quality_data['quality_score']) * 0.5,
            title="Quality Score Trend with Uncertainty"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current metrics
        col1, col2 = st.columns(2)
        
        # Helper function to safely get metric values and deltas
        def safe_quality_metric(df, column, default="N/A"):
            if df is None or df.empty or column not in df.columns or len(df) < 1:
                return default, "0"
            
            current = df[column].iloc[-1]
            
            # For delta, we need at least 2 rows
            if len(df) < 2:
                delta = 0
            else:
                delta = current - df[column].iloc[-2]
            
            return current, delta
        
        with col1:
            q_score, q_delta = safe_quality_metric(quality_data, 'quality_score')
            if q_score != "N/A":
                st.metric("Quality Score", f"{q_score:.1f}%", f"{q_delta:.1f}%")
            else:
                st.metric("Quality Score", "N/A", "0%")
                
        with col2:
            d_rate, d_delta = safe_quality_metric(quality_data, 'defect_rate')
            if d_rate != "N/A":
                st.metric("Defect Rate", f"{d_rate:.2f}%", f"{d_delta:.2f}%")
            else:
                st.metric("Defect Rate", "N/A", "0%")
    
    with tab2:
        st.subheader("Process Capability Analysis")
        
        # Generate process data if not in session state
        if 'process_capability_data' not in st.session_state:
            st.session_state.process_capability_data = np.random.normal(100, 3, 1000)
        
        # Process specifications
        usl = 106  # Upper specification limit
        lsl = 94   # Lower specification limit
        
        # Calculate process capability indices
        mean = np.mean(st.session_state.process_capability_data)
        std = np.std(st.session_state.process_capability_data)
        cp = (usl - lsl) / (6 * std)
        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)
        
        # Create histogram with specification limits
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=st.session_state.process_capability_data,
            nbinsx=30,
            name='Process Data',
            marker_color='rgba(100, 255, 200, 0.6)'
        ))
        
        # Add specification limits
        fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
        fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
        fig.add_vline(x=mean, line_dash="solid", line_color="green", annotation_text="Mean")
        
        fig.update_layout(
            title='Process Capability Distribution',
            xaxis_title='Measurement',
            yaxis_title='Frequency',
            height=500,
            plot_bgcolor='rgba(30, 40, 50, 0.7)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display capability indices in glass cards
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: rgba(100, 255, 200, 0.9);">Process Capability Indices</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cp", f"{cp:.2f}", 
                     "Good" if cp >= 1.33 else "Needs Improvement",
                     delta_color="normal" if cp >= 1.33 else "off")
        with col2:
            st.metric("Cpk", f"{cpk:.2f}",
                     "Good" if cpk >= 1.33 else "Needs Improvement",
                     delta_color="normal" if cpk >= 1.33 else "off")
        with col3:
            process_yield = 100 * (1 - (abs(st.session_state.process_capability_data > usl).sum() + 
                                      abs(st.session_state.process_capability_data < lsl).sum()) / 
                                 len(st.session_state.process_capability_data))
            st.metric("Process Yield", f"{process_yield:.1f}%",
                     "Acceptable" if process_yield >= 99.73 else "Needs Improvement",
                     delta_color="normal" if process_yield >= 99.73 else "off")

def render_energy_optimization():
    """Render the Energy Optimization page"""
    st.header("Energy Optimization")
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Real-time Energy", "Optimization Insights"])
    
    with tab1:
        # Energy consumption trend
        fig = st.session_state.visualizer.render_uncertainty_visualization(
            x=st.session_state.process_data['timestamp'],
            mean=st.session_state.process_data['energy_consumption'],
            std=np.ones_like(st.session_state.process_data['energy_consumption']) * 0.2,
            title="Energy Consumption Trend"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Current energy metrics
        if (st.session_state.process_data is not None and 
            not st.session_state.process_data.empty and 
            'energy_consumption' in st.session_state.process_data.columns and 
            len(st.session_state.process_data) > 0):
            
            current_consumption = st.session_state.process_data['energy_consumption'].iloc[-1]
            optimization_message = st.session_state.energy_optimization.optimize(current_consumption)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                # Check if we have at least 2 rows for delta calculation
                if len(st.session_state.process_data) > 1:
                    delta = current_consumption - st.session_state.process_data['energy_consumption'].iloc[-2]
                    st.metric("Current Consumption", f"{current_consumption:.1f} kW", f"{delta:.1f} kW")
                else:
                    st.metric("Current Consumption", f"{current_consumption:.1f} kW", "0 kW")
            with col2:
                efficiency = 100 * (1 - (current_consumption - 8) / 4)  # Assuming 8-12 kW range
                st.metric("Energy Efficiency", f"{efficiency:.1f}%",
                         "Optimal" if efficiency > 90 else "Needs Improvement")
        else:
            st.warning("No energy consumption data available. Please ensure the process data is loaded correctly.")
    
    with tab2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: rgba(100, 255, 200, 0.9);">Energy Optimization Recommendations</h3>
            <p>{}</p>
        </div>
        """.format(optimization_message), unsafe_allow_html=True)
        
        # Energy distribution by component
        components = ['Motors', 'Heating', 'Cooling', 'Lighting', 'Other']
        values = [35, 25, 20, 15, 5]
        
        fig = go.Figure(data=[go.Pie(
            labels=components,
            values=values,
            hole=.3,
            marker=dict(colors=['rgba(100,255,200,0.6)', 'rgba(100,200,255,0.6)',
                              'rgba(255,100,200,0.6)', 'rgba(200,100,255,0.6)',
                              'rgba(255,200,100,0.6)'])
        )])
        
        fig.update_layout(
            title='Energy Distribution by Component',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_tool_wear():
    """Render the comprehensive tool wear analysis interface"""
    st.title("Tool Wear Analysis Suite")
    
    # Create tabs for different analysis modules
    tab1, tab2, tab3 = st.tabs([
        "Condition Monitoring",
        "Life Prediction",
        "Pattern Recognition"
    ])
    
    with tab1:
        from tool_condition_monitoring import render_tool_condition_monitoring
        render_tool_condition_monitoring()
    
    with tab2:
        from tool_life_prediction import render_tool_life_prediction
        render_tool_life_prediction()
    
    with tab3:
        from wear_pattern_recognition import render_wear_pattern_recognition
        render_wear_pattern_recognition()

def render_copilot():
    """Render the AI-Driven Operator CoPilot page with ML insights"""
    st.title("🤖 AI-Driven Operator CoPilot")
    
    # Create two columns for metrics and recommendations
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Current Process Metrics")
        # Sample metrics (replace with real-time data in production)
        metrics = {
            'temperature': st.number_input("Temperature (°C)", value=72.5, step=0.1),
            'pressure': st.number_input("Pressure (PSI)", value=98.6, step=0.1),
            'vibration': st.number_input("Vibration (Hz)", value=120.3, step=0.1),
            'quality_score': st.number_input("Quality Score", value=95.8, step=0.1)
        }
        
        # Display current state visualization
        st.subheader("System Health Indicators")
        health_metrics = {
            "Temperature": metrics['temperature'] / 100,
            "Pressure": metrics['pressure'] / 100,
            "Vibration": metrics['vibration'] / 150,
            "Quality": metrics['quality_score'] / 100
        }
        
        # Create health indicator bars
        for metric, value in health_metrics.items():
            color = "green" if 0.4 <= value <= 0.8 else "red"
            st.progress(value, text=f"{metric}: {value*100:.1f}%")
    
    with col2:
        st.subheader("AI Recommendations")
        recommendations = st.session_state.operator_copilot.get_recommendation(metrics)
        
        # Group recommendations by type
        alerts = [r for r in recommendations if "⚠️" in r]
        maintenance = [r for r in recommendations if "🔧" in r]
        predictions = [r for r in recommendations if "📊" in r]
        normal = [r for r in recommendations if "✅" in r]
        
        # Display recommendations in expandable sections
        if alerts:
            with st.expander("🚨 Critical Alerts", expanded=True):
                for alert in alerts:
                    st.error(alert)
        
        if maintenance:
            with st.expander("🔧 Maintenance Recommendations", expanded=True):
                for rec in maintenance:
                    st.warning(rec)
        
        if predictions:
            with st.expander("📊 Process Predictions", expanded=True):
                for pred in predictions:
                    st.info(pred)
        
        if normal:
            with st.expander("✅ Status Update", expanded=True):
                for status in normal:
                    st.success(status)
    
    # Historical trends
    st.subheader("Historical Performance")
    tab1, tab2 = st.tabs(["Process Metrics", "Quality Trends"])
    
    with tab1:
        # Create sample historical data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='h')
        historical_data = pd.DataFrame({
            'Temperature': np.random.normal(72.5, 2, 24),
            'Pressure': np.random.normal(98.6, 3, 24),
            'Vibration': np.random.normal(120.3, 5, 24)
        }, index=dates)
        
        fig = go.Figure()
        for col in historical_data.columns:
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data[col],
                name=col,
                mode='lines+markers'
            ))
        fig.update_layout(
            title="24-Hour Process Metrics History",
            xaxis_title="Time",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create sample quality trend data
        quality_data = pd.DataFrame({
            'Quality Score': np.random.normal(95.8, 1.5, 24),
            'Target': [95] * 24,
            'Upper Limit': [98] * 24,
            'Lower Limit': [92] * 24
        }, index=dates)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=quality_data.index,
            y=quality_data['Quality Score'],
            name='Quality Score',
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=quality_data.index,
            y=quality_data['Target'],
            name='Target',
            line=dict(dash='dash', color='green')
        ))
        fig.add_trace(go.Scatter(
            x=quality_data.index,
            y=quality_data['Upper Limit'],
            name='Upper Limit',
            line=dict(dash='dot', color='red')
        ))
        fig.add_trace(go.Scatter(
            x=quality_data.index,
            y=quality_data['Lower Limit'],
            name='Lower Limit',
            line=dict(dash='dot', color='red')
        ))
        fig.update_layout(
            title="24-Hour Quality Score Trend",
            xaxis_title="Time",
            yaxis_title="Quality Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def render_digital_twin_simulation():
    """Render the Digital Twin Simulation page with real-time 3D visualization and ML capabilities"""
    st.title("🔄 Digital Twin Simulation")
    
    # Create main columns for control panel and visualization
    control_col, viz_col = st.columns([1, 2])
    
    with control_col:
        st.markdown("""
        <div class="glass-card">
            <h3>Machine Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Machine identification and connection
        machine_id = st.text_input("Machine ID/Serial Number", placeholder="Enter machine identifier")
        api_endpoint = st.text_input("API Endpoint", placeholder="Enter machine API endpoint")
        
        if machine_id and api_endpoint:
            if st.button("Connect to Machine"):
                with st.spinner("Connecting to machine..."):
                    # Connect to machine API using a synchronous approach
                    try:
                        # Create a new event loop for this operation
                        connect_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(connect_loop)
                        connect_loop.run_until_complete(
                            st.session_state.machine_connector.connect_to_machine(
                                machine_id, api_endpoint
                            )
                        )
                        connect_loop.close()
                    except Exception as e:
                        st.error(f"Failed to connect to machine: {str(e)}")
            
            # Get machine specifications
            machine_specs = st.session_state.machine_connector.machine_specs.get(machine_id)
            if machine_specs:
                st.markdown("""
                <div class="glass-card">
                    <h4>Machine Specifications</h4>
                    <ul>
                        <li>Type: {type}</li>
                        <li>Model: {model}</li>
                        <li>Axes: {axes}</li>
                        <li>Max Speed: {max_speed} RPM</li>
                        <li>Workspace: {workspace[0]}x{workspace[1]}x{workspace[2]} mm</li>
                    </ul>
                </div>
                """.format(**machine_specs.__dict__), unsafe_allow_html=True)
                
                # Simulation controls
                st.subheader("Simulation Controls")
                
                # Operation mode selection
                mode = st.selectbox("Operation Mode", 
                                  ["Training", "Parallel Operation", "Autonomous Operation"])
                
                # Task configuration
                if mode == "Training":
                    st.markdown("""
                    <div class="glass-card">
                        <h4>Training Configuration</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    task_type = st.selectbox("Task Type", 
                                           ["Surface Milling", "Pocket Milling", "Contour Milling"])
                    material = st.selectbox("Material", 
                                          ["Aluminum", "Steel", "Titanium", "Plastic"])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        feed_rate = st.slider("Feed Rate (%)", 0, 100, 50)
                        spindle_speed = st.slider("Spindle Speed (%)", 0, 100, 60)
                    
                    with col2:
                        depth_of_cut = st.slider("Depth of Cut (mm)", 0.1, 5.0, 1.0)
                        step_over = st.slider("Step Over (%)", 10, 100, 40)
                
                # Real-time sensor data
                st.subheader("Sensor Readings")
                
                # Get latest sensor data
                sensor_data = st.session_state.machine_connector.get_latest_data(machine_id)
                if sensor_data:
                    # Process sensor data
                    processed_data = st.session_state.sensor_processor.process_sensor_data(
                        machine_id, sensor_data
                    )
                    
                    # Display sensor metrics
                    sensor_cols = st.columns(2)
                    with sensor_cols[0]:
                        st.metric("Temperature", f"{sensor_data.get('temperature', 0):.1f} °C",
                                f"{processed_data['metrics'].get('temperature_trend', 0):.2f} °C/min")
                        st.metric("Vibration", f"{sensor_data.get('vibration', 0):.3f} mm/s",
                                f"RMS: {processed_data['metrics'].get('vibration_rms', 0):.3f}")
                    with sensor_cols[1]:
                        st.metric("Power Load", f"{sensor_data.get('power_output', 0):.1f}%",
                                f"Efficiency: {processed_data['metrics'].get('power_efficiency', 0)*100:.1f}%")
                        st.metric("Accuracy", f"±{sensor_data.get('accuracy', 0):.3f} mm",
                                "Within tolerance" if sensor_data.get('accuracy', 1) < 0.01 else "Check calibration")
                    
                    # Display alerts if any
                    if processed_data['alerts']:
                        st.warning("⚠️ Alerts detected:")
                        for alert in processed_data['alerts']:
                            st.error(f"{alert['sensor']}: {alert['value']} exceeds threshold of {alert['threshold']}")
    
    with viz_col:
        # Create tabs for different visualization modes
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["3D Simulation", "Live Camera", "Performance"])
        
        with viz_tab1:
            st.markdown("""
            <div class="glass-card" style="height: 600px;">
                <h3>3D Digital Twin Visualization</h3>
                <div id="digital-twin-3d" style="height: 500px;">
                    <!-- 3D visualization will be rendered here -->
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if machine_id in st.session_state.machine_connector.machine_specs:
                # Initialize 3D visualization if not already done
                if not st.session_state.digital_twin_viz.scene:
                    st.session_state.digital_twin_viz.initialize_3d_scene("digital-twin-3d")
                    # Load machine model
                    model_path = f"models/{machine_specs.type.lower().replace(' ', '_')}.glb"
                    st.session_state.digital_twin_viz.load_machine_model(machine_id, model_path)
                
                # Update machine state in visualization
                if sensor_data:
                    st.session_state.digital_twin_viz.update_machine_state(machine_id, sensor_data)
                    st.session_state.digital_twin_viz.render()
            
            # Simulation controls
            col1, col2, col3 = st.columns(3)
            with col1:
                st.button("Start Simulation", type="primary")
            with col2:
                st.button("Pause")
            with col3:
                st.button("Reset")
        
        with viz_tab2:
            st.markdown("""
            <div class="glass-card" style="height: 600px;">
                <h3>Live Camera Feed</h3>
                <div id="camera-feed" style="height: 500px;">
                    <!-- Camera feed will be rendered here -->
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Camera controls
            camera_view = st.selectbox("Camera View", ["Front", "Top", "Side", "Isometric"])
            
            if machine_id in st.session_state.machine_connector.machine_specs:
                # Connect to camera if not already connected
                camera_id = f"{machine_id}_{camera_view.lower()}"
                if camera_id not in st.session_state.camera_manager.camera_streams:
                    camera_url = machine_specs.sensor_endpoints.get(f"camera_{camera_view.lower()}")
                    if camera_url:
                        st.session_state.camera_manager.connect_camera(camera_id, camera_url)
                
                # Display camera feed
                frame = st.session_state.camera_manager.get_latest_frame(camera_id)
                if frame is not None:
                    st.image(frame, channels="BGR", use_container_width=True)
        
        with viz_tab3:
            st.subheader("Real-time Performance Analysis")
            
            if machine_id in st.session_state.machine_connector.machine_specs:
                # Get historical sensor data
                temp_history = st.session_state.sensor_processor.get_sensor_history(
                    machine_id, 'temperature', '1h'
                )
                efficiency_history = st.session_state.sensor_processor.get_sensor_history(
                    machine_id, 'power_efficiency', '1h'
                )
                accuracy_history = st.session_state.sensor_processor.get_sensor_history(
                    machine_id, 'accuracy', '1h'
                )
                
                # Create performance visualization
                fig = go.Figure()
                
                # Add traces for each metric
                fig.add_trace(go.Scatter(
                    x=temp_history.index,
                    y=temp_history['temperature'],
                    name='Temperature',
                    mode='lines',
                    line=dict(width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=efficiency_history.index,
                    y=efficiency_history['power_efficiency'],
                    name='Efficiency',
                    mode='lines',
                    line=dict(width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=accuracy_history.index,
                    y=accuracy_history['accuracy'],
                    name='Accuracy',
                    mode='lines',
                    line=dict(width=2)
                ))
                
                fig.update_layout(
                    title="Performance Metrics",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Bottom section for recommendations and fine-tuning
    st.markdown("""
    <div class="glass-card">
        <h3>AI Recommendations & Fine-tuning</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if machine_id in st.session_state.machine_connector.machine_specs:
        col1, col2 = st.columns(2)
        
        with col1:
            # Get recommendations based on sensor data
            if sensor_data:
                recommendations = []
                if sensor_data.get('temperature', 0) > 50:
                    recommendations.append("🔄 Reduce spindle speed to lower temperature")
                if sensor_data.get('vibration', 0) > 0.2:
                    recommendations.append("⚡ Adjust feed rate to minimize vibration")
                if sensor_data.get('accuracy', 0) > 0.01:
                    recommendations.append("🎯 Recalibrate for improved accuracy")
                if sensor_data.get('power_efficiency', 0) < 0.8:
                    recommendations.append("⚡ Optimize power usage patterns")
                
                st.markdown("""
                <div class="glass-card">
                    <h4>Process Recommendations</h4>
                    <ul>
                        {}
                    </ul>
                </div>
                """.format(''.join([f"<li>{r}</li>" for r in recommendations])), 
                unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4>Fine-tuning Parameters</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Fine-tuning controls with real-time updates
            tool_path = st.slider("Tool Path Optimization", 0, 100, 80)
            surface_finish = st.slider("Surface Finish Priority", 0, 100, 70)
            energy_efficiency = st.slider("Energy Efficiency", 0, 100, 85)
            cycle_time = st.slider("Cycle Time Optimization", 0, 100, 75)
            
            # Update machine parameters if connected
            if st.button("Apply Parameters"):
                with st.spinner("Updating machine parameters..."):
                    # Simulate parameter update
                    time.sleep(1)
                    st.success("Parameters updated successfully")

def render_factory_dashboard():
    """Render the main factory dashboard with modern design."""
    # Initialize factory components if not in session state
    if 'factory_components' not in st.session_state:
        st.session_state.factory_components = FactoryComponents()
        
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #ffffff;
        }
        
        /* Quick Actions styling */
        .quick-actions-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 20px 0 15px 0;
            color: white;
        }
        
        /* Custom styling for the buttons */
        button[kind="secondary"] {
            background-color: rgba(30, 33, 56, 0.8) !important;
            color: white !important;
            border: 1px solid rgba(100, 100, 200, 0.3) !important;
            border-radius: 10px !important;
            padding: 15px 25px !important;
            font-size: 18px !important;
            font-weight: 500 !important;
            min-height: 60px !important; /* Reduced height */
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
            width: 100% !important; /* Ensure full width */
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin-bottom: 5px !important;
            letter-spacing: 0.5px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Import the live metrics module
    from live_metrics import render_live_metrics
    
    # Create a two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Render the live metrics dashboard
        render_live_metrics()
    
    with col2:
        # Quick Actions Section
        st.markdown('<h3 class="quick-actions-title">Quick Actions</h3>', unsafe_allow_html=True)
        
        if st.button("🚀 Optimize System", use_container_width=True):
            with st.spinner("Optimizing system parameters..."):
                time.sleep(2)
                st.success("System optimized! Efficiency increased by 3.2%")
                # Set the clicked state
                st.session_state.optimize_clicked = True
                
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Refreshing data from all sources..."):
                time.sleep(1.5)
                st.success("Data refreshed successfully!")
                # Set the clicked state
                st.session_state.refresh_clicked = True
                
        if st.button("📊 Generate Report", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                time.sleep(2.5)
                st.success("Report generated and sent to stakeholders!")
                # Set the clicked state
                st.session_state.report_clicked = True
                
        if st.button("⚙️ System Settings", use_container_width=True):
            with st.spinner("Loading system settings..."):
                time.sleep(1)
                # Trigger navigation to settings
                st.session_state.settings_clicked = True
                st.session_state.current_page = "Settings"
                st.rerun()
        
        # Notifications Section
        st.markdown("""
        <div style="background: rgba(25, 30, 50, 0.6); border-radius: 10px; padding: 15px; margin-top: 20px; border: 1px solid rgba(100, 120, 200, 0.2);">
            <h3 style="font-size: 1.2rem; color: rgba(200, 220, 255, 0.9); margin-bottom: 10px;">Recent Notifications</h3>
            <div style="background: rgba(40, 45, 70, 0.7); border-radius: 8px; padding: 10px; margin-bottom: 8px; border-left: 3px solid #4CAF50;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: 500; color: rgba(200, 220, 255, 0.9);">System Update Complete</span>
                    <span style="font-size: 0.8rem; color: rgba(180, 190, 220, 0.7);">10 min ago</span>
                </div>
                <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: rgba(180, 190, 220, 0.8);">All systems updated to latest version.</p>
            </div>
            <div style="background: rgba(40, 45, 70, 0.7); border-radius: 8px; padding: 10px; margin-bottom: 8px; border-left: 3px solid #FFC107;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: 500; color: rgba(200, 220, 255, 0.9);">Maintenance Scheduled</span>
                    <span style="font-size: 0.8rem; color: rgba(180, 190, 220, 0.7);">1 hour ago</span>
                </div>
                <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: rgba(180, 190, 220, 0.8);">Planned maintenance on Mar 15, 2025.</p>
            </div>
            <div style="background: rgba(40, 45, 70, 0.7); border-radius: 8px; padding: 10px; border-left: 3px solid #2196F3;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: 500; color: rgba(200, 220, 255, 0.9);">New Feature Available</span>
                    <span style="font-size: 0.8rem; color: rgba(180, 190, 220, 0.7);">3 hours ago</span>
                </div>
                <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: rgba(180, 190, 220, 0.8);">YouTube streaming is now available.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # The factory components dashboard is handled by our component
    st.session_state.factory_components.render_factory_dashboard()

def render_provider_management():
    """Render the Provider Management page with modern floating card design."""
    st.title("Provider Management")
    
    # Initialize factory components if not in session state
    if 'factory_components' not in st.session_state:
        st.session_state.factory_components = FactoryComponents()
    
    # Create tabs for different provider management sections
    tab1, tab2, tab3 = st.tabs(["Active Providers", "Performance Analytics", "Configuration"])
    
    with tab1:
        st.markdown("## Active Providers")
        
        # Get provider information from factory components
        providers = {
            'HyperSyncDT_Quantum_Core': {
                'status': 'Active',
                'description': 'Advanced quantum-inspired processing engine for complex manufacturing optimization',
                'latency': 15,
                'uptime': 99.9,
                'load': 45,
                'success_rate': 98.5
            },
            'HyperSyncDT_Neural_Fabric': {
                'status': 'Active',
                'description': 'Neural network infrastructure for real-time process adaptation and learning',
                'latency': 12,
                'uptime': 99.8,
                'load': 60,
                'success_rate': 99.1
            },
            'HyperSyncDT_Cognitive_Engine': {
                'status': 'Active',
                'description': 'Advanced reasoning and decision-making system for autonomous operations',
                'latency': 18,
                'uptime': 99.7,
                'load': 55,
                'success_rate': 97.8
            }
        }
        
        # Render provider cards in a grid
        cols = st.columns(3)
        for idx, (provider_name, provider_info) in enumerate(providers.items()):
            with cols[idx % 3]:
                st.session_state.factory_components.render_provider_card(provider_name, provider_info)
    
    with tab2:
        st.markdown("## Performance Analytics")
        
        # Create performance comparison chart
        performance_data = {
            'Provider': list(providers.keys()),
            'Latency': [info['latency'] for info in providers.values()],
            'Uptime': [info['uptime'] for info in providers.values()],
            'Load': [info['load'] for info in providers.values()],
            'Success Rate': [info['success_rate'] for info in providers.values()]
        }
        df = pd.DataFrame(performance_data)
        
        fig = go.Figure()
        metrics = ['Latency', 'Uptime', 'Load', 'Success Rate']
        colors = ['#ff6b6b', '#2ed573', '#1e90ff', '#ffd93d']
        
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Bar(
                name=metric,
                x=df['Provider'],
                y=df[metric],
                marker_color=color,
                hovertemplate=f"{metric}: %{{y}}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Provider Performance Comparison",
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor='rgba(255,255,255,0.18)',
                borderwidth=1
            ),
            margin=dict(t=30, l=0, r=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## Provider Configuration")
        
        # Configuration form with modern styling
        st.markdown("""
            <style>
            .config-form {
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        with st.form("provider_config"):
            st.markdown("<div class='config-form'>", unsafe_allow_html=True)
            
            # Provider selection
            provider = st.selectbox("Select Provider", list(providers.keys()))
            
            col1, col2 = st.columns(2)
            with col1:
                api_key = st.text_input("API Key", type="password")
                max_load = st.slider("Max Load (%)", 0, 100, 80)
            
            with col2:
                timeout = st.number_input("Timeout (ms)", 1000, 10000, 5000)
                retries = st.number_input("Max Retries", 1, 10, 3)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.form_submit_button("Update Configuration"):
                st.success(f"Configuration updated for {provider}")

def render_system_monitoring():
    """Render the System Monitoring page with comprehensive monitoring capabilities."""
    st.title("System Monitoring")
    
    # Create tabs for different monitoring sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "System Health",
        "Resource Usage",
        "Network & Services",
        "Alerts & Logs"
    ])
    
    with tab1:
        st.subheader("System Health Overview")
        
        # System health metrics in sorted order
        metrics = {
            "CPU Usage": {"value": 42, "delta": 1.2, "suffix": "%"},
            "Memory Usage": {"value": 68, "delta": -5, "suffix": "%"},
            "Network Load": {"value": 45, "delta": -3, "suffix": "%"},
            "Storage": {"value": 76, "delta": 2, "suffix": "%"}
        }
        
        # Display metrics in a grid
        cols = st.columns(4)
        for (metric, data), col in zip(sorted(metrics.items()), cols):
            with col:
                st.metric(
                    metric,
                    f"{data['value']}{data['suffix']}",
                    f"{data['delta']}{data['suffix']}"
                )
        
        # System components status
        st.markdown("""
        <div class="monitoring-grid">
            <div class="monitoring-card">
                <div class="monitoring-header">
                    <span>Core Services</span>
                    <div class="status-indicator">
                        <div class="status-dot healthy"></div>
                        <span>Healthy</span>
                    </div>
                </div>
                <div class="component-list">
                    • API Gateway: Active<br>
                    • Authentication Service: Active<br>
                    • Database Cluster: Active<br>
                    • Message Queue: Active
                </div>
            </div>
            
            <div class="monitoring-card">
                <div class="monitoring-header">
                    <span>ML Services</span>
                    <div class="status-indicator">
                        <div class="status-dot healthy"></div>
                        <span>Optimal</span>
                    </div>
                </div>
                <div class="component-list">
                    • Feature Store: Synced<br>
                    • Inference Engine: Ready<br>
                    • Model Server: Running<br>
                    • Training Pipeline: Active
                </div>
            </div>
            
            <div class="monitoring-card">
                <div class="monitoring-header">
                    <span>Infrastructure</span>
                    <div class="status-indicator">
                        <div class="status-dot warning"></div>
                        <span>Warning</span>
                    </div>
                </div>
                <div class="component-list">
                    • Backup System: Active<br>
                    • Cache Layer: Warning<br>
                    • Load Balancer: Active<br>
                    • Storage Cluster: Active
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Resource Usage Analytics")
        
        # Resource usage charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # CPU Usage over time
            cpu_data = np.random.normal(60, 10, 100)
            fig_cpu = go.Figure()
            fig_cpu.add_trace(go.Scatter(
                y=cpu_data,
                mode='lines',
                name='CPU Usage',
                line=dict(color='rgba(100, 255, 200, 0.8)')
            ))
            fig_cpu.update_layout(
                title="CPU Usage Trend",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with chart_col2:
            # Memory Usage over time
            memory_data = np.random.normal(75, 8, 100)
            fig_memory = go.Figure()
            fig_memory.add_trace(go.Scatter(
                y=memory_data,
                mode='lines',
                name='Memory Usage',
                line=dict(color='rgba(100, 200, 255, 0.8)')
            ))
            fig_memory.update_layout(
                title="Memory Usage Trend",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig_memory, use_container_width=True)
    
    with tab3:
        st.subheader("Network & Services Status")
        
        # Network metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="monitoring-card">
                <div class="monitoring-header">
                    <span>Network Metrics</span>
                </div>
                • Active Connections: 1,250<br>
                • Latency: 12ms<br>
                • Packet Loss: 0.01%<br>
                • Throughput: 1.2 GB/s
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="monitoring-card">
                <div class="monitoring-header">
                    <span>Service Health</span>
                </div>
                • API Response Time: 45ms<br>
                • Cache Hit Rate: 94%<br>
                • Database Queries/s: 3,500<br>
                • Error Rate: 0.05%
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("System Alerts & Logs")
        
        # Alert severity filter
        severity = st.multiselect(
            "Filter by Severity",
            ["Critical", "Warning", "Info"],
            default=["Critical", "Warning"]
        )
        
        # Sample alerts sorted by severity and time
        alerts = [
            {"severity": "Critical", "message": "High memory usage detected", "time": "1 hour ago"},
            {"severity": "Warning", "message": "Cache layer performance degraded", "time": "2 mins ago"},
            {"severity": "Info", "message": "Backup completed successfully", "time": "15 mins ago"}
        ]
        
        # Sort alerts by severity (Critical > Warning > Info) and then by time
        severity_order = {"Critical": 0, "Warning": 1, "Info": 2}
        sorted_alerts = sorted(alerts, key=lambda x: (severity_order[x["severity"]], x["time"]))
        
        for alert in sorted_alerts:
            if alert["severity"] in severity:
                st.markdown(f"""
                <div class="monitoring-card">
                    <div class="monitoring-header">
                        <span>{alert['message']}</span>
                        <div class="status-indicator">
                            <div class="status-dot {alert['severity'].lower()}"></div>
                            <span>{alert['severity']}</span>
                        </div>
                    </div>
                    <div style="color: rgba(255,255,255,0.6);">{alert['time']}</div>
                </div>
                """, unsafe_allow_html=True)

def render_settings():
    """Render the Settings page with comprehensive configuration options."""
    st.title("System Settings")
    
    # Create settings tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "General", "Security", "Integration", "Advanced"
    ])
    
    with tab1:
        st.subheader("General Settings")
        
        # System preferences
        st.markdown("### System Preferences")
        theme = st.selectbox("Theme", ["Dark", "Light", "System Default"])
        language = st.selectbox("Language", ["English", "Spanish", "German", "French", "Chinese"])
        timezone = st.selectbox("Timezone", ["UTC", "UTC+1", "UTC+2", "UTC+3", "UTC-5"])
        
        # Notification settings
        st.markdown("### Notification Settings")
        email_notifications = st.checkbox("Email Notifications", value=True)
        slack_notifications = st.checkbox("Slack Notifications")
        notification_frequency = st.select_slider(
            "Notification Frequency",
            options=["Real-time", "Hourly", "Daily", "Weekly"]
        )
    
    with tab2:
        st.subheader("Security Settings")
        
        # Authentication
        st.markdown("### Authentication")
        enable_2fa = st.checkbox("Enable Two-Factor Authentication", value=True)
        session_timeout = st.number_input("Session Timeout (minutes)", min_value=5, max_value=120, value=30)
        
        # Access Control
        st.markdown("### Access Control")
        ip_whitelist = st.text_area("IP Whitelist (one per line)")
        max_login_attempts = st.number_input("Max Login Attempts", min_value=3, max_value=10, value=5)
        
        # API Security
        st.markdown("### API Security")
        api_rate_limit = st.number_input("API Rate Limit (requests/minute)", min_value=60, max_value=1000, value=120)
        enable_api_logging = st.checkbox("Enable API Logging", value=True)
    
    with tab3:
        st.subheader("Integration Settings")
        
        # Cloud Integration
        st.markdown("### Cloud Services")
        cloud_provider = st.selectbox("Cloud Provider", ["AWS", "Azure", "Google Cloud"])
        region = st.selectbox("Region", ["US East", "US West", "EU West", "Asia Pacific"])
        
        # Database Configuration
        st.markdown("### Database Configuration")
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MongoDB", "MySQL"])
        connection_pool = st.number_input("Connection Pool Size", min_value=5, max_value=100, value=20)
        
        # External Services
        st.markdown("### External Services")
        enable_monitoring = st.checkbox("Enable External Monitoring", value=True)
        monitoring_service = st.selectbox("Monitoring Service", ["Datadog", "New Relic", "Grafana Cloud"])
    
    with tab4:
        st.subheader("Advanced Settings")
        
        # Performance Tuning
        st.markdown("### Performance Tuning")
        cache_size = st.slider("Cache Size (GB)", min_value=1, max_value=32, value=8)
        worker_threads = st.number_input("Worker Threads", min_value=1, max_value=32, value=4)
        
        # Debug Options
        st.markdown("### Debug Options")
        debug_mode = st.checkbox("Enable Debug Mode")
        log_level = st.selectbox("Log Level", ["ERROR", "WARNING", "INFO", "DEBUG"])
        
        # Maintenance
        st.markdown("### Maintenance")
        backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
        retention_period = st.number_input("Log Retention Period (days)", min_value=7, max_value=365, value=30)
    
    # Save settings button
    if st.button("Save Settings", type="primary"):
        st.success("Settings updated successfully!")
        st.info("Some changes may require a system restart to take effect.")

def main():
    # Custom CSS for modern sidebar styling
    st.markdown("""
        <style>
        .sidebar-role-selector {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .sidebar-nav {
            margin-top: 20px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 10px;
        }
        
        .nav-section {
            margin-top: 15px;
            padding: 10px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.02);
        }
        
        .nav-section-title {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        
        .nav-subsection {
            margin-left: 10px;
            padding: 5px;
            border-left: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Highlight for selected page */
        div[data-testid="stSelectbox"] ul li:has(div[aria-selected="true"]) {
            background-color: rgba(72, 109, 209, 0.2) !important;
            color: #ffffff !important;
            border-left: 3px solid #486DD1 !important;
            padding-left: 10px !important;
        }
        
        /* Active button styling */
        button:active, button:focus, button[data-baseweb="tab"][aria-selected="true"] {
            background-color: rgba(72, 109, 209, 0.4) !important;
            color: white !important;
            border-color: #486DD1 !important;
            box-shadow: 0 0 5px rgba(72, 109, 209, 0.5) !important;
        }
        
        /* Button hover effect */
        button:hover {
            transform: translateY(-2px) !important;
            transition: transform 0.2s ease !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar header with logo
    st.sidebar.title("🔄 HyperSyncDT")
    
    # Role selector
    st.sidebar.markdown('<div class="sidebar-role-selector">', unsafe_allow_html=True)
    selected_role = st.sidebar.selectbox(
        "Select Role",
        ["Operator", "Data Scientist", "COO"],
        key="role_selector"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state variables for Quick Action buttons
    if "optimize_clicked" not in st.session_state:
        st.session_state.optimize_clicked = False
        
    if "refresh_clicked" not in st.session_state:
        st.session_state.refresh_clicked = False
        
    if "report_clicked" not in st.session_state:
        st.session_state.report_clicked = False
        
    if "settings_clicked" not in st.session_state:
        st.session_state.settings_clicked = False
    
    # Remove the persistent live dashboard
    # dashboard_container = st.container()
    # with dashboard_container:
    #     st.markdown("""
    #     <style>
    #     .dashboard-border {
    #         border: 1px solid rgba(255, 255, 255, 0.1);
    #         border-radius: 10px;
    #         padding: 10px;
    #         margin-bottom: 20px;
    #         background-color: rgba(0, 0, 0, 0.2);
    #     }
    #     </style>
    #     <div class="dashboard-border">
    #     """, unsafe_allow_html=True)
    #     st.markdown("</div>", unsafe_allow_html=True)
    
    # Define pages based on role
    common_pages = {
        "Factory Dashboard": render_factory_dashboard,
        "Provider Management": render_provider_management,
        "System Monitoring": render_system_monitoring,
        "Settings": render_settings,
        "AI Assistant": render_rag_assistant,
        "RAG Agent Creator": render_rag_agent_creator
    }
    
    factory_pages = {
        "Factory Connect": render_factory_connect,
        "Factory Build": render_factory_build,
        "Factory Analyze": render_factory_analyze,
        "Factory Operate": render_factory_operate
    }
    
    tool_wear_pages = {
        "Tool Wear Analysis": render_tool_wear_analysis,
        "Tool Condition Monitoring": render_tool_condition_monitoring,
        "Tool Life Prediction": render_tool_life_prediction,
        "Wear Pattern Recognition": render_wear_pattern_recognition
    }
    
    simulation_pages = {
        "Digital Twin": render_digital_twin_simulation,
        "Process Simulation": render_process_simulation,
        "What-If Analysis": render_what_if_analysis,
        "Virtual Testing": render_virtual_testing
    }
    
    operator_pages = {
        **factory_pages,
        **tool_wear_pages,
        **simulation_pages,
        "Process Monitoring": render_process_monitoring,
        "Quality Control": render_quality_control,
        "Energy Optimization": render_energy_optimization,
        "Operator CoPilot": render_copilot
    }
    
    data_scientist_pages = {
        "EDA Workspace": render_eda_workspace,
        "Advanced Visualizations": render_advanced_visualization_page,
        "Research Roadmap": render_research_roadmap,
        "Model Performance": render_model_performance,
        "Experiment Tracking": render_experiment_tracking,
        **tool_wear_pages,
        **simulation_pages,
    }
    
    coo_pages = {
        "Performance Analytics": lambda: st.info("Performance Analytics page is under development"),
        "Resource Planning": lambda: st.info("Resource Planning page is under development"),
        "Risk Management": render_risk_mitigation,
        "Sustainability": render_sustainability,
        **factory_pages
    }
    
    # Combine pages based on role
    if selected_role == "Operator":
        available_pages = {**common_pages, **operator_pages}
    elif selected_role == "Data Scientist":
        available_pages = {**common_pages, **data_scientist_pages}
    else:  # COO
        available_pages = {**common_pages, **coo_pages}
    
    # Create page categories for better organization
    page_categories = {
        "System Overview": ["Factory Dashboard", "Provider Management", "System Monitoring", "Settings"],
        "AI Assistance": ["AI Assistant", "RAG Agent Creator"],
        "Factory Operations": ["Factory Connect", "Factory Build", "Factory Analyze", "Factory Operate"],
        "Tool Management": ["Tool Wear Analysis", "Tool Condition Monitoring", "Tool Life Prediction", "Wear Pattern Recognition"],
        "Simulations": ["Digital Twin", "Process Simulation", "What-If Analysis", "Virtual Testing"],
        "Monitoring": ["Process Monitoring", "Quality Control", "Energy Optimization"],
        "Analysis": ["EDA Workspace", "Advanced Visualizations", "Research Roadmap"],
        "AI/ML": ["Model Performance", "Experiment Tracking", "Operator CoPilot"],
        "Management": ["Performance Analytics", "Resource Planning", "Risk Management", "Sustainability"]
    }
    
    # Navigation sections with categories
    st.sidebar.markdown('<div class="nav-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="nav-section-title">Navigate To</div>', unsafe_allow_html=True)
    
    # Create a dictionary of available pages by category
    categorized_pages = {}
    for category, pages in page_categories.items():
        category_pages = {page: available_pages[page] for page in pages if page in available_pages}
        if category_pages:
            categorized_pages[category] = category_pages
    
    # Create a two-level selectbox for navigation
    selected_category = st.sidebar.selectbox("Category", list(categorized_pages.keys()))
    page = st.sidebar.selectbox("Page", list(categorized_pages[selected_category].keys()))
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions section
    st.sidebar.markdown('<div class="nav-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="nav-section-title">Quick Actions</div>', unsafe_allow_html=True)
    if st.sidebar.button("⚡ New Analysis"):
        st.session_state.new_analysis = True
    if st.sidebar.button("📊 Generate Report"):
        st.session_state.generate_report = True
    if st.sidebar.button("🔄 Sync Data"):
        st.session_state.sync_data = True
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Status indicator
    st.sidebar.markdown('<div class="nav-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="nav-section-title">System Status</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div style="color: #2ed573">●</div> Connected as {selected_role}',
        unsafe_allow_html=True
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Add backup location information
    st.sidebar.markdown('<div class="nav-section-title">Backup Information</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        📂 **Backup Location**:  
        `~/Desktop/hyper-synced-dt-mvp-backup`
        
        Last backup: {}
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M")),
        unsafe_allow_html=True
    )
    
    # Render the selected page
    with st.spinner(f"Loading {page}..."):
        categorized_pages[selected_category][page]()

    # Add interactive tool for parameter adjustments
    st.sidebar.markdown("### Interactive Tools")
    
    # Create columns for the interactive tools
    tool_col1, tool_col2 = st.sidebar.columns(2)
    
    # Optimize Parameters Button with enhanced functionality
    with tool_col1:
        if st.button("Optimize Parameters", key="optimize_params", use_container_width=True):
            with st.sidebar:
                # Create a container for progress updates
                optimization_status = st.empty()
                optimization_status.info("Starting parameter optimization...")
                
                # Create progress bar
                optimization_progress = st.progress(0)
                
                # Optimization phases
                phases = ["Analyzing current setup", "Running simulations", 
                          "Applying genetic algorithm", "Fine-tuning parameters", 
                          "Validating results"]
                
                # Initialize container for optimization results
                optimization_result = st.empty()
                
                # Simulate optimization with detailed progress
                for i, phase in enumerate(phases):
                    # Calculate progress percentage
                    progress_pct = int((i / len(phases)) * 100)
                    
                    # Update progress
                    optimization_progress.progress(progress_pct)
                    optimization_status.info(f"Phase {i+1}/{len(phases)}: {phase}")
                    
                    # Simulate processing time
                    time.sleep(0.5)
                
                # Complete the optimization
                optimization_progress.progress(100)
                optimization_status.success("Parameters optimized successfully!")
                
                # Generate optimized parameters with meaningful improvements
                current_params = st.session_state.get("optimized_parameters", {
                    "temperature": 70.0,
                    "pressure": 100.0,
                    "speed": 1800
                })
                
                # Create new parameters with small improvements
                new_params = {
                    "temperature": round(current_params["temperature"] * random.uniform(0.95, 1.05), 1),
                    "pressure": round(current_params["pressure"] * random.uniform(0.98, 1.02), 1),
                    "speed": round(current_params["speed"] * random.uniform(0.97, 1.03))
                }
                
                # Calculate efficiency gains
                efficiency_gain = round(random.uniform(5, 15), 1)
                energy_savings = round(random.uniform(8, 20), 1)
                quality_improvement = round(random.uniform(3, 12), 1)
                
                # Store the optimized parameters
                st.session_state.optimized_parameters = new_params
                
                # Display the optimized parameters with comparison
                st.subheader("Optimized Parameters")
                
                # Create comparison dataframe
                comparison_data = {
                    "Parameter": list(new_params.keys()),
                    "Current Value": [current_params[k] for k in new_params.keys()],
                    "Optimized Value": list(new_params.values()),
                    "Change": [f"{round((new_params[k] - current_params[k])/current_params[k] * 100, 1)}%" 
                              for k in new_params.keys()]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, hide_index=True)
                
                # Display improvements
                st.success(f"""
                    Expected improvements:
                    - Overall Efficiency: +{efficiency_gain}%
                    - Energy Consumption: -{energy_savings}%
                    - Product Quality: +{quality_improvement}%
                """)
                
                # Button to apply these parameters
                if st.button("Apply Optimized Parameters"):
                    st.success("Parameters applied to the manufacturing process!")
                    
                    # You'd have code here to actually apply the parameters to the system
    
    # Run Simulation Button with enhanced functionality
    with tool_col2:
        if st.button("Run Simulation", key="run_sim", use_container_width=True):
            with st.sidebar:
                # Create a container for progress updates
                simulation_status = st.empty()
                simulation_status.info("Initializing simulation...")
                
                # Create progress bar
                simulation_progress = st.progress(0)
                
                # Initialize session state for results if not present
                if "simulation_results" not in st.session_state:
                    st.session_state.simulation_results = []
                
                # Simulation steps
                steps = ["Configuring environment", "Loading process parameters", 
                         "Setting up digital twin", "Running physics engine", 
                         "Calculating outputs", "Generating visualization"]
                
                # Simulate the process with detailed updates
                for i, step in enumerate(steps):
                    # Calculate progress percentage
                    progress_pct = int((i / len(steps)) * 100)
                    
                    # Update the progress bar
                    simulation_progress.progress(progress_pct)
                    simulation_status.info(f"Step {i+1}/{len(steps)}: {step}")
                    
                    # Sleep to simulate processing time
                    time.sleep(0.5)
                    
                    # Add a result at each major step
                    result = {
                        "step": i+1,
                        "description": step,
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {
                            "temperature": round(random.uniform(60, 90), 1),
                            "pressure": round(random.uniform(90, 110), 1),
                            "vibration": round(random.uniform(0.1, 0.5), 2)
                        }
                    }
                    st.session_state.simulation_results.append(result)
                    
                    # Show latest metrics
                    metrics_str = ", ".join([f"{k}: {v}" for k, v in result['metrics'].items()])
                    st.text(f"Metrics: {metrics_str}")
                
                # Complete the simulation
                simulation_progress.progress(100)
                simulation_status.success("Simulation completed successfully!")
                
                # Show the final results
                st.subheader("Simulation Results")
                st.json(st.session_state.simulation_results[-1]['metrics'])
                
                # Add download button for simulation results
                st.download_button(
                    label="Download Results", 
                    data=json.dumps(st.session_state.simulation_results),
                    file_name="simulation_results.json",
                    mime="application/json"
                )
    
    # Add a section for collaboration tools
    st.sidebar.title("Collaboration Tools")
    
    # Initialize collaboration tool states if they don't exist
    if "active_collab_tool" not in st.session_state:
        st.session_state.active_collab_tool = None
    if "notion_pages" not in st.session_state:
        st.session_state.notion_pages = [
            {"title": "Project Overview", "updated": "Today at 11:23 AM", "content": "This document contains the main project objectives and timelines."},
            {"title": "Technical Specifications", "updated": "Yesterday", "content": "Detailed specifications for the HyperSyncDT platform architecture."},
            {"title": "Meeting Notes", "updated": "Mar 10, 2025", "content": "Notes from the weekly team sync and product review."}
        ]
    if "teams_messages" not in st.session_state:
        st.session_state.teams_messages = [
            {"sender": "Maria Chen", "time": "10:42 AM", "message": "Has everyone reviewed the latest test results?"},
            {"sender": "David Wong", "time": "10:45 AM", "message": "Yes, I've added my comments to the spreadsheet."},
            {"sender": "Sarah Johnson", "time": "10:50 AM", "message": "The energy optimization module is showing a 12% improvement!"}
        ]
    if "slack_channels" not in st.session_state:
        st.session_state.slack_channels = [
            {"name": "general", "unread": 0},
            {"name": "dev-team", "unread": 3},
            {"name": "manufacturing", "unread": 1},
            {"name": "digital-twin", "unread": 7}
        ]
    if "current_slack_channel" not in st.session_state:
        st.session_state.current_slack_channel = "dev-team"
    if "slack_messages" not in st.session_state:
        st.session_state.slack_messages = [
            {"sender": "Carlos", "time": "11:02 AM", "message": "Just pushed the updated sensor calibration module"},
            {"sender": "Priya", "time": "11:05 AM", "message": "Great! I'll test it with the physical setup this afternoon"},
            {"sender": "Bot", "time": "11:10 AM", "message": "🤖 CI/CD Pipeline: Build #342 passed all tests"}
        ]
    
    # Collaboration tools with embedded functionality
    collab_cols = st.sidebar.columns(3)
    
    # Notion button
    if collab_cols[0].button("📝 Notion", key="notion_btn", use_container_width=True):
        st.session_state.active_collab_tool = "notion"
        # Close any open media tool when switching to collaboration tool
        st.session_state.active_media_tool = None
    
    # Teams button
    if collab_cols[1].button("👥 Teams", key="teams_btn", use_container_width=True):
        st.session_state.active_collab_tool = "teams"
        # Close any open media tool when switching to collaboration tool
        st.session_state.active_media_tool = None
    
    # Slack button
    if collab_cols[2].button("💬 Slack", key="slack_btn", use_container_width=True):
        st.session_state.active_collab_tool = "slack"
        # Close any open media tool when switching to collaboration tool
        st.session_state.active_media_tool = None
    
    # Add Media Tools section in the sidebar
    st.sidebar.title("Media Tools")
    
    # Initialize media tool states if they don't exist
    if "active_media_tool" not in st.session_state:
        st.session_state.active_media_tool = None
    if "recording_started" not in st.session_state:
        st.session_state.recording_started = False
    if "recording_time" not in st.session_state:
        st.session_state.recording_time = 0
    if "last_recording_update" not in st.session_state:
        st.session_state.last_recording_update = time.time()
    if "speech_text_result" not in st.session_state:
        st.session_state.speech_text_result = ""
    if "text_to_speak" not in st.session_state:
        st.session_state.text_to_speak = "Welcome to HyperSyncDT platform."
    if "stream_quality" not in st.session_state:
        st.session_state.stream_quality = "Medium"
    
    # Media tools with embedded functionality
    media_cols = st.sidebar.columns(2)
    
    # Screen Recorder
    if media_cols[0].button("🎬 Screen Recorder", use_container_width=True):
        st.session_state.active_media_tool = "screen_recorder"
    
    # Speech to Text
    if media_cols[0].button("🎤 Speech to Text", use_container_width=True):
        st.session_state.active_media_tool = "speech_to_text"
    
    # Text to Speech
    if media_cols[1].button("🔊 Text to Speech", use_container_width=True):
        st.session_state.active_media_tool = "text_to_speech"
    
    # Video Stream
    if media_cols[1].button("📹 Video Stream", use_container_width=True):
        st.session_state.active_media_tool = "video_stream"
    
    # YouTube Stream
    if media_cols[0].button("📺 YouTube Stream", use_container_width=True):
        st.session_state.active_media_tool = "youtube_stream"
    
    # Meeting Room
    if media_cols[1].button("👥 Meeting Room", use_container_width=True):
        st.session_state.active_media_tool = "meeting_room"
    
    # Display the selected media tool interface in a container below the main content
    if st.session_state.active_media_tool:
        # Create a divider
        st.markdown("---")
        st.subheader(f"Media Tool: {st.session_state.active_media_tool.replace('_', ' ').title()}")
        
        # Container for the media tool
        media_container = st.container()
        
        with media_container:
            # Screen Recorder Tool
            if st.session_state.active_media_tool == "screen_recorder":
                cols = st.columns([3, 1])
                
                with cols[0]:
                    st.markdown("### Screen Recording")
                    
                    # Preview area
                    if not st.session_state.recording_started:
                        st.image("https://via.placeholder.com/800x450?text=Screen+Preview", use_container_width=True)
                    else:
                        # Simulate active recording with red border
                        st.markdown("""
                        <style>
                        .recording-active {
                            border: 4px solid red;
                            padding: 10px;
                        }
                        </style>
                        <div class="recording-active">
                            <img src="https://via.placeholder.com/800x450?text=Recording+Active" width="100%">
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Update recording time
                        current_time = time.time()
                        st.session_state.recording_time += current_time - st.session_state.last_recording_update
                        st.session_state.last_recording_update = current_time
                        
                        mins, secs = divmod(int(st.session_state.recording_time), 60)
                        st.markdown(f"**Recording Time**: {mins:02d}:{secs:02d}")
                
                with cols[1]:
                    st.markdown("### Controls")
                    
                    # Recording controls
                    if not st.session_state.recording_started:
                        if st.button("Start Recording", key="start_rec", use_container_width=True):
                            st.session_state.recording_started = True
                            st.session_state.recording_time = 0
                            st.session_state.last_recording_update = time.time()
                            st.rerun()
                    else:
                        if st.button("Stop Recording", key="stop_rec", use_container_width=True):
                            st.session_state.recording_started = False
                            
                            # Provide download option
                            st.success("Recording saved!")
                            st.download_button(
                                "Download Recording",
                                data=b"Sample recording data",
                                file_name="screen_recording.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                    
                    # Screen capture options
                    st.subheader("Capture Options")
                    st.radio("Capture Area", ["Full Screen", "Application Window", "Custom Selection"])
                    st.checkbox("Include Audio")
                    st.checkbox("Show Mouse Cursor")
                    
                    # Recording quality
                    st.select_slider("Quality", options=["Low", "Medium", "High"], value="Medium")
            
            # Speech to Text Tool
            elif st.session_state.active_media_tool == "speech_to_text":
                cols = st.columns([3, 1])
                
                with cols[0]:
                    st.markdown("### Speech to Text Converter")
                    
                    # Upload or record option
                    option = st.radio("Input Method", ["Upload Audio", "Record Audio"])
                    
                    if option == "Upload Audio":
                        # File uploader
                        audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])
                        if audio_file is not None:
                            st.audio(audio_file)
                            if st.button("Transcribe Audio"):
                                # Simulate transcription
                                with st.spinner("Transcribing..."):
                                    time.sleep(1.5)
                                    st.session_state.speech_text_result = "This is a simulated transcription of the uploaded audio file. The actual transcription would contain the spoken content from your audio."
                                st.success("Transcription complete!")
                    else:
                        # Microphone recording
                        st.markdown("#### Record Audio")
                        if st.button("Start Recording", key="start_mic"):
                            with st.spinner("Recording..."):
                                # Simulate recording
                                time.sleep(2)
                                st.session_state.speech_text_result = "This is a simulated transcription of your speech. In a real app, this would contain the words you spoke into the microphone."
                            st.success("Speech recognized!")
                
                with cols[1]:
                    st.markdown("### Options")
                    st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])
                    st.checkbox("Filter Background Noise")
                    st.checkbox("Punctuation")
                    st.checkbox("Speaker Identification")
                
                # Display transcription result
                if st.session_state.speech_text_result:
                    st.subheader("Transcription Result")
                    st.text_area("", value=st.session_state.speech_text_result, height=200)
                    
                    # Actions for the transcription
                    action_cols = st.columns(3)
                    action_cols[0].download_button("Download as Text", data=st.session_state.speech_text_result, file_name="transcription.txt")
                    action_cols[1].button("Copy to Clipboard")
                    action_cols[2].button("Clear Result", on_click=lambda: setattr(st.session_state, "speech_text_result", ""))
            
            # Text to Speech Tool
            elif st.session_state.active_media_tool == "text_to_speech":
                cols = st.columns([3, 1])
                
                with cols[0]:
                    st.markdown("### Text to Speech Converter")
                    
                    # Text input area
                    st.session_state.text_to_speak = st.text_area(
                        "Enter text to convert to speech",
                        value=st.session_state.text_to_speak,
                        height=200
                    )
                    
                    # Generate and play audio
                    if st.button("Generate Speech"):
                        with st.spinner("Generating audio..."):
                            # Simulate audio generation
                            time.sleep(1.5)
                        
                        # Use a sample audio file as placeholder
                        st.success("Audio generated!")
                        st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")
                        
                        # Download option
                        st.download_button(
                            "Download Audio",
                            data=b"Sample audio data",
                            file_name="generated_speech.mp3",
                            mime="audio/mp3"
                        )
                
                with cols[1]:
                    st.markdown("### Voice Settings")
                    st.selectbox("Voice", ["Default", "Male", "Female", "Child", "Robot"])
                    st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])
                    st.slider("Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
                    st.slider("Pitch", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            
            # Video Stream Tool
            elif st.session_state.active_media_tool == "video_stream":
                cols = st.columns([3, 1])
                
                with cols[0]:
                    st.markdown("### Video Stream")
                    
                    # Video stream preview
                    st.image("https://via.placeholder.com/800x450?text=Video+Stream", use_container_width=True)
                    
                    # Stream controls below the video
                    control_cols = st.columns(4)
                    control_cols[0].button("⏺️ Start", use_container_width=True)
                    control_cols[1].button("⏹️ Stop", use_container_width=True)
                    control_cols[2].button("🔇 Mute", use_container_width=True)
                    control_cols[3].button("📷 Switch Camera", use_container_width=True)
                
                with cols[1]:
                    st.markdown("### Stream Settings")
                    
                    # Stream quality
                    st.session_state.stream_quality = st.radio(
                        "Quality",
                        ["Low", "Medium", "High"],
                        index=["Low", "Medium", "High"].index(st.session_state.stream_quality)
                    )
                    
                    # Show bandwidth based on quality
                    bandwidth = {"Low": "1-2", "Medium": "3-5", "High": "6-10"}
                    st.info(f"Bandwidth: {bandwidth[st.session_state.stream_quality]} Mbps")
                    
                    # Other stream options
                    st.checkbox("Enable Audio")
                    st.checkbox("Background Blur")
                    st.checkbox("Virtual Background")
                    st.selectbox("Aspect Ratio", ["16:9", "4:3", "1:1"])
                    
                    # Copy stream link
                    st.text_input("Stream URL", value="rtmp://stream.example.com/live/stream-key")
                    st.button("Copy Stream URL", use_container_width=True)
            
            # YouTube Stream Tool
            elif st.session_state.active_media_tool == "youtube_stream":
                # Replace import with inline implementation
                st.title("Media Tool: Youtube Stream")
                
                # Main layout with two columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # YouTube Live Stream section
                    st.header("YouTube Live Stream")
                    
                    # Placeholder for the live stream preview
                    stream_container = st.container()
                    with stream_container:
                        # YouTube logo as placeholder when not streaming
                        st.markdown(
                            """
                            <div style="display: flex; justify-content: center; align-items: center; 
                            height: 400px; background-color: #000; border-radius: 10px;">
                                <div style="background-color: #FF0000; border-radius: 20px; 
                                width: 80px; height: 60px; display: flex; justify-content: center; align-items: center;">
                                    <div style="border-style: solid; border-width: 15px 0 15px 30px; 
                                    border-color: transparent transparent transparent #FFFFFF;"></div>
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Create tabs for stream info, chat, and analytics
                    info_tab, chat_tab, analytics_tab = st.tabs(["Stream Info", "Live Chat", "Analytics"])
                    
                    with info_tab:
                        st.subheader("Live Viewers")
                        viewers_col1, viewers_col2 = st.columns([1, 2])
                        
                        with viewers_col1:
                            st.metric(label="", value="24", delta="+8")
                        
                        with viewers_col2:
                            st.markdown("**Watch Time**")
                            st.metric(label="", value="48 minutes", delta="+12%")
                            
                        st.markdown("**Engagement**")
                        st.metric(label="", value="86%", delta="+5%")
                    
                    with chat_tab:
                        st.markdown("### Live Chat")
                        st.markdown("_Chat messages will appear here_")
                    
                    with analytics_tab:
                        st.markdown("### Real-time Analytics")
                        st.bar_chart({"Viewers": [5, 10, 15, 24, 22, 18, 20, 24]})
                
                with col2:
                    # Stream Controls section
                    st.header("Stream Controls")
                    
                    # Stream status dropdown
                    st.markdown("**Stream Status**")
                    stream_status = st.selectbox(
                        label="",
                        options=["Live", "Scheduled", "Offline"],
                        index=0,
                        key="stream_status"
                    )
                    
                    # Stream control buttons
                    if stream_status == "Live":
                        if st.button("End Stream", key="end_stream", use_container_width=True):
                            st.success("Stream ended successfully")
                    else:
                        if st.button("Start Stream", key="start_stream", use_container_width=True):
                            st.success("Stream started successfully")
                            
                            # Show stream key and URL
                            with st.expander("Stream Details"):
                                st.code("rtmp://a.rtmp.youtube.com/live2", language=None)
                                st.code("simulated-stream-key-789", language=None)
                    
                    # Settings section
                    st.header("Settings")
                    
                    # Privacy settings
                    st.markdown("**Privacy**")
                    privacy_col1, privacy_col2, privacy_col3 = st.columns(3)
                    
                    with privacy_col1:
                        st.radio("", ["Public"], key="privacy_public", label_visibility="collapsed")
                    
                    with privacy_col2:
                        st.radio("", ["Unlisted"], key="privacy_unlisted", label_visibility="collapsed")
                    
                    with privacy_col3:
                        st.radio("", ["Private"], key="privacy_private", label_visibility="collapsed")
                    
                    # DVR option
                    st.checkbox("Enable DVR", value=True)
                    
                    # Auto-start option
                    st.checkbox("Auto-start when live")
                    
                    # Stream delay slider
                    st.markdown("**Stream Delay**")
                    st.markdown("5 sec")
                    delay_value = st.slider("", min_value=0, max_value=30, value=5, label_visibility="collapsed")
            
            # Meeting Room Tool
            elif st.session_state.active_media_tool == "meeting_room":
                cols = st.columns([3, 1])
                
                with cols[0]:
                    st.markdown("### Virtual Meeting Room")
                    
                    # Video conference grid
                    conference_cols = st.columns(3)
                    
                    # Add participant videos (placeholders)
                    for i, col in enumerate(conference_cols):
                        with col:
                            if i == 0:
                                st.markdown("**You (Host)**")
                                st.image("https://via.placeholder.com/300x200?text=You", use_container_width=True)
                            else:
                                st.markdown(f"**Participant {i}**")
                                st.image(f"https://via.placeholder.com/300x200?text=Participant+{i}", use_container_width=True)
                    
                    # Meeting actions
                    action_cols = st.columns(5)
                    action_cols[0].button("🎤 Mute", use_container_width=True)
                    action_cols[1].button("📷 Camera", use_container_width=True)
                    action_cols[2].button("🖥️ Share", use_container_width=True)
                    action_cols[3].button("✋ Raise Hand", use_container_width=True)
                    action_cols[4].button("❌ Leave", use_container_width=True)
                    
                    # Chat and participants tabs
                    meeting_tabs = st.tabs(["Chat", "Participants", "Settings"])
                    
                    with meeting_tabs[0]:
                        # Meeting chat
                        st.markdown("""
                        **John**: Hi everyone, let's discuss the manufacturing improvements.  
                        **Sara**: I've compiled the data from last week's tests.  
                        **Miguel**: Great! Can we look at the energy consumption metrics?  
                        **You**: Let me share my screen to show the dashboard.
                        """)
                        
                        chat_cols = st.columns([3, 1])
                        meeting_msg = chat_cols[0].text_input("Type a message", key="meeting_msg")
                        chat_cols[1].button("Send", key="send_meeting_msg", use_container_width=True)
                    
                    with meeting_tabs[1]:
                        # Participants list
                        st.markdown("""
                        **Participants (4)**
                        
                        👑 You (Host)  
                        👤 John Smith  
                        👤 Sara Johnson  
                        👤 Miguel Rodriguez
                        """)
                        
                        st.button("Invite More Participants")
                    
                    with meeting_tabs[2]:
                        # Meeting settings
                        st.checkbox("Mute participants on entry")
                        st.checkbox("Lock meeting")
                        st.checkbox("Enable waiting room")
                        st.selectbox("Video quality", ["Standard", "HD", "Full HD"])
                
                with cols[1]:
                    st.markdown("### Meeting Info")
                    
                    # Meeting details
                    st.markdown("**Meeting ID:**")
                    meeting_id = "123-456-789"
                    st.code(meeting_id)
                    
                    st.markdown("**Invite Link:**")
                    invite_link = "https://meet.example.com/abc-def-ghi"
                    st.code(invite_link)
                    
                    # Copy buttons
                    st.button("Copy ID", use_container_width=True)
                    st.button("Copy Link", use_container_width=True)
                    st.button("Email Invite", use_container_width=True)
                    
                    # Recording options
                    st.subheader("Recording")
                    st.checkbox("Record meeting")
                    st.selectbox("Save to", ["Cloud", "Local Device"])
    
    # Display the selected collaboration tool interface if a media tool is not active
    elif st.session_state.active_collab_tool:
        # Create a divider
        st.markdown("---")
        st.subheader(f"Collaboration Tool: {st.session_state.active_collab_tool.title()}")
        
        # Container for the collaboration tool
        collab_container = st.container()
        
        with collab_container:
            # Notion Interface
            if st.session_state.active_collab_tool == "notion":
                cols = st.columns([1, 3])
                
                with cols[0]:
                    st.markdown("### Workspace")
                    
                    # Add search bar
                    st.text_input("Search", placeholder="Search Notion...", key="notion_search")
                    
                    # Add pages and sections
                    st.markdown("#### Pages")
                    for i, page in enumerate(st.session_state.notion_pages):
                        if st.button(page["title"], key=f"notion_page_{i}", use_container_width=True):
                            # Set the selected page
                            st.session_state.selected_notion_page = i
                            st.rerun()
                    
                    # Add new page button
                    st.button("+ New Page", type="primary", key="notion_new_page", use_container_width=True)
                    
                    # Favorites and recent sections
                    st.markdown("#### Favorites")
                    st.markdown("- Dashboard")
                    st.markdown("- Project Plan")
                    st.markdown("- Tasks")
                    
                    st.markdown("#### Recent")
                    st.markdown("- Meeting Notes")
                    st.markdown("- Technical Specs")
                
                with cols[1]:
                    st.markdown("### Document")
                    
                    # Determine which page to show
                    page_index = getattr(st.session_state, "selected_notion_page", 0)
                    page = st.session_state.notion_pages[page_index]
                    
                    # Show page title and metadata
                    st.markdown(f"## {page['title']}")
                    st.markdown(f"*Last updated: {page['updated']}*")
                    
                    # Document content
                    content = st.text_area("", value=page['content'], height=300, 
                                 key=f"notion_content_{page_index}")
                    
                    # Update document content in session state
                    st.session_state.notion_pages[page_index]['content'] = content
                    
                    # Document actions
                    action_cols = st.columns(4)
                    action_cols[0].button("💾 Save", key="notion_save", use_container_width=True)
                    action_cols[1].button("📤 Share", key="notion_share", use_container_width=True)
                    action_cols[2].button("📎 Attach", key="notion_attach", use_container_width=True)
                    action_cols[3].button("🔄 Sync", key="notion_sync", use_container_width=True)
                    
                    # Add comments section
                    st.markdown("### Comments")
                    
                    # Simulate a few comments
                    st.markdown("""
                    **Alex:** I think we should add more detail to the sensor calibration section.  
                    **Mia:** Agreed, I'll update with the latest specs this afternoon.
                    """)
                    
                    # Add comment input
                    comment_cols = st.columns([3, 1])
                    comment = comment_cols[0].text_input("Add a comment", key="notion_comment")
                    comment_cols[1].button("Post", key="notion_post", use_container_width=True)
            
            # Teams Interface
            elif st.session_state.active_collab_tool == "teams":
                cols = st.columns([1, 3])
                
                with cols[0]:
                    st.markdown("### Channels")
                    
                    # Add search bar
                    st.text_input("Search", placeholder="Search Teams...", key="teams_search")
                    
                    # Add teams and channels
                    st.markdown("#### Teams")
                    teams = ["Manufacturing Team", "R&D", "Operations", "Management"]
                    for i, team in enumerate(teams):
                        st.button(team, key=f"team_{i}", use_container_width=True)
                    
                    # Add direct messages
                    st.markdown("#### Chat")
                    chats = ["Maria Chen", "David Wong", "Sarah Johnson", "Team Chat"]
                    for i, chat in enumerate(chats):
                        st.button(chat, key=f"chat_{i}", use_container_width=True)
                    
                    # Calendar and meetings
                    st.markdown("#### Meetings")
                    st.markdown("- Daily Standup (9:00 AM)")
                    st.markdown("- Product Review (2:00 PM)")
                    
                    # Add new meeting button
                    st.button("+ New Meeting", type="primary", key="teams_new_meeting", use_container_width=True)
                
                with cols[1]:
                    st.markdown("### Manufacturing Team Chat")
                    
                    # Chat conversation area with border
                    st.markdown("""
                    <style>
                    .chat-container {
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 10px;
                        height: 400px;
                        overflow-y: auto;
                        background-color: #f9f9f9;
                        margin-bottom: 15px;
                    }
                    </style>
                    <div class="chat-container">
                    """, unsafe_allow_html=True)
                    
                    # Display existing messages
                    for msg in st.session_state.teams_messages:
                        st.markdown(f"**{msg['sender']}** ({msg['time']})  \n{msg['message']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Message input and send button
                    msg_cols = st.columns([3, 1])
                    new_message = msg_cols[0].text_input("Type a message", key="teams_message_input")
                    
                    # Send button functionality
                    if msg_cols[1].button("Send", key="teams_send", use_container_width=True):
                        if new_message.strip():
                            # Add the new message to the conversation
                            current_time = datetime.now().strftime("%I:%M %p")
                            st.session_state.teams_messages.append({
                                "sender": "You",
                                "time": current_time,
                                "message": new_message
                            })
                            # Clear the input field (requires a rerun)
                            st.rerun()
                    
                    # Chat actions
                    action_cols = st.columns(4)
                    action_cols[0].button("📞 Call", key="teams_call", use_container_width=True)
                    action_cols[1].button("📹 Video", key="teams_video", use_container_width=True)
                    action_cols[2].button("🖥️ Share", key="teams_share", use_container_width=True)
                    action_cols[3].button("📎 Attach", key="teams_attach", use_container_width=True)
            
            # Slack Interface
            elif st.session_state.active_collab_tool == "slack":
                cols = st.columns([1, 3])
                
                with cols[0]:
                    st.markdown("### Workspace")
                    
                    # Add search bar
                    st.text_input("Search", placeholder="Search Slack...", key="slack_search")
                    
                    # Add channels section
                    st.markdown("#### Channels")
                    for i, channel in enumerate(st.session_state.slack_channels):
                        # Create a button for each channel with unread indicator
                        channel_text = f"# {channel['name']}"
                        if channel["unread"] > 0:
                            channel_text += f" ({channel['unread']})"
                        
                        # Make the current channel appear selected
                        button_type = "primary" if channel["name"] == st.session_state.current_slack_channel else "secondary"
                        
                        if st.button(channel_text, key=f"slack_channel_{i}", 
                                   use_container_width=True, type=button_type):
                            # Set as current channel and clear unread count
                            st.session_state.current_slack_channel = channel["name"]
                            # Find the channel and set unread to 0
                            for ch in st.session_state.slack_channels:
                                if ch["name"] == channel["name"]:
                                    ch["unread"] = 0
                                    break
                            st.rerun()
                    
                    # Add new channel button
                    st.button("+ Add Channel", key="slack_new_channel", use_container_width=True)
                    
                    # Direct messages section
                    st.markdown("#### Direct Messages")
                    dms = ["Nikolai", "Jamal", "Priya", "Carlos"]
                    for i, dm in enumerate(dms):
                        st.button(dm, key=f"slack_dm_{i}", use_container_width=True)
                
                with cols[1]:
                    # Get the current channel name
                    current_channel = st.session_state.current_slack_channel
                    st.markdown(f"### #{current_channel}")
                    
                    # Channel description
                    st.markdown("Development team discussions and updates")
                    
                    # Chat conversation area with styling
                    st.markdown("""
                    <style>
                    .slack-container {
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 10px;
                        height: 400px;
                        overflow-y: auto;
                        background-color: #f9f9f9;
                        margin-bottom: 15px;
                    }
                    </style>
                    <div class="slack-container">
                    """, unsafe_allow_html=True)
                    
                    # Display existing messages
                    for msg in st.session_state.slack_messages:
                        st.markdown(f"**{msg['sender']}** ({msg['time']})  \n{msg['message']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Message input and send button
                    msg_cols = st.columns([3, 1])
                    new_message = msg_cols[0].text_input("Message", key="slack_message_input")
                    
                    # Send button functionality
                    if msg_cols[1].button("Send", key="slack_send", use_container_width=True):
                        if new_message.strip():
                            # Add the new message to the conversation
                            current_time = datetime.now().strftime("%I:%M %p")
                            st.session_state.slack_messages.append({
                                "sender": "You",
                                "time": current_time,
                                "message": new_message
                            })
                            # Clear the input field (requires a rerun)
                            st.rerun()
                    
                    # Message actions
                    action_cols = st.columns(4)
                    action_cols[0].button("😀 Emoji", key="slack_emoji", use_container_width=True)
                    action_cols[1].button("📎 Attach", key="slack_attach", use_container_width=True)
                    action_cols[2].button("👤 Mention", key="slack_mention", use_container_width=True)
                    action_cols[3].button("🔗 Link", key="slack_link", use_container_width=True)

# Initialize session state for real-time data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['timestamp', 'wear', 'temperature', 'vibration'])

# Simulate real-time data
async def update_real_time_data():
    # Ensure we have a running event loop
    ensure_event_loop()
    
    while True:
        try:
            new_data = get_real_time_data()
            # Filter out empty or all-NA entries before concatenation
            new_data_filtered = new_data.dropna(how='all')
            
            # Ensure data exists in session state
            if 'data' not in st.session_state:
                st.session_state.data = pd.DataFrame(columns=['timestamp', 'wear', 'temperature', 'vibration'])
            
            # Fix for FutureWarning: ensure both DataFrames have the same columns
            # Filter out empty or NA columns from new_data_filtered
            if not new_data_filtered.empty:
                # Ensure both DataFrames have the same columns
                for col in st.session_state.data.columns:
                    if col not in new_data_filtered.columns:
                        new_data_filtered[col] = pd.NA
                
                # Only keep columns that exist in the session state DataFrame
                new_data_filtered = new_data_filtered[st.session_state.data.columns]
                
                # Check if new_data_filtered is empty
                if not new_data_filtered.empty:
                    # Filter out columns with all NA values before concatenation
                    cols_to_use = [col for col in new_data_filtered.columns 
                                  if not new_data_filtered[col].isna().all()]
                    
                    if cols_to_use:
                        # Now concatenate with matching columns
                        st.session_state.data = pd.concat(
                            [st.session_state.data, new_data_filtered[cols_to_use]], 
                            ignore_index=True
                        )
                        # Ensure any missing columns are added back
                        for col in st.session_state.data.columns:
                            if col not in cols_to_use:
                                st.session_state.data[col] = st.session_state.data[col].astype(
                                    st.session_state.data[col].dtype
                                )
            await asyncio.sleep(1)
        except Exception as e:
            st.error(f"Error in real-time data update: {str(e)}")
            await asyncio.sleep(5)  # Wait longer before retrying on error

# Start real-time data update in a try-except block to handle errors gracefully
try:
    # Replace asyncio.run with a synchronous approach
    if 'update_thread' not in st.session_state or not st.session_state.update_thread.is_alive():
        def run_async_update():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(update_real_time_data())
            except Exception as e:
                print(f"Error in update thread: {str(e)}")
            finally:
                new_loop.close()
        
        # Create and start the thread
        import threading
        st.session_state.update_thread = threading.Thread(target=run_async_update, daemon=True)
        st.session_state.update_thread.start()
except Exception as e:
    st.error(f"Failed to start real-time data update: {str(e)}")

# Function to send a message to a Slack channel
def send_slack_message(channel: str, message: str, token: str):
    url = 'https://slack.com/api/chat.postMessage'
    headers = {'Authorization': f'Bearer {token}'}
    data = {
        'channel': channel,
        'text': message
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Function to create a note in Notion
def create_notion_note(page_id: str, title: str, content: str, token: str):
    url = f'https://api.notion.com/v1/pages'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'Notion-Version': '2021-05-13'
    }
    data = {
        'parent': {'page_id': page_id},
        'properties': {
            'title': {
                'title': [
                    {
                        'text': {
                            'content': title
                        }
                    }
                ]
            },
            'content': content
        }
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Function to send a message to a Microsoft Teams channel
def send_teams_message(webhook_url: str, message: str):
    headers = {'Content-Type': 'application/json'}
    data = {
        'text': message
    }
    response = requests.post(webhook_url, headers=headers, json=data)
    return response.json()

# Example usage
# slack_token = 'your-slack-token'
# notion_token = 'your-notion-token'
# send_slack_message('#general', 'Hello from Streamlit!', slack_token)
# send_teams_message('your-teams-webhook-url', 'Hello from Streamlit!')
# send_notion_message('your-page-id', 'New Note', 'This is the content of the note.', notion_token)

def render_manufacturing_tools():
    """Render the Manufacturing Tools page with interactive controls"""
    st.header("Manufacturing Tools")
    
    # Initialize session state variables if they don't exist
    if "simulation_complete" not in st.session_state:
        st.session_state.simulation_complete = False
    if "optimization_complete" not in st.session_state:
        st.session_state.optimization_complete = False
    if "optimized_params" not in st.session_state:
        st.session_state.optimized_params = None
    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = None
    if "simulation_progress" not in st.session_state:
        st.session_state.simulation_progress = 0
    if "optimization_progress" not in st.session_state:
        st.session_state.optimization_progress = 0
    
    # Create tabs for different manufacturing tool options
    tools_tab1, tools_tab2, tools_tab3 = st.tabs([
        "Process Parameters",
        "Interactive Tools", 
        "Results Visualization"
    ])
    
    # Process Parameters tab
    with tools_tab1:
        st.subheader("Manufacturing Process Parameters")
        
        # Create columns for material and parameters
        col1, col2 = st.columns([1, 2])
        
if __name__ == "__main__":
    main()
