import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(os.path.dirname(current_dir), '..'))

# Import the modern dashboard UI
from components.modern_ui import ModernDashboardUI

# Import other required components
try:
    from components.factory_components import (
        FactoryComponents, 
        OperatorCoPilot,
        AgentFactory,
        SelfCalibratingShadow
    )
    factory_components_available = True
except ImportError:
    factory_components_available = False
    
try:
    from components.digital_twin_components import (
        MachineConnector,
        DigitalTwinVisualizer,
        CameraManager,
        SensorProcessor,
        SynchronizedDigitalTwin
    )
    digital_twin_available = True
except ImportError:
    digital_twin_available = False

# Initialize the dashboard UI
dashboard = ModernDashboardUI()

# Configure the page with dark theme
st.set_page_config(
    page_title="HyperSyncDT Autonomous Agent Factory",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for dark theme
st.markdown("""
<style>
    .reportview-container {
        background-color: #121212;
        color: white;
    }
    .main .block-container {
        background-color: #121212;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1E1E1E;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #333333;
    }
    .css-10trblm {
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    [data-testid="stSidebar"] {
        background-color: #121212;
    }
    [data-testid="stSidebarContent"] {
        background-color: #121212;
    }
    div.stButton > button {
        background-color: #333333;
        color: white;
        border-radius: 5px;
    }
    div.stButton > button:hover {
        background-color: #444444;
        color: white;
    }
    .css-1aumxhk {
        background-color: #1E1E1E;
    }
</style>
""", unsafe_allow_html=True)

# Create the sidebar
with st.sidebar:
    st.title("HyperSyncDT")
    st.markdown("## Select Role")
    role = st.selectbox("", ["Operator", "Administrator", "Maintenance", "Supervisor"], label_visibility="collapsed")
    
    st.markdown("## NAVIGATE TO")
    st.markdown("### Category")
    category = st.selectbox("", ["Factory Operations", "Digital Twin", "Maintenance", "Analytics"], key="category_select", label_visibility="collapsed")
    
    st.markdown("### Page")
    page = st.selectbox("", ["Factory Connect", "Factory Build", "Factory Analyze", "Factory Operate"], key="page_select", label_visibility="collapsed")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("📊 Generate Report", use_container_width=True)
    with col2:
        st.button("🔄 Sync Data", use_container_width=True)
        
    st.markdown("---")
    st.markdown("## SYSTEM STATUS")
    
    # Show a green dot for connected status
    st.markdown("🟢 Connected as Operator")
    
    st.markdown("## BACKUP INFORMATION")
    st.markdown("📁 Backup Location:")
    st.markdown("~/Desktop/hyper-synced-dt-mvp-test")
    
    st.progress(0.7)
    
    st.markdown(f"Last backup: 2025-03-13 09:08")

# Main content container
main_container = st.container()
with main_container:
    # Quick Actions & Active Screens
    st.markdown("## Quick Actions & Active Screens")
    
    # Create a row of action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 24px;'>⚡</div>
            <div style='font-size: 14px;'>Optimize</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 24px;'>📝</div>
            <div style='font-size: 14px;'>Refresh</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 24px;'>📊</div>
            <div style='font-size: 14px;'>Report</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 24px;'>⚙️</div>
            <div style='font-size: 14px;'>Settings</div>
        </div>
        """, unsafe_allow_html=True)
    
    # System metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        dashboard.render_metric_card("CPU Usage", "49.7%", color="green")
        
    with metrics_col2:
        dashboard.render_metric_card("Memory", "41.0%", color="green")
        
    with metrics_col3:
        dashboard.render_metric_card("Network", "32.7%", color="green")
        
    with metrics_col4:
        dashboard.render_metric_card("Active Sensors", "10", subtitle="of 12 sensors online", color="green")
    
    # Advanced Metrics
    st.markdown("## Advanced Metrics")
    
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        dashboard.render_advanced_metric_card("Efficiency", "88.5%", color="green")
        
    with adv_col2:
        dashboard.render_advanced_metric_card("Quality Score", "95.1", color="green")
        
    adv_col3, adv_col4 = st.columns(2)
    
    with adv_col3:
        dashboard.render_advanced_metric_card("Throughput", "781 units", color="green")
        
    with adv_col4:
        dashboard.render_advanced_metric_card("Utilization", "79.8%", color="green")
    
    # Real-time monitoring
    st.markdown("## Real-Time Monitoring")
    
    tabs = st.tabs(["System Performance", "Resource Usage", "Temperature", "Response Time", "Errors"])
    
    with tabs[0]:
        # Create sample CPU usage data
        dates = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='1h')
        cpu_values = np.random.uniform(30, 70, size=len(dates))
        target_values = np.ones(len(dates)) * 60
        
        fig = go.Figure()
        
        # Add CPU usage line
        fig.add_trace(go.Scatter(
            x=dates,
            y=cpu_values,
            mode='lines',
            name='CPU Usage',
            line=dict(color='#3498db', width=2)
        ))
        
        # Add target line
        fig.add_trace(go.Scatter(
            x=dates,
            y=target_values,
            mode='lines',
            name='Target',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="CPU Usage Over Time",
            xaxis_title="Time",
            yaxis_title="Percentage",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font=dict(color='#CCCCCC')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # System Alerts
    with st.expander("System Alerts", expanded=True):
        dashboard.render_system_alerts() 