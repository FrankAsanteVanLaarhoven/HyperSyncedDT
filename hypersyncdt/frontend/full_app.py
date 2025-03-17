import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(current_dir)
sys.path.append(parent_dir)

# Import the modern dashboard UI
from components.modern_ui import ModernDashboardUI

# Import local project components with proper path handling
try:
    from components.factory_components import (
        FactoryComponents, 
        OperatorCoPilot,
        AgentFactory,
        SelfCalibratingShadow,
        render_factory_connect,
        render_factory_build,
        render_factory_analyze,
        render_factory_operate,
        render_sustainability,
        render_risk_mitigation,
        render_copilot,
        render_agent_factory
    )
    factory_components_available = True
except ImportError as e:
    st.warning(f"Failed to import factory components: {e}")
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
except ImportError as e:
    st.warning(f"Failed to import digital twin components: {e}")
    digital_twin_available = False
    
try:
    from components.live_metrics import render_live_metrics
    live_metrics_available = True
except ImportError as e:
    st.warning(f"Failed to import live metrics: {e}")
    live_metrics_available = False
    
try:
    from components.advanced_visualizations import MultiModalVisualizer
    advanced_viz_available = True
except ImportError as e:
    advanced_viz_available = False

try:
    from components.interactive_header import AdvancedInteractiveHeader
    interactive_header_available = True
except ImportError as e:
    interactive_header_available = False

# Initialize the dashboard UI
dashboard = ModernDashboardUI()

# Initialize session state for components
if factory_components_available and 'factory' not in st.session_state:
    st.session_state.factory = FactoryComponents()
    
if factory_components_available and 'copilot' not in st.session_state:
    st.session_state.copilot = OperatorCoPilot()
    
if factory_components_available and 'agent_factory' not in st.session_state:
    st.session_state.agent_factory = AgentFactory()
    
if factory_components_available and 'shadow' not in st.session_state:
    st.session_state.shadow = SelfCalibratingShadow()
    
if digital_twin_available and 'machine_connector' not in st.session_state:
    st.session_state.machine_connector = MachineConnector()
    
if digital_twin_available and 'digital_twin' not in st.session_state:
    st.session_state.digital_twin = SynchronizedDigitalTwin()
    
if advanced_viz_available and 'visualizer' not in st.session_state:
    st.session_state.visualizer = MultiModalVisualizer()
    
if interactive_header_available and 'header' not in st.session_state:
    st.session_state.header = AdvancedInteractiveHeader()

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
    role = st.selectbox("Role", ["Operator", "Administrator", "Maintenance", "Supervisor"], label_visibility="collapsed")
    
    st.markdown("## NAVIGATE TO")
    st.markdown("### Category")
    category = st.selectbox("Category", ["Factory Operations", "Digital Twin", "Maintenance", "Analytics"], key="category_select", label_visibility="collapsed")
    
    st.markdown("### Page")
    page = st.selectbox("Page", ["Factory Connect", "Factory Build", "Factory Analyze", "Factory Operate"], key="page_select", label_visibility="collapsed")
    
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
    
    # System metrics - Use metrics from factory_components if available
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
    
    # Use factory metrics if available
    efficiency = "88.5%"
    quality_score = "95.1"
    throughput = "781 units"
    utilization = "79.8%"
    
    if factory_components_available and hasattr(st.session_state, 'factory'):
        factory = st.session_state.factory
        if hasattr(factory, 'metrics') and 'production' in factory.metrics:
            efficiency = f"{factory.metrics['production'].get('efficiency', 0.885) * 100:.1f}%"
            quality_score = f"{factory.metrics['production'].get('quality_rate', 0.951) * 100:.1f}"
            utilization = f"{factory.metrics['resources'].get('utilization', 0.798) * 100:.1f}%"
            throughput = f"{factory.metrics['production'].get('units_produced', 781)} units"
    
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        dashboard.render_advanced_metric_card("Efficiency", efficiency, color="green")
        
    with adv_col2:
        # Add percentage sign for quality score to ensure proper handling
        if quality_score and not quality_score.endswith("%"):
            try:
                # Check if it's already a percentage value
                value = float(quality_score)
                if value > 0 and value <= 100:
                    quality_score = f"{value}%"
            except:
                pass
        dashboard.render_advanced_metric_card("Quality Score", quality_score, color="green")
        
    adv_col3, adv_col4 = st.columns(2)
    
    with adv_col3:
        dashboard.render_advanced_metric_card("Throughput", throughput, color="green")
        
    with adv_col4:
        dashboard.render_advanced_metric_card("Utilization", utilization, color="green")
    
    # Integrate factory visualizations if available
    if factory_components_available:
        st.markdown("## Factory Operations")
        
        if page == "Factory Connect" and 'factory' in st.session_state:
            render_factory_connect()
        elif page == "Factory Build" and 'factory' in st.session_state:
            render_factory_build()
        elif page == "Factory Analyze" and 'factory' in st.session_state:
            render_factory_analyze()
        elif page == "Factory Operate" and 'factory' in st.session_state:
            render_factory_operate()
    
    # Real-time monitoring
    st.markdown("## Real-Time Monitoring")
    
    if live_metrics_available:
        # Use the project's live metrics if available
        render_live_metrics()
    else:
        # Fall back to our mock implementation
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
    
    # Display operator recommendations if available
    if factory_components_available and 'copilot' in st.session_state:
        st.markdown("## Operator Recommendations")
        recommendations = st.session_state.copilot.get_recommendations()
        
        rec_data = []
        for rec in recommendations:
            rec_data.append({
                "Priority": rec['priority'],
                "Type": rec['type'],
                "Message": rec['message'],
                "Impact": rec['impact']
            })
        
        rec_df = pd.DataFrame(rec_data)
        st.dataframe(rec_df, use_container_width=True)
    
    # Display machine information if available
    if digital_twin_available and 'machine_connector' in st.session_state and 'digital_twin' in st.session_state:
        st.markdown("## Machine Information")
        
        # Get machine list
        machines = st.session_state.machine_connector.machine_specs
        
        # Display machine table
        machine_data = []
        for machine_id, specs in machines.items():
            machine_data.append({
                "Machine ID": machine_id,
                "Type": specs.type,
                "Model": specs.model,
                "Status": np.random.choice(["Operational", "Maintenance", "Standby"])
            })
        
        machine_df = pd.DataFrame(machine_data)
        st.dataframe(machine_df, use_container_width=True)
        
        # Show maintenance recommendations for selected machine
        selected_machine = "MHI-M8"
        
        if hasattr(st.session_state.digital_twin, 'get_maintenance_recommendations'):
            recommendations = st.session_state.digital_twin.get_maintenance_recommendations(selected_machine)
            
            st.markdown(f"### Maintenance Recommendations for {selected_machine}")
            
            if recommendations:
                rec_data = []
                for rec in recommendations:
                    rec_data.append({
                        "Component": rec['component'],
                        "Issue": rec['issue'],
                        "Current Value": rec['current_value'],
                        "Threshold": rec['threshold'],
                        "Priority": rec['priority'],
                        "Recommendation": rec['recommendation']
                    })
                
                rec_df = pd.DataFrame(rec_data)
                st.dataframe(rec_df, use_container_width=True)
            else:
                st.info("No maintenance recommendations at this time.")
    
    # System Alerts
    with st.expander("System Alerts", expanded=True):
        dashboard.render_system_alerts() 