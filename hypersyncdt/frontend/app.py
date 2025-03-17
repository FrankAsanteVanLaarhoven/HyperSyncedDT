import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration at the very beginning
st.set_page_config(
    page_title="HyperSyncDT Autonomous Agent Factory",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the components directly from the components directory
from components.digital_twin_components import (
    MachineConnector,
    DigitalTwinVisualizer,
    CameraManager,
    SensorProcessor,
    SynchronizedDigitalTwin
)

# Initialize session state objects
if 'machine_connector' not in st.session_state:
    st.session_state.machine_connector = MachineConnector()

if 'digital_twin' not in st.session_state:
    st.session_state.digital_twin = SynchronizedDigitalTwin()

if 'visualizer' not in st.session_state:
    st.session_state.visualizer = DigitalTwinVisualizer()

if 'sensor_processor' not in st.session_state:
    st.session_state.sensor_processor = SensorProcessor()

# Main app container
main_container = st.container()
with main_container:
    # Header
    st.title("HyperSyncDT Digital Twin Platform")
    st.markdown("### Factory Intelligence & Monitoring")
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Production Efficiency", f"{np.random.uniform(85, 95):.1f}%", f"{np.random.uniform(-2, 4):.1f}%")
    
    with col2:
        st.metric("Quality Score", f"{np.random.uniform(90, 98):.1f}%", f"{np.random.uniform(-1, 3):.1f}%")
    
    with col3:
        st.metric("Energy Usage", f"{np.random.uniform(300, 400):.0f} kWh", f"{np.random.uniform(-5, 2):.1f}%")
    
    with col4:
        st.metric("Active Systems", f"{np.random.randint(8, 12)}/12", None)
    
    # Use the DigitalTwinVisualizer to render the dashboard
    st.markdown("## Machine Monitoring")
    st.session_state.visualizer.render_dashboard()
    
    # Show machine list
    st.markdown("## Available Machines")
    machines = st.session_state.machine_connector.machine_specs
    
    machine_data = []
    for machine_id, specs in machines.items():
        machine_data.append({
            "Machine ID": machine_id,
            "Type": specs.type,
            "Model": specs.model,
            "Axes": specs.axes,
            "Max Speed": f"{specs.max_speed} RPM",
            "Status": np.random.choice(["Operational", "Maintenance", "Standby"])
        })
    
    machine_df = pd.DataFrame(machine_data)
    st.dataframe(machine_df, use_container_width=True)
    
    # Show sample sensor data
    st.markdown("## Recent Sensor Readings")
    
    # Sample machine for demonstration
    selected_machine = "MHI-M8"
    
    # Get the digital twin for this machine
    machine_state = st.session_state.digital_twin.get_machine_state(selected_machine)
    
    # Display machine status
    st.markdown(f"### Machine: {selected_machine}")
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        st.metric("Temperature", f"{machine_state.temperature:.1f}°C", 
                 f"{np.random.uniform(-0.5, 1.5):.1f}°C")
    
    with status_col2:
        st.metric("Pressure", f"{machine_state.pressure:.1f} PSI", 
                 f"{np.random.uniform(-2, 2):.1f} PSI")
    
    with status_col3:
        st.metric("Vibration", f"{machine_state.vibration:.3f} mm/s", 
                 f"{np.random.uniform(-0.01, 0.02):.3f} mm/s")
    
    with status_col4:
        st.metric("Power", f"{machine_state.power_consumption:.1f} kW", 
                 f"{np.random.uniform(-0.3, 0.5):.1f} kW")
    
    # Show sensor history
    sensor_history = st.session_state.sensor_processor.get_sensor_history(
        selected_machine, "temperature", "1h")
    
    # Create the chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sensor_history['timestamp'],
        y=sensor_history['value'],
        name="Temperature",
        line=dict(color='firebrick', width=2)
    ))
    
    fig.update_layout(
        title="Temperature History (Last Hour)",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Maintenance recommendations
    st.markdown("## Maintenance Recommendations")
    recommendations = st.session_state.digital_twin.get_maintenance_recommendations(selected_machine)
    
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
        
    # Footer
    st.markdown("---")
    st.markdown("### HyperSyncDT Platform | Digital Twin Solution")
    st.markdown("© 2025 Advanced Manufacturing Intelligence")

# Sidebar
with st.sidebar:
    st.title("HyperSyncDT")
    st.markdown("## Navigation")
    
    # Create basic navigation buttons
    page = st.radio("Select Page", [
        "Dashboard", 
        "Process Monitoring", 
        "Quality Control", 
        "Predictive Maintenance",
        "Energy Optimization",
        "System Settings"
    ])
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success("All Systems Online")
    st.info(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Add a simple authentication box
    st.markdown("---")
    st.markdown("### User Access")
    user = st.text_input("Username", key="username")
    pwd = st.text_input("Password", type="password", key="password")
    if st.button("Login"):
        st.success("Logged in successfully!")
