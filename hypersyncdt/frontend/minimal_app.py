import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Set page configuration
st.set_page_config(
    page_title="HyperSyncDT",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple header
st.title("HyperSyncDT Digital Twin Platform")
st.markdown("### Factory Intelligence & Monitoring")

# Generate sample data
def generate_sample_data(n_points=50):
    """Generate sample time series data"""
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(n_points)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': 70 + np.sin(np.linspace(0, 6*np.pi, n_points)) * 10 + np.random.normal(0, 1, n_points),
        'pressure': 100 + np.cos(np.linspace(0, 4*np.pi, n_points)) * 15 + np.random.normal(0, 2, n_points),
        'flow_rate': 50 + np.sin(np.linspace(0, 8*np.pi, n_points)) * 8 + np.random.normal(0, 1, n_points),
        'vibration': 0.5 + 0.2 * np.sin(np.linspace(0, 10*np.pi, n_points)) + np.random.normal(0, 0.05, n_points),
        'power': 120 + np.sin(np.linspace(0, 3*np.pi, n_points)) * 20 + np.random.normal(0, 3, n_points),
        'efficiency': 85 + np.cos(np.linspace(0, 5*np.pi, n_points)) * 8 + np.random.normal(0, 1, n_points),
        'quality': 95 + np.sin(np.linspace(0, 7*np.pi, n_points)) * 3 + np.random.normal(0, 0.5, n_points)
    })

# Create metrics for dashboard
def render_metrics():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Production Efficiency", f"{random.uniform(85, 95):.1f}%", f"{random.uniform(-2, 4):.1f}%")
    
    with col2:
        st.metric("Quality Score", f"{random.uniform(90, 98):.1f}%", f"{random.uniform(-1, 3):.1f}%")
    
    with col3:
        st.metric("Energy Usage", f"{random.uniform(300, 400):.0f} kWh", f"{random.uniform(-5, 2):.1f}%")
    
    with col4:
        st.metric("Active Systems", f"{random.randint(8, 12)}/12", None)

# Create visualization
def render_charts():
    # Get sample data
    data = generate_sample_data()
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature & Pressure")
        fig1 = go.Figure()
        
        # Add temperature line
        fig1.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['temperature'],
            name="Temperature (°C)",
            line=dict(color='#FF5733', width=2)
        ))
        
        # Add pressure line with secondary y-axis
        fig1.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['pressure'],
            name="Pressure (PSI)",
            line=dict(color='#33A1FF', width=2),
            yaxis="y2"
        ))
        
        # Update layout with dual y-axes
        fig1.update_layout(
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="Pressure (PSI)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("System Performance")
        fig2 = go.Figure()
        
        # Add efficiency area
        fig2.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['efficiency'],
            name="Efficiency (%)",
            fill='tozeroy',
            line=dict(color='#0CCA4A', width=2)
        ))
        
        # Add quality line
        fig2.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['quality'],
            name="Quality (%)",
            line=dict(color='#AC33FF', width=2)
        ))
        
        # Update layout
        fig2.update_layout(
            yaxis=dict(title="Percentage (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig2, use_container_width=True)

# Add sidebar
with st.sidebar:
    st.title("HyperSyncDT")
    st.markdown("## Navigation")
    
    # Create basic navigation buttons
    page = st.radio("", [
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
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        st.success("Logged in successfully!")

# Main content
st.markdown("## Factory Operational Dashboard")
render_metrics()

# Add process monitoring section
st.markdown("---")
render_charts()

# Add table with sample data
st.markdown("---")
st.subheader("Recent Process Data")
data = generate_sample_data(10)
data['timestamp'] = data['timestamp'].dt.strftime('%H:%M:%S')
st.dataframe(data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### HyperSyncDT Platform | Digital Twin Solution")
st.markdown("© 2025 Advanced Manufacturing Intelligence") 