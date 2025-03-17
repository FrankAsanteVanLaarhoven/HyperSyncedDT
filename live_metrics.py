import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate sample metrics data for demonstration"""
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(30)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': np.random.normal(60, 10, 30),
        'memory_usage': np.random.normal(70, 8, 30),
        'network_traffic': np.random.normal(45, 15, 30),
        'active_processes': np.random.randint(10, 20, 30)
    })

def create_metric_chart(df, metric_name, color):
    """Create a line chart for a specific metric"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[metric_name],
        mode='lines',
        name=metric_name,
        line=dict(color=color, width=2)
    ))
    
    fig.update_layout(
        title=f"{metric_name.replace('_', ' ').title()} Over Time",
        xaxis_title="Time",
        yaxis_title="Value",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def render_live_metrics():
    """Render the live metrics dashboard"""
    st.markdown("""
        <style>
        .metrics-container {
            background: rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    df = generate_sample_data()
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU Usage chart
        st.plotly_chart(
            create_metric_chart(df, 'cpu_usage', '#00ff99'),
            use_container_width=True
        )
        
        # Network Traffic chart
        st.plotly_chart(
            create_metric_chart(df, 'network_traffic', '#ff9900'),
            use_container_width=True
        )
    
    with col2:
        # Memory Usage chart
        st.plotly_chart(
            create_metric_chart(df, 'memory_usage', '#00ccff'),
            use_container_width=True
        )
        
        # Active Processes chart
        st.plotly_chart(
            create_metric_chart(df, 'active_processes', '#ff66cc'),
            use_container_width=True
        ) 