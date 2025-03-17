import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

class LiveMetricsTracker:
    def __init__(self):
        self.digital_twin = SynchronizedDigitalTwin()
        self.visualizer = MultiModalVisualizer()
        self.last_update = datetime.now()
    
    def generate_live_metrics(self) -> Dict[str, pd.DataFrame]:
        """Generate sample live metrics data."""
        current_time = datetime.now()
        time_diff = (current_time - self.last_update).total_seconds()
        self.last_update = current_time
        
        # Generate timestamps for the last hour with 1-second intervals
        timestamps = pd.date_range(
            end=current_time,
            periods=3600,
            freq='S'
        )
        
        # Generate process metrics
        process_metrics = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': 150 + np.sin(np.linspace(0, 4*np.pi, 3600)) * 10 + np.random.normal(0, 2, 3600),
            'pressure': 100 + np.cos(np.linspace(0, 4*np.pi, 3600)) * 5 + np.random.normal(0, 1, 3600),
            'flow_rate': 75 + np.sin(np.linspace(0, 2*np.pi, 3600)) * 3 + np.random.normal(0, 0.5, 3600),
            'vibration': 0.5 + np.sin(np.linspace(0, 8*np.pi, 3600)) * 0.2 + np.random.normal(0, 0.05, 3600)
        })
        
        # Generate resource metrics
        resource_metrics = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': 45 + np.sin(np.linspace(0, 2*np.pi, 3600)) * 15 + np.random.normal(0, 5, 3600),
            'memory_usage': 60 + np.cos(np.linspace(0, 2*np.pi, 3600)) * 10 + np.random.normal(0, 3, 3600),
            'network_throughput': 50 + np.sin(np.linspace(0, 4*np.pi, 3600)) * 20 + np.random.normal(0, 5, 3600),
            'disk_io': 30 + np.cos(np.linspace(0, 4*np.pi, 3600)) * 15 + np.random.normal(0, 3, 3600)
        })
        
        # Generate quality metrics
        quality_metrics = pd.DataFrame({
            'timestamp': timestamps,
            'defect_rate': 0.02 + np.sin(np.linspace(0, 2*np.pi, 3600)) * 0.01 + np.random.normal(0, 0.002, 3600),
            'yield_rate': 0.95 + np.cos(np.linspace(0, 2*np.pi, 3600)) * 0.03 + np.random.normal(0, 0.01, 3600),
            'quality_score': 90 + np.sin(np.linspace(0, 2*np.pi, 3600)) * 5 + np.random.normal(0, 1, 3600)
        })
        
        return {
            'process': process_metrics,
            'resources': resource_metrics,
            'quality': quality_metrics
        }

def render_live_metrics():
    """Render the live metrics dashboard."""
    st.header("Live Metrics Dashboard")
    
    # Initialize tracker
    tracker = LiveMetricsTracker()
    
    # Sidebar controls
    st.sidebar.subheader("Monitoring Settings")
    update_interval = st.sidebar.selectbox(
        "Update Interval",
        ["1 second", "5 seconds", "10 seconds", "30 seconds"]
    )
    
    view_mode = st.sidebar.selectbox(
        "View Mode",
        ["All Metrics", "Process Metrics", "Resource Metrics", "Quality Metrics"]
    )
    
    time_window = st.sidebar.selectbox(
        "Time Window",
        ["Last Minute", "Last 5 Minutes", "Last 15 Minutes", "Last Hour"]
    )
    
    # Generate live data
    metrics_data = tracker.generate_live_metrics()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Process Metrics",
        "Resource Usage",
        "Quality Metrics"
    ])
    
    with tab1:
        st.subheader("System Overview")
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_temp = metrics_data['process']['temperature'].iloc[-1]
            st.metric(
                "Temperature",
                f"{current_temp:.1f}°C",
                f"{(current_temp - metrics_data['process']['temperature'].iloc[-60]):.1f}°C"
            )
        
        with col2:
            current_pressure = metrics_data['process']['pressure'].iloc[-1]
            st.metric(
                "Pressure",
                f"{current_pressure:.1f} PSI",
                f"{(current_pressure - metrics_data['process']['pressure'].iloc[-60]):.1f}"
            )
        
        with col3:
            current_cpu = metrics_data['resources']['cpu_usage'].iloc[-1]
            st.metric(
                "CPU Usage",
                f"{current_cpu:.1f}%",
                f"{(current_cpu - metrics_data['resources']['cpu_usage'].iloc[-60]):.1f}%"
            )
        
        with col4:
            current_quality = metrics_data['quality']['quality_score'].iloc[-1]
            st.metric(
                "Quality Score",
                f"{current_quality:.1f}",
                f"{(current_quality - metrics_data['quality']['quality_score'].iloc[-60]):.1f}"
            )
        
        # Multi-metric timeline
        st.write("### Real-time Process Overview")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics_data['process']['timestamp'].iloc[-300:],
            y=metrics_data['process']['temperature'].iloc[-300:],
            name='Temperature'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics_data['process']['timestamp'].iloc[-300:],
            y=metrics_data['process']['pressure'].iloc[-300:],
            name='Pressure',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Process Parameters (Last 5 Minutes)',
            yaxis=dict(title='Temperature (°C)'),
            yaxis2=dict(
                title='Pressure (PSI)',
                overlaying='y',
                side='right'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Process Metrics")
        
        # Process parameters over time
        st.write("### Process Parameters")
        
        fig = go.Figure()
        process_params = ['temperature', 'pressure', 'flow_rate', 'vibration']
        
        for param in process_params:
            fig.add_trace(go.Scatter(
                x=metrics_data['process']['timestamp'].iloc[-600:],
                y=metrics_data['process'][param].iloc[-600:],
                name=param.replace('_', ' ').title()
            ))
        
        fig.update_layout(title='Process Parameters (Last 10 Minutes)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameter distributions
        st.write("### Parameter Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                metrics_data['process'].iloc[-600:],
                x='temperature',
                title='Temperature Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                metrics_data['process'].iloc[-600:],
                x='pressure',
                title='Pressure Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Resource Usage")
        
        # Resource usage over time
        st.write("### System Resources")
        
        fig = go.Figure()
        resource_metrics = ['cpu_usage', 'memory_usage', 'network_throughput', 'disk_io']
        
        for metric in resource_metrics:
            fig.add_trace(go.Scatter(
                x=metrics_data['resources']['timestamp'].iloc[-600:],
                y=metrics_data['resources'][metric].iloc[-600:],
                name=metric.replace('_', ' ').title()
            ))
        
        fig.update_layout(title='Resource Usage (Last 10 Minutes)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Resource usage gauges
        st.write("### Current Resource Usage")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_data['resources']['cpu_usage'].iloc[-1],
                title={'text': "CPU Usage"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_data['resources']['memory_usage'].iloc[-1],
                title={'text': "Memory Usage"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_data['resources']['network_throughput'].iloc[-1],
                title={'text': "Network"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_data['resources']['disk_io'].iloc[-1],
                title={'text': "Disk I/O"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Quality Metrics")
        
        # Quality metrics over time
        st.write("### Quality Trends")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics_data['quality']['timestamp'].iloc[-600:],
            y=metrics_data['quality']['defect_rate'].iloc[-600:] * 100,
            name='Defect Rate %'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics_data['quality']['timestamp'].iloc[-600:],
            y=metrics_data['quality']['yield_rate'].iloc[-600:] * 100,
            name='Yield Rate %'
        ))
        
        fig.update_layout(
            title='Quality Metrics (Last 10 Minutes)',
            yaxis_title='Percentage'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality score distribution
        st.write("### Quality Score Distribution")
        
        fig = px.histogram(
            metrics_data['quality'].iloc[-600:],
            x='quality_score',
            title='Quality Score Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality metrics summary
        st.write("### Quality Metrics Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_defect_rate = metrics_data['quality']['defect_rate'].iloc[-600:].mean()
            st.metric(
                "Average Defect Rate",
                f"{avg_defect_rate:.1%}",
                f"{(avg_defect_rate - metrics_data['quality']['defect_rate'].iloc[-660:-600].mean()):.1%}"
            )
        
        with col2:
            avg_yield_rate = metrics_data['quality']['yield_rate'].iloc[-600:].mean()
            st.metric(
                "Average Yield Rate",
                f"{avg_yield_rate:.1%}",
                f"{(avg_yield_rate - metrics_data['quality']['yield_rate'].iloc[-660:-600].mean()):.1%}"
            )
        
        with col3:
            avg_quality = metrics_data['quality']['quality_score'].iloc[-600:].mean()
            st.metric(
                "Average Quality Score",
                f"{avg_quality:.1f}",
                f"{(avg_quality - metrics_data['quality']['quality_score'].iloc[-660:-600].mean()):.1f}"
            )
