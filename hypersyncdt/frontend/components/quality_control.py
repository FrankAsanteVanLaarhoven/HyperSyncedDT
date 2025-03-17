import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

def generate_sample_quality_data() -> pd.DataFrame:
    """Generate sample quality control data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-03-17', freq='H')
    data = {
        'timestamp': dates,
        'measurement': np.random.normal(100, 5, len(dates)),
        'upper_limit': 110,
        'lower_limit': 90,
        'target': 100
    }
    df = pd.DataFrame(data)
    df['in_control'] = (df['measurement'] >= df['lower_limit']) & (df['measurement'] <= df['upper_limit'])
    return df

def render_quality_control():
    """Render the quality control page with interactive quality monitoring tools."""
    st.header("Quality Control Dashboard")
    
    # Initialize components
    digital_twin = SynchronizedDigitalTwin()
    visualizer = MultiModalVisualizer()
    
    # Sidebar controls
    st.sidebar.subheader("Quality Parameters")
    quality_metric = st.sidebar.selectbox(
        "Quality Metric",
        ["Surface Finish", "Dimensional Accuracy", "Material Properties", "Visual Inspection"]
    )
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Hour", "Last 24 Hours", "Last Week", "Last Month"]
    )
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Real-time Monitoring",
        "Statistical Analysis",
        "Defect Analysis",
        "Reports"
    ])
    
    # Generate sample data
    quality_data = generate_sample_quality_data()
    
    with tab1:
        st.subheader("Real-time Quality Metrics")
        
        # Current status indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Value", "98.5", "+0.3")
        with col2:
            st.metric("Defect Rate", "0.5%", "-0.2%")
        with col3:
            st.metric("Process Capability", "1.33", "+0.02")
        with col4:
            st.metric("First Pass Yield", "99.2%", "+0.1%")
        
        # Control chart
        st.subheader("Control Chart")
        fig = go.Figure()
        
        # Add measurement line
        fig.add_trace(go.Scatter(
            x=quality_data['timestamp'],
            y=quality_data['measurement'],
            name='Measurement',
            mode='lines+markers'
        ))
        
        # Add control limits
        fig.add_trace(go.Scatter(
            x=quality_data['timestamp'],
            y=quality_data['upper_limit'],
            name='Upper Control Limit',
            line=dict(dash='dash', color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=quality_data['timestamp'],
            y=quality_data['lower_limit'],
            name='Lower Control Limit',
            line=dict(dash='dash', color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=quality_data['timestamp'],
            y=quality_data['target'],
            name='Target',
            line=dict(dash='dot', color='green')
        ))
        
        fig.update_layout(
            title='Quality Control Chart',
            xaxis_title='Time',
            yaxis_title='Measurement',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Statistical Analysis")
        
        # Calculate statistics
        stats = quality_data['measurement'].describe()
        
        # Display statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Basic Statistics")
            st.write(pd.DataFrame({
                'Statistic': stats.index,
                'Value': stats.values
            }))
        
        with col2:
            st.write("### Process Capability Indices")
            cp = (quality_data['upper_limit'].iloc[0] - quality_data['lower_limit'].iloc[0]) / (6 * quality_data['measurement'].std())
            cpk = min(
                (quality_data['upper_limit'].iloc[0] - quality_data['measurement'].mean()) / (3 * quality_data['measurement'].std()),
                (quality_data['measurement'].mean() - quality_data['lower_limit'].iloc[0]) / (3 * quality_data['measurement'].std())
            )
            
            st.write(f"Cp: {cp:.2f}")
            st.write(f"Cpk: {cpk:.2f}")
            
            # Histogram with normal distribution
            fig = px.histogram(
                quality_data,
                x='measurement',
                nbins=30,
                title='Measurement Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Defect Analysis")
        
        # Sample defect data
        defect_types = {
            'Surface Scratches': 15,
            'Dimensional Error': 8,
            'Material Defect': 5,
            'Assembly Issue': 3,
            'Other': 2
        }
        
        # Pareto chart
        fig = go.Figure()
        sorted_defects = dict(sorted(defect_types.items(), key=lambda x: x[1], reverse=True))
        
        fig.add_trace(go.Bar(
            x=list(sorted_defects.keys()),
            y=list(sorted_defects.values()),
            name='Defect Count'
        ))
        
        # Calculate cumulative percentage
        total = sum(sorted_defects.values())
        cumulative = np.cumsum(list(sorted_defects.values())) / total * 100
        
        fig.add_trace(go.Scatter(
            x=list(sorted_defects.keys()),
            y=cumulative,
            name='Cumulative %',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Defect Pareto Chart',
            yaxis=dict(title='Defect Count'),
            yaxis2=dict(
                title='Cumulative %',
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab4:
        st.subheader("Quality Reports")
        
        report_type = st.selectbox(
            "Report Type",
            ["Daily Summary", "Weekly Trend", "Monthly Analysis", "Custom"]
        )
        
        if report_type == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
        
        # Report generation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Report"):
                st.info("Generating quality report...")
        with col2:
            if st.button("Export Data"):
                st.success("Data exported successfully") 