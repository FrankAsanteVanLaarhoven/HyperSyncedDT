"""
Dashboard component for the HyperSyncDT platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

class Dashboard:
    def __init__(self):
        self.digital_twin = SynchronizedDigitalTwin()
        self.visualizer = MultiModalVisualizer()
    
    def generate_sample_metrics(self) -> Dict[str, pd.DataFrame]:
        """Generate sample metrics data."""
        np.random.seed(42)
        
        # Generate production metrics
        production = pd.DataFrame({
            'timestamp': pd.date_range('2024-03-17', periods=24, freq='H'),
            'units_produced': np.random.randint(80, 120, 24),
            'quality_rate': np.random.uniform(0.95, 0.99, 24),
            'efficiency': np.random.uniform(0.85, 0.95, 24),
            'downtime_minutes': np.random.randint(0, 30, 24)
        })
        
        # Generate machine metrics
        machines = pd.DataFrame({
            'machine_id': [f'Machine-{i}' for i in range(1, 6)],
            'status': np.random.choice(
                ['Running', 'Idle', 'Maintenance', 'Error'],
                5
            ),
            'temperature': np.random.uniform(60, 80, 5),
            'vibration': np.random.uniform(0.1, 0.5, 5),
            'power_consumption': np.random.uniform(75, 95, 5)
        })
        
        # Generate quality metrics
        quality = pd.DataFrame({
            'timestamp': pd.date_range('2024-03-17', periods=24, freq='H'),
            'defect_rate': np.random.uniform(0.01, 0.05, 24),
            'first_pass_yield': np.random.uniform(0.90, 0.98, 24),
            'customer_complaints': np.random.randint(0, 3, 24)
        })
        
        return {
            'production': production,
            'machines': machines,
            'quality': quality
        }

def render_dashboard():
    """Render the integrated dashboard."""
    st.header("Integrated Dashboard")
    
    # Initialize dashboard
    dashboard = Dashboard()
    
    # Sidebar controls
    st.sidebar.subheader("Dashboard Settings")
    view_mode = st.sidebar.selectbox(
        "View Mode",
        ["Overview", "Production", "Quality", "Maintenance"]
    )
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Hour", "Last 24 Hours", "Last Week", "Real-time"]
    )
    
    # Generate sample data
    metrics_data = dashboard.generate_sample_metrics()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Production Metrics",
        "Machine Status",
        "Quality Metrics"
    ])
    
    with tab1:
        st.subheader("Factory Overview")
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_efficiency = metrics_data['production']['efficiency'].iloc[-1]
            st.metric(
                "Overall Efficiency",
                f"{current_efficiency:.1%}",
                f"{(current_efficiency - metrics_data['production']['efficiency'].iloc[-2]):.1%}"
            )
        
        with col2:
            current_quality = metrics_data['production']['quality_rate'].iloc[-1]
            st.metric(
                "Quality Rate",
                f"{current_quality:.1%}",
                f"{(current_quality - metrics_data['production']['quality_rate'].iloc[-2]):.1%}"
            )
        
        with col3:
            units_produced = metrics_data['production']['units_produced'].sum()
            st.metric(
                "Units Produced",
                f"{units_produced:,}",
                f"{metrics_data['production']['units_produced'].iloc[-1] - metrics_data['production']['units_produced'].mean():.0f}"
            )
        
        with col4:
            machines_running = len(metrics_data['machines'][
                metrics_data['machines']['status'] == 'Running'
            ])
            st.metric(
                "Machines Running",
                f"{machines_running}/5",
                machines_running - 3
            )
        
        # Production timeline
        st.write("### Production Timeline")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics_data['production']['timestamp'],
            y=metrics_data['production']['units_produced'],
            name='Units Produced'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics_data['production']['timestamp'],
            y=metrics_data['production']['efficiency'] * 100,
            name='Efficiency %',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Production and Efficiency Over Time',
            yaxis=dict(title='Units Produced'),
            yaxis2=dict(
                title='Efficiency %',
                overlaying='y',
                side='right'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Machine status overview
        st.write("### Machine Status")
        
        # Create status cards for each machine
        cols = st.columns(5)
        for i, (_, machine) in enumerate(metrics_data['machines'].iterrows()):
            with cols[i]:
                status_color = {
                    'Running': 'green',
                    'Idle': 'orange',
                    'Maintenance': 'blue',
                    'Error': 'red'
                }[machine['status']]
                
                st.markdown(f"""
                <div style='padding: 10px; border: 2px solid {status_color}; border-radius: 5px; text-align: center;'>
                    <h4>{machine['machine_id']}</h4>
                    <p style='color: {status_color};'><strong>{machine['status']}</strong></p>
                    <p>Temp: {machine['temperature']:.1f}°C</p>
                    <p>Vib: {machine['vibration']:.2f}g</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Production Metrics")
        
        # Production rate over time
        st.write("### Production Rate")
        fig = px.line(
            metrics_data['production'],
            x='timestamp',
            y='units_produced',
            title='Units Produced Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency analysis
        st.write("### Efficiency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Efficiency distribution
            fig = px.histogram(
                metrics_data['production'],
                x='efficiency',
                title='Efficiency Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Downtime analysis
            fig = px.bar(
                metrics_data['production'],
                x='timestamp',
                y='downtime_minutes',
                title='Downtime Analysis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Machine Status")
        
        # Machine status summary
        status_counts = metrics_data['machines']['status'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Machine Status Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Temperature vs. Vibration
            fig = px.scatter(
                metrics_data['machines'],
                x='temperature',
                y='vibration',
                color='status',
                hover_data=['machine_id'],
                title='Temperature vs. Vibration'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Machine details table
        st.write("### Machine Details")
        st.dataframe(
            metrics_data['machines'].style.background_gradient(
                subset=['temperature', 'vibration', 'power_consumption']
            )
        )
    
    with tab4:
        st.subheader("Quality Metrics")
        
        # Quality trends
        st.write("### Quality Trends")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics_data['quality']['timestamp'],
            y=metrics_data['quality']['defect_rate'] * 100,
            name='Defect Rate %'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics_data['quality']['timestamp'],
            y=metrics_data['quality']['first_pass_yield'] * 100,
            name='First Pass Yield %'
        ))
        
        fig.update_layout(
            title='Quality Metrics Over Time',
            yaxis_title='Percentage'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_defect_rate = metrics_data['quality']['defect_rate'].mean()
            st.metric(
                "Average Defect Rate",
                f"{avg_defect_rate:.1%}",
                f"{(avg_defect_rate - metrics_data['quality']['defect_rate'].iloc[0]):.1%}"
            )
        
        with col2:
            avg_fpy = metrics_data['quality']['first_pass_yield'].mean()
            st.metric(
                "Average First Pass Yield",
                f"{avg_fpy:.1%}",
                f"{(avg_fpy - metrics_data['quality']['first_pass_yield'].iloc[0]):.1%}"
            )
        
        with col3:
            total_complaints = metrics_data['quality']['customer_complaints'].sum()
            st.metric(
                "Total Customer Complaints",
                total_complaints,
                -metrics_data['quality']['customer_complaints'].iloc[-1]
            )
        
        # Customer complaints trend
        st.write("### Customer Complaints Trend")
        fig = px.bar(
            metrics_data['quality'],
            x='timestamp',
            y='customer_complaints',
            title='Customer Complaints Over Time'
        )
        st.plotly_chart(fig, use_container_width=True) 