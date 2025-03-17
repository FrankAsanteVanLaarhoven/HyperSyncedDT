import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

class AnalyticsDashboard:
    def __init__(self):
        self.metrics = self._generate_sample_metrics()
        self.trends = self._generate_sample_trends()
        
    def _generate_sample_metrics(self) -> Dict:
        """Generate sample metrics data."""
        return {
            "production": {
                "total_output": np.random.randint(10000, 20000),
                "efficiency": np.random.uniform(0.85, 0.95),
                "quality_rate": np.random.uniform(0.92, 0.99),
                "uptime": np.random.uniform(0.90, 0.98)
            },
            "maintenance": {
                "scheduled": np.random.randint(10, 30),
                "unscheduled": np.random.randint(5, 15),
                "mtbf": np.random.uniform(150, 200),
                "mttr": np.random.uniform(2, 8)
            },
            "quality": {
                "defect_rate": np.random.uniform(0.01, 0.05),
                "first_pass_yield": np.random.uniform(0.90, 0.98),
                "customer_complaints": np.random.randint(0, 10),
                "scrap_rate": np.random.uniform(0.02, 0.06)
            }
        }
        
    def _generate_sample_trends(self) -> Dict[str, pd.DataFrame]:
        """Generate sample trend data."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        production_data = pd.DataFrame({
            'date': dates,
            'output': np.random.normal(15000, 1000, 30),
            'efficiency': np.random.normal(0.90, 0.05, 30),
            'quality_rate': np.random.normal(0.95, 0.02, 30)
        })
        
        maintenance_data = pd.DataFrame({
            'date': dates,
            'downtime_hours': np.random.normal(4, 1, 30),
            'maintenance_cost': np.random.normal(5000, 1000, 30),
            'repairs_completed': np.random.randint(5, 15, 30)
        })
        
        return {
            'production': production_data,
            'maintenance': maintenance_data
        }

def render_analytics_dashboard():
    """Render the analytics dashboard."""
    st.header("Analytics Dashboard", divider="rainbow")
    
    # Initialize dashboard
    dashboard = AnalyticsDashboard()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Dashboard Controls")
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last Quarter"]
        )
        metric_type = st.selectbox(
            "Metric Type",
            ["Production", "Maintenance", "Quality"]
        )
        
        st.divider()
        st.checkbox("Auto-refresh", value=True)
        if st.button("Export Report", use_container_width=True):
            st.info("Generating report...")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Production Analytics",
        "Maintenance Insights",
        "Quality Metrics"
    ])
    
    with tab1:
        st.subheader("Key Performance Indicators")
        
        # KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Output",
                f"{dashboard.metrics['production']['total_output']:,}",
                "+5.2%"
            )
        with col2:
            st.metric(
                "Efficiency",
                f"{dashboard.metrics['production']['efficiency']:.1%}",
                "+1.8%"
            )
        with col3:
            st.metric(
                "Quality Rate",
                f"{dashboard.metrics['quality']['first_pass_yield']:.1%}",
                "-0.5%"
            )
        with col4:
            st.metric(
                "Uptime",
                f"{dashboard.metrics['production']['uptime']:.1%}",
                "+0.7%"
            )
        
        # Production trend
        fig = px.line(
            dashboard.trends['production'],
            x='date',
            y=['output', 'efficiency', 'quality_rate'],
            title='Production Trends'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance overview
        st.subheader("Maintenance Overview")
        maintenance_data = pd.DataFrame({
            'Metric': ['MTBF', 'MTTR', 'Scheduled', 'Unscheduled'],
            'Value': [
                f"{dashboard.metrics['maintenance']['mtbf']:.1f} hrs",
                f"{dashboard.metrics['maintenance']['mttr']:.1f} hrs",
                dashboard.metrics['maintenance']['scheduled'],
                dashboard.metrics['maintenance']['unscheduled']
            ]
        })
        st.dataframe(maintenance_data, use_container_width=True)
    
    with tab2:
        st.subheader("Production Performance")
        
        # Production metrics over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dashboard.trends['production']['date'],
            y=dashboard.trends['production']['output'],
            name='Output',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=dashboard.trends['production']['date'],
            y=dashboard.trends['production']['efficiency'] * 20000,  # Scale for visualization
            name='Efficiency',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Production Output vs Efficiency',
            yaxis=dict(title='Units Produced'),
            yaxis2=dict(
                title='Efficiency',
                overlaying='y',
                side='right',
                tickformat=',.0%',
                range=[0, 1]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Maintenance Analytics")
        
        # Maintenance metrics
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                values=[
                    dashboard.metrics['maintenance']['scheduled'],
                    dashboard.metrics['maintenance']['unscheduled']
                ],
                names=['Scheduled', 'Unscheduled'],
                title='Maintenance Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                dashboard.trends['maintenance'],
                x='date',
                y='downtime_hours',
                title='Daily Downtime Hours'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance cost trend
        fig = px.line(
            dashboard.trends['maintenance'],
            x='date',
            y='maintenance_cost',
            title='Maintenance Cost Trend'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Quality Analysis")
        
        # Quality metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Defect Rate",
                f"{dashboard.metrics['quality']['defect_rate']:.1%}",
                "-0.3%"
            )
        with col2:
            st.metric(
                "First Pass Yield",
                f"{dashboard.metrics['quality']['first_pass_yield']:.1%}",
                "+1.2%"
            )
        with col3:
            st.metric(
                "Scrap Rate",
                f"{dashboard.metrics['quality']['scrap_rate']:.1%}",
                "-0.5%"
            )
        
        # Quality trend visualization
        quality_data = pd.DataFrame({
            'Category': ['Excellent', 'Good', 'Fair', 'Poor'],
            'Percentage': np.random.dirichlet(np.ones(4)) * 100
        })
        
        fig = px.bar(
            quality_data,
            x='Category',
            y='Percentage',
            title='Quality Distribution',
            color='Category'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Customer complaints
        st.subheader("Customer Feedback Analysis")
        complaints_data = pd.DataFrame({
            'Type': ['Product Quality', 'Packaging', 'Delivery', 'Other'],
            'Count': np.random.randint(0, 10, 4)
        })
        
        fig = px.pie(
            complaints_data,
            values='Count',
            names='Type',
            title='Customer Complaints Distribution'
        )
        st.plotly_chart(fig, use_container_width=True) 