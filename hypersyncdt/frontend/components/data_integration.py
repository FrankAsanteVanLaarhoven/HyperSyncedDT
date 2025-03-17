import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional

class DataIntegrationManager:
    def __init__(self):
        self.supported_sources = {
            "Database": ["PostgreSQL", "MySQL", "MongoDB", "TimescaleDB"],
            "File": ["CSV", "Excel", "JSON", "Parquet"],
            "API": ["REST", "GraphQL", "MQTT", "OPC UA"],
            "Streaming": ["Kafka", "RabbitMQ", "Redis Streams"]
        }
        
    def generate_sample_integration_status(self) -> pd.DataFrame:
        """Generate sample integration status data."""
        sources = []
        for category, types in self.supported_sources.items():
            for type_ in types:
                sources.append({
                    "source_type": category,
                    "source_name": type_,
                    "status": np.random.choice(["Active", "Inactive", "Error"], p=[0.7, 0.2, 0.1]),
                    "last_sync": datetime.now() - timedelta(minutes=np.random.randint(0, 120)),
                    "records_processed": np.random.randint(1000, 100000),
                    "success_rate": np.random.uniform(0.95, 1.0),
                    "latency_ms": np.random.randint(10, 500)
                })
        return pd.DataFrame(sources)

def render_data_integration():
    """Render the data integration dashboard."""
    st.header("Data Integration Hub", divider="rainbow")
    
    # Initialize the manager
    manager = DataIntegrationManager()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Integration Controls")
        source_type = st.selectbox(
            "Source Type",
            options=list(manager.supported_sources.keys())
        )
        source_name = st.selectbox(
            "Source Name",
            options=manager.supported_sources[source_type]
        )
        
        st.divider()
        st.subheader("Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Connection", use_container_width=True):
                st.success("Connection test successful!")
        with col2:
            if st.button("Sync Now", use_container_width=True):
                st.info("Synchronization started...")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "Integration Status",
        "Data Mapping",
        "Validation Rules",
        "Monitoring"
    ])
    
    # Get sample data
    status_df = manager.generate_sample_integration_status()
    
    with tab1:
        st.subheader("Integration Status Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            active_count = sum(status_df['status'] == 'Active')
            st.metric("Active Integrations", active_count)
        with col2:
            avg_success = status_df['success_rate'].mean() * 100
            st.metric("Average Success Rate", f"{avg_success:.1f}%")
        with col3:
            total_records = status_df['records_processed'].sum()
            st.metric("Total Records Processed", f"{total_records:,}")
        with col4:
            avg_latency = status_df['latency_ms'].mean()
            st.metric("Average Latency", f"{avg_latency:.0f}ms")
        
        # Status table
        st.dataframe(
            status_df.style.background_gradient(subset=['success_rate']),
            hide_index=True,
            use_container_width=True
        )
        
        # Success rate by source type
        fig = px.bar(
            status_df.groupby('source_type')['success_rate'].mean().reset_index(),
            x='source_type',
            y='success_rate',
            title="Success Rate by Source Type",
            color='source_type'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Data Mapping Configuration")
        
        # Sample mapping interface
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Source Field")
            st.selectbox("Data Type", ["String", "Integer", "Float", "DateTime", "Boolean"])
        with col2:
            st.text_input("Target Field")
            st.selectbox("Transformation", ["None", "Upper Case", "Lower Case", "Round", "Format Date"])
        
        st.divider()
        st.button("Save Mapping", use_container_width=True)
    
    with tab3:
        st.subheader("Data Validation Rules")
        
        # Sample validation rules interface
        rule_type = st.selectbox(
            "Rule Type",
            ["Range Check", "Null Check", "Format Check", "Custom Logic"]
        )
        
        if rule_type == "Range Check":
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Minimum Value")
            with col2:
                st.number_input("Maximum Value")
        
        st.text_area("Rule Description")
        st.button("Add Rule", use_container_width=True)
    
    with tab4:
        st.subheader("Integration Monitoring")
        
        # Sample monitoring visualizations
        fig = go.Figure()
        
        # Add traces for different metrics
        times = pd.date_range(end=datetime.now(), periods=24, freq='H')
        
        fig.add_trace(go.Scatter(
            x=times,
            y=np.random.normal(95, 2, 24),
            name='Success Rate',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=np.random.normal(200, 50, 24),
            name='Latency (ms)',
            line=dict(color='blue'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Integration Performance Over Time',
            yaxis=dict(title='Success Rate (%)', range=[0, 100]),
            yaxis2=dict(title='Latency (ms)', overlaying='y', side='right'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True) 