import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

class PredictiveMaintenanceAnalyzer:
    def __init__(self):
        self.digital_twin = SynchronizedDigitalTwin()
        self.visualizer = MultiModalVisualizer()
    
    def generate_sample_maintenance_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample maintenance data."""
        np.random.seed(42)
        
        # Generate equipment data
        n_equipment = 10
        equipment_data = pd.DataFrame({
            'equipment_id': [f'EQ-{i:03d}' for i in range(1, n_equipment + 1)],
            'type': np.random.choice(
                ['Pump', 'Motor', 'Compressor', 'Valve', 'Heat Exchanger'],
                n_equipment
            ),
            'installation_date': pd.date_range(
                start='2023-01-01',
                periods=n_equipment,
                freq='M'
            ),
            'last_maintenance': pd.date_range(
                start='2024-01-01',
                periods=n_equipment,
                freq='M'
            ),
            'condition_score': np.random.uniform(60, 95, n_equipment),
            'remaining_life': np.random.uniform(30, 365, n_equipment)
        })
        
        # Generate sensor readings
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=1000,
            freq='H'
        )
        
        sensor_data = pd.DataFrame()
        for eq in equipment_data['equipment_id']:
            base_temp = np.random.uniform(60, 80)
            base_vibration = np.random.uniform(0.1, 0.3)
            
            eq_data = pd.DataFrame({
                'timestamp': timestamps,
                'equipment_id': eq,
                'temperature': base_temp + np.sin(np.linspace(0, 8*np.pi, 1000)) * 5 + np.random.normal(0, 1, 1000),
                'vibration': base_vibration + np.sin(np.linspace(0, 4*np.pi, 1000)) * 0.1 + np.random.normal(0, 0.02, 1000),
                'pressure': 100 + np.sin(np.linspace(0, 6*np.pi, 1000)) * 10 + np.random.normal(0, 2, 1000),
                'power': 75 + np.sin(np.linspace(0, 2*np.pi, 1000)) * 15 + np.random.normal(0, 3, 1000)
            })
            sensor_data = pd.concat([sensor_data, eq_data])
        
        # Generate maintenance history
        n_records = 50
        maintenance_history = pd.DataFrame({
            'maintenance_id': [f'M-{i:04d}' for i in range(1, n_records + 1)],
            'equipment_id': np.random.choice(equipment_data['equipment_id'], n_records),
            'maintenance_date': pd.date_range(
                start='2023-01-01',
                periods=n_records,
                freq='W'
            ),
            'maintenance_type': np.random.choice(
                ['Preventive', 'Corrective', 'Predictive'],
                n_records
            ),
            'cost': np.random.uniform(1000, 5000, n_records),
            'duration_hours': np.random.uniform(2, 24, n_records)
        })
        
        return {
            'equipment': equipment_data,
            'sensors': sensor_data,
            'maintenance': maintenance_history
        }

def render_predictive_maintenance():
    """Render the predictive maintenance dashboard."""
    st.header("Predictive Maintenance Dashboard")
    
    # Initialize analyzer
    analyzer = PredictiveMaintenanceAnalyzer()
    
    # Sidebar controls
    st.sidebar.subheader("Analysis Settings")
    equipment_type = st.sidebar.selectbox(
        "Equipment Type",
        ["All", "Pump", "Motor", "Compressor", "Valve", "Heat Exchanger"]
    )
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Week", "Last Month", "Last Quarter", "Last Year"]
    )
    
    risk_threshold = st.sidebar.slider(
        "Risk Threshold",
        min_value=0,
        max_value=100,
        value=70,
        help="Equipment with condition scores below this threshold will be flagged"
    )
    
    # Generate sample data
    maintenance_data = analyzer.generate_sample_maintenance_data()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Equipment Overview",
        "Sensor Analysis",
        "Maintenance History",
        "Predictions"
    ])
    
    with tab1:
        st.subheader("Equipment Status")
        
        # Equipment condition summary
        equipment_data = maintenance_data['equipment']
        critical_count = len(equipment_data[equipment_data['condition_score'] < risk_threshold])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Equipment",
                len(equipment_data),
                None
            )
        
        with col2:
            st.metric(
                "Critical Equipment",
                critical_count,
                None
            )
        
        with col3:
            avg_condition = equipment_data['condition_score'].mean()
            st.metric(
                "Average Condition",
                f"{avg_condition:.1f}%",
                None
            )
        
        with col4:
            avg_life = equipment_data['remaining_life'].mean()
            st.metric(
                "Avg Remaining Life",
                f"{avg_life:.0f} days",
                None
            )
        
        # Equipment condition distribution
        st.write("### Equipment Condition Distribution")
        
        fig = px.histogram(
            equipment_data,
            x='condition_score',
            color='type',
            title='Equipment Condition Distribution by Type'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Equipment details table
        st.write("### Equipment Details")
        st.dataframe(
            equipment_data.style.background_gradient(
                subset=['condition_score', 'remaining_life']
            ),
            hide_index=True
        )
    
    with tab2:
        st.subheader("Sensor Data Analysis")
        
        # Equipment selector for sensor data
        selected_equipment = st.selectbox(
            "Select Equipment",
            maintenance_data['equipment']['equipment_id']
        )
        
        # Filter sensor data for selected equipment
        sensor_data = maintenance_data['sensors'][
            maintenance_data['sensors']['equipment_id'] == selected_equipment
        ].copy()
        
        # Sensor readings over time
        st.write("### Sensor Readings")
        
        fig = go.Figure()
        sensors = ['temperature', 'vibration', 'pressure', 'power']
        
        for sensor in sensors:
            fig.add_trace(go.Scatter(
                x=sensor_data['timestamp'],
                y=sensor_data[sensor],
                name=sensor.title()
            ))
        
        fig.update_layout(title=f'Sensor Readings for {selected_equipment}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensor correlations
        st.write("### Sensor Correlations")
        
        correlation_matrix = sensor_data[sensors].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=sensors,
            y=sensors,
            colorscale='RdBu'
        ))
        
        fig.update_layout(title='Sensor Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Maintenance History")
        
        # Maintenance statistics
        maintenance_data_df = maintenance_data['maintenance']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_cost = maintenance_data_df['cost'].sum()
            st.metric(
                "Total Maintenance Cost",
                f"${total_cost:,.0f}",
                None
            )
        
        with col2:
            avg_duration = maintenance_data_df['duration_hours'].mean()
            st.metric(
                "Average Duration",
                f"{avg_duration:.1f} hours",
                None
            )
        
        with col3:
            maintenance_types = maintenance_data_df['maintenance_type'].value_counts()
            st.metric(
                "Most Common Type",
                maintenance_types.index[0],
                None
            )
        
        # Maintenance history timeline
        st.write("### Maintenance Timeline")
        
        fig = px.scatter(
            maintenance_data_df,
            x='maintenance_date',
            y='equipment_id',
            color='maintenance_type',
            size='cost',
            hover_data=['duration_hours', 'cost'],
            title='Maintenance Events Timeline'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance cost analysis
        st.write("### Maintenance Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                maintenance_data_df,
                x='maintenance_type',
                y='cost',
                title='Maintenance Cost by Type'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly_costs = maintenance_data_df.set_index('maintenance_date')['cost'].resample('M').sum()
            fig = px.line(
                monthly_costs,
                title='Monthly Maintenance Costs'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Maintenance Predictions")
        
        # Generate some predicted failures
        predictions = maintenance_data['equipment'].copy()
        predictions['predicted_failure'] = pd.date_range(
            start=datetime.now(),
            periods=len(predictions),
            freq='D'
        ) + pd.to_timedelta(predictions['remaining_life'], unit='D')
        
        predictions['days_until_failure'] = (predictions['predicted_failure'] - datetime.now()).dt.days
        predictions['risk_level'] = np.where(
            predictions['condition_score'] < risk_threshold,
            'High',
            np.where(
                predictions['condition_score'] < risk_threshold + 15,
                'Medium',
                'Low'
            )
        )
        
        # Risk assessment
        st.write("### Risk Assessment")
        
        fig = px.scatter(
            predictions,
            x='condition_score',
            y='days_until_failure',
            color='risk_level',
            hover_data=['equipment_id', 'type'],
            title='Risk Assessment Matrix'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance schedule
        st.write("### Recommended Maintenance Schedule")
        
        schedule = predictions[['equipment_id', 'type', 'condition_score', 'days_until_failure', 'risk_level']]
        schedule = schedule.sort_values('days_until_failure')
        
        st.dataframe(
            schedule.style.background_gradient(
                subset=['condition_score', 'days_until_failure']
            ),
            hide_index=True
        )
        
        # Maintenance cost projections
        st.write("### Cost Projections")
        
        # Generate projected costs
        n_months = 12
        projected_dates = pd.date_range(
            start=datetime.now(),
            periods=n_months,
            freq='M'
        )
        
        projected_costs = pd.DataFrame({
            'date': projected_dates,
            'projected_cost': np.random.uniform(8000, 12000, n_months),
            'actual_cost': np.random.uniform(7000, 13000, n_months)
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=projected_dates,
            y=projected_costs['projected_cost'],
            name='Projected Cost',
            line=dict(dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=projected_dates,
            y=projected_costs['actual_cost'],
            name='Actual Cost'
        ))
        
        fig.update_layout(
            title='Maintenance Cost Projections',
            yaxis_title='Cost ($)'
        )
        st.plotly_chart(fig, use_container_width=True) 