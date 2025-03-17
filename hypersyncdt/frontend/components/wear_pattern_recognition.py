import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

class WearPatternAnalyzer:
    def __init__(self):
        self.digital_twin = SynchronizedDigitalTwin()
        self.visualizer = MultiModalVisualizer()
    
    def generate_sample_wear_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample wear pattern data."""
        np.random.seed(42)
        
        # Generate equipment wear data
        n_equipment = 10
        equipment_data = pd.DataFrame({
            'equipment_id': [f'EQ-{i:03d}' for i in range(1, n_equipment + 1)],
            'type': np.random.choice(
                ['Bearing', 'Gear', 'Belt', 'Chain', 'Shaft'],
                n_equipment
            ),
            'installation_date': pd.date_range(
                start='2023-01-01',
                periods=n_equipment,
                freq='M'
            ),
            'operating_hours': np.random.uniform(1000, 5000, n_equipment),
            'wear_level': np.random.uniform(10, 90, n_equipment),
            'expected_lifetime': np.random.uniform(5000, 10000, n_equipment)
        })
        
        # Generate wear measurements over time
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=1000,
            freq='H'
        )
        
        wear_measurements = pd.DataFrame()
        for _, eq in equipment_data.iterrows():
            base_wear = eq['wear_level']
            wear_rate = np.random.uniform(0.001, 0.005)
            
            eq_data = pd.DataFrame({
                'timestamp': timestamps,
                'equipment_id': eq['equipment_id'],
                'wear_measurement': base_wear + np.cumsum(
                    np.random.normal(wear_rate, wear_rate/10, 1000)
                ),
                'temperature': 60 + np.sin(np.linspace(0, 8*np.pi, 1000)) * 5 + np.random.normal(0, 1, 1000),
                'vibration': 0.2 + np.sin(np.linspace(0, 4*np.pi, 1000)) * 0.1 + np.random.normal(0, 0.02, 1000)
            })
            wear_measurements = pd.concat([wear_measurements, eq_data])
        
        # Generate pattern classifications
        n_patterns = 50
        pattern_data = pd.DataFrame({
            'pattern_id': [f'P-{i:03d}' for i in range(1, n_patterns + 1)],
            'equipment_id': np.random.choice(equipment_data['equipment_id'], n_patterns),
            'detection_date': pd.date_range(
                start='2024-01-01',
                periods=n_patterns,
                freq='D'
            ),
            'pattern_type': np.random.choice(
                ['Linear', 'Exponential', 'Cyclic', 'Step', 'Random'],
                n_patterns
            ),
            'severity': np.random.choice(['Low', 'Medium', 'High'], n_patterns),
            'confidence': np.random.uniform(0.6, 0.99, n_patterns)
        })
        
        return {
            'equipment': equipment_data,
            'measurements': wear_measurements,
            'patterns': pattern_data
        }

def render_wear_pattern_recognition():
    """Render the wear pattern recognition dashboard."""
    st.header("Wear Pattern Recognition Dashboard")
    
    # Initialize analyzer
    analyzer = WearPatternAnalyzer()
    
    # Sidebar controls
    st.sidebar.subheader("Analysis Settings")
    equipment_type = st.sidebar.selectbox(
        "Equipment Type",
        ["All", "Bearing", "Gear", "Belt", "Chain", "Shaft"]
    )
    
    pattern_type = st.sidebar.selectbox(
        "Pattern Type",
        ["All", "Linear", "Exponential", "Cyclic", "Step", "Random"]
    )
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Day", "Last Week", "Last Month", "Last Quarter"]
    )
    
    # Generate sample data
    wear_data = analyzer.generate_sample_wear_data()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Pattern Overview",
        "Wear Analysis",
        "Pattern Detection",
        "Predictions"
    ])
    
    with tab1:
        st.subheader("Pattern Overview")
        
        # Pattern statistics
        pattern_data = wear_data['patterns']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_patterns = len(pattern_data)
            st.metric(
                "Total Patterns",
                total_patterns,
                None
            )
        
        with col2:
            high_severity = len(pattern_data[pattern_data['severity'] == 'High'])
            st.metric(
                "High Severity",
                high_severity,
                None
            )
        
        with col3:
            avg_confidence = pattern_data['confidence'].mean()
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.1%}",
                None
            )
        
        with col4:
            pattern_types = pattern_data['pattern_type'].nunique()
            st.metric(
                "Pattern Types",
                pattern_types,
                None
            )
        
        # Pattern distribution
        st.write("### Pattern Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                pattern_data,
                names='pattern_type',
                title='Pattern Type Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                pattern_data,
                names='severity',
                title='Severity Distribution',
                color='severity',
                color_discrete_map={
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Pattern timeline
        st.write("### Pattern Detection Timeline")
        
        fig = px.scatter(
            pattern_data,
            x='detection_date',
            y='equipment_id',
            color='severity',
            size='confidence',
            hover_data=['pattern_type', 'confidence'],
            title='Pattern Detection Timeline'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Wear Analysis")
        
        # Equipment selector
        selected_equipment = st.selectbox(
            "Select Equipment",
            wear_data['equipment']['equipment_id']
        )
        
        # Filter measurements for selected equipment
        equipment_measurements = wear_data['measurements'][
            wear_data['measurements']['equipment_id'] == selected_equipment
        ].copy()
        
        # Wear progression
        st.write("### Wear Progression")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equipment_measurements['timestamp'],
            y=equipment_measurements['wear_measurement'],
            name='Wear Level'
        ))
        
        fig.update_layout(title=f'Wear Progression for {selected_equipment}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.write("### Parameter Correlations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                equipment_measurements,
                x='temperature',
                y='wear_measurement',
                title='Wear vs Temperature'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                equipment_measurements,
                x='vibration',
                y='wear_measurement',
                title='Wear vs Vibration'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Pattern Detection")
        
        # Pattern detection results
        st.write("### Recent Pattern Detections")
        
        recent_patterns = pattern_data.sort_values('detection_date', ascending=False).head(10)
        
        for _, pattern in recent_patterns.iterrows():
            severity_color = {
                'Low': 'green',
                'Medium': 'orange',
                'High': 'red'
            }[pattern['severity']]
            
            st.markdown(f"""
            <div style='padding: 10px; border-left: 5px solid {severity_color}; margin: 10px 0;'>
                <strong>{pattern['equipment_id']}</strong> - {pattern['pattern_type']} Pattern<br>
                Detected: {pattern['detection_date'].strftime('%Y-%m-%d')}<br>
                Severity: {pattern['severity']}<br>
                Confidence: {pattern['confidence']:.1%}
            </div>
            """, unsafe_allow_html=True)
        
        # Pattern characteristics
        st.write("### Pattern Characteristics")
        
        fig = px.box(
            pattern_data,
            x='pattern_type',
            y='confidence',
            color='severity',
            title='Pattern Confidence by Type and Severity'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Wear Predictions")
        
        # Generate wear predictions
        equipment_data = wear_data['equipment'].copy()
        equipment_data['predicted_replacement'] = pd.date_range(
            start=datetime.now(),
            periods=len(equipment_data),
            freq='D'
        ) + pd.to_timedelta(
            (equipment_data['expected_lifetime'] - equipment_data['operating_hours']) / 24,
            unit='H'
        )
        
        equipment_data['days_until_replacement'] = (
            equipment_data['predicted_replacement'] - datetime.now()
        ).dt.days
        
        equipment_data['risk_level'] = np.where(
            equipment_data['wear_level'] > 80,
            'High',
            np.where(
                equipment_data['wear_level'] > 60,
                'Medium',
                'Low'
            )
        )
        
        # Risk matrix
        st.write("### Risk Assessment Matrix")
        
        fig = px.scatter(
            equipment_data,
            x='wear_level',
            y='days_until_replacement',
            color='risk_level',
            hover_data=['equipment_id', 'type'],
            title='Risk Assessment Matrix'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Replacement schedule
        st.write("### Recommended Replacement Schedule")
        
        schedule = equipment_data[[
            'equipment_id', 'type', 'wear_level',
            'days_until_replacement', 'risk_level'
        ]]
        schedule = schedule.sort_values('days_until_replacement')
        
        st.dataframe(
            schedule.style.background_gradient(
                subset=['wear_level', 'days_until_replacement']
            ),
            hide_index=True
        )
        
        # Wear rate analysis
        st.write("### Wear Rate Analysis")
        
        # Calculate wear rates
        wear_rates = []
        for eq_id in equipment_data['equipment_id']:
            eq_measurements = wear_data['measurements'][
                wear_data['measurements']['equipment_id'] == eq_id
            ]
            wear_rate = (
                eq_measurements['wear_measurement'].iloc[-1] -
                eq_measurements['wear_measurement'].iloc[0]
            ) / len(eq_measurements)
            wear_rates.append({
                'equipment_id': eq_id,
                'wear_rate': wear_rate
            })
        
        wear_rate_df = pd.DataFrame(wear_rates)
        
        fig = px.bar(
            wear_rate_df,
            x='equipment_id',
            y='wear_rate',
            title='Wear Rate by Equipment'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_wear_pattern_recognition() 