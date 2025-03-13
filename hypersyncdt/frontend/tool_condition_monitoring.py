import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn

class ToolConditionMonitor:
    def __init__(self):
        self.data_path = Path("data/tool_monitoring")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.setup_monitor()
        
    def setup_monitor(self):
        """Initialize monitoring components"""
        if 'monitoring_state' not in st.session_state:
            st.session_state.monitoring_state = {
                'active': False,
                'current_tool': None,
                'alerts': [],
                'history': []
            }
        
        # Initialize ML model for condition assessment
        self.model = self._initialize_model()
    
    def _initialize_model(self) -> nn.Module:
        """Initialize the ML model for condition assessment"""
        model = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        
        # Use MPS if available (for M1 Macs)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            model = model.to(self.device)
        else:
            self.device = torch.device("cpu")
            
        return model
    
    def assess_condition(self, metrics: Dict[str, float]) -> Dict[str, any]:
        """Assess tool condition based on current metrics"""
        # Convert metrics to tensor
        inputs = torch.tensor([
            metrics['vibration'],
            metrics['temperature'],
            metrics['acoustic'],
            metrics['power'],
            metrics['speed'],
            metrics['feed_rate']
        ], dtype=torch.float32).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(inputs.unsqueeze(0))
            condition_scores = torch.softmax(outputs, dim=1)[0]
        
        # Map scores to conditions
        conditions = ['Optimal', 'Fair', 'Warning', 'Critical']
        condition_dict = {
            cond: score.item() 
            for cond, score in zip(conditions, condition_scores)
        }
        
        # Determine overall condition
        overall_condition = conditions[torch.argmax(condition_scores).item()]
        
        return {
            'scores': condition_dict,
            'overall': overall_condition
        }
    
    def generate_alerts(self, metrics: Dict[str, float], 
                       thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, any]]:
        """Generate alerts based on metric thresholds"""
        alerts = []
        
        for metric, value in metrics.items():
            if metric in thresholds:
                if value > thresholds[metric]['critical']:
                    alerts.append({
                        'level': 'Critical',
                        'metric': metric,
                        'value': value,
                        'threshold': thresholds[metric]['critical'],
                        'timestamp': datetime.now().isoformat()
                    })
                elif value > thresholds[metric]['warning']:
                    alerts.append({
                        'level': 'Warning',
                        'metric': metric,
                        'value': value,
                        'threshold': thresholds[metric]['warning'],
                        'timestamp': datetime.now().isoformat()
                    })
        
        return alerts
    
    def save_monitoring_data(self, tool_id: str, data: Dict[str, any]):
        """Save monitoring data to file"""
        file_path = self.data_path / f"{tool_id}_monitoring.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
            existing_data['history'].append(data)
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
        else:
            with open(file_path, 'w') as f:
                json.dump({'tool_id': tool_id, 'history': [data]}, f, indent=4)

def render_tool_condition_monitoring():
    """Render the Tool Condition Monitoring interface"""
    st.title("🔍 Tool Condition Monitoring")
    
    # Initialize monitor if not in session state
    if 'condition_monitor' not in st.session_state:
        st.session_state.condition_monitor = ToolConditionMonitor()
    
    # Create tabs for different monitoring aspects
    tab1, tab2, tab3, tab4 = st.tabs([
        "Real-time Monitoring",
        "Condition Analysis",
        "Alert Management",
        "Historical Data"
    ])
    
    with tab1:
        st.subheader("Real-time Tool Monitoring")
        
        # Tool selection and monitoring controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            tool_id = st.text_input("Tool ID", "TOOL_001")
            
            # Monitoring toggle
            if st.button("Start Monitoring" if not st.session_state.monitoring_state['active'] 
                        else "Stop Monitoring"):
                st.session_state.monitoring_state['active'] = \
                    not st.session_state.monitoring_state['active']
                st.session_state.monitoring_state['current_tool'] = \
                    tool_id if st.session_state.monitoring_state['active'] else None
        
        with col2:
            st.metric("Monitoring Status", 
                     "Active" if st.session_state.monitoring_state['active'] else "Inactive",
                     delta="Online" if st.session_state.monitoring_state['active'] else "Offline")
        
        if st.session_state.monitoring_state['active']:
            # Simulate real-time data
            current_metrics = {
                'vibration': np.random.normal(0.5, 0.1),
                'temperature': np.random.normal(65, 5),
                'acoustic': np.random.normal(70, 3),
                'power': np.random.normal(80, 2),
                'speed': np.random.normal(1000, 50),
                'feed_rate': np.random.normal(100, 5)
            }
            
            # Display current metrics
            cols = st.columns(3)
            for i, (metric, value) in enumerate(current_metrics.items()):
                with cols[i % 3]:
                    st.metric(
                        metric.replace('_', ' ').title(),
                        f"{value:.1f}",
                        f"{np.random.normal(0, 0.1):.2f}"
                    )
            
            # Assess condition
            condition_results = st.session_state.condition_monitor.assess_condition(
                current_metrics
            )
            
            # Display condition gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = max(condition_results['scores'].values()) * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ]
                },
                title = {'text': f"Overall Condition: {condition_results['overall']}"}
            ))
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Condition Analysis")
        
        # Historical condition data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
        condition_history = pd.DataFrame({
            'timestamp': dates,
            'vibration': np.random.normal(0.5, 0.1, 100),
            'temperature': np.random.normal(65, 5, 100),
            'acoustic': np.random.normal(70, 3, 100),
            'condition_score': np.cumsum(np.random.normal(0, 0.01, 100)) + 0.8
        })
        
        # Create trend visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Condition Score Trend", "Key Metrics"),
            vertical_spacing=0.15
        )
        
        # Add condition score trend
        fig.add_trace(
            go.Scatter(
                x=condition_history['timestamp'],
                y=condition_history['condition_score'],
                name="Condition Score",
                line=dict(color='rgba(100, 255, 200, 0.8)')
            ),
            row=1, col=1
        )
        
        # Add key metrics
        for metric in ['vibration', 'temperature', 'acoustic']:
            fig.add_trace(
                go.Scatter(
                    x=condition_history['timestamp'],
                    y=condition_history[metric],
                    name=metric.title(),
                    line=dict(width=1)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Condition breakdown
        st.subheader("Condition Analysis Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Component health scores
            components = {
                'Cutting Edge': 92,
                'Tool Holder': 88,
                'Cooling System': 95,
                'Drive System': 90
            }
            
            for component, score in components.items():
                st.metric(component, f"{score}%",
                         "Optimal" if score >= 90 else "Good")
        
        with col2:
            # Wear analysis
            st.markdown("""
            #### Wear Analysis
            - Primary wear type: Flank wear
            - Wear rate: 0.02 mm/hour
            - Estimated remaining life: 42 hours
            """)
    
    with tab3:
        st.subheader("Alert Management")
        
        # Alert configuration
        st.markdown("### Alert Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            warning_temp = st.number_input("Temperature Warning (°C)", 50.0, 100.0, 70.0)
            warning_vib = st.number_input("Vibration Warning (mm/s)", 0.1, 2.0, 0.8)
        
        with col2:
            critical_temp = st.number_input("Temperature Critical (°C)", 60.0, 120.0, 85.0)
            critical_vib = st.number_input("Vibration Critical (mm/s)", 0.5, 3.0, 1.2)
        
        # Configure thresholds
        thresholds = {
            'temperature': {'warning': warning_temp, 'critical': critical_temp},
            'vibration': {'warning': warning_vib, 'critical': critical_vib}
        }
        
        # Generate and display alerts
        if st.session_state.monitoring_state['active']:
            alerts = st.session_state.condition_monitor.generate_alerts(
                current_metrics, thresholds
            )
            
            if alerts:
                for alert in alerts:
                    if alert['level'] == 'Critical':
                        st.error(
                            f"🚨 {alert['level']}: {alert['metric']} is {alert['value']:.1f} "
                            f"(threshold: {alert['threshold']:.1f})"
                        )
                    else:
                        st.warning(
                            f"⚠️ {alert['level']}: {alert['metric']} is {alert['value']:.1f} "
                            f"(threshold: {alert['threshold']:.1f})"
                        )
            else:
                st.success("✅ No active alerts")
    
    with tab4:
        st.subheader("Historical Data Analysis")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=7)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now()
            )
        
        # Generate sample historical data
        historical_data = pd.DataFrame({
            'timestamp': pd.date_range(start=start_date, end=end_date, freq='1h'),
            'condition_score': np.random.normal(0.85, 0.05, 
                                             len(pd.date_range(start=start_date, 
                                                             end=end_date, freq='1h'))),
            'temperature': np.random.normal(65, 5, 
                                         len(pd.date_range(start=start_date, 
                                                         end=end_date, freq='1h'))),
            'vibration': np.random.normal(0.5, 0.1, 
                                        len(pd.date_range(start=start_date, 
                                                        end=end_date, freq='1h')))
        })
        
        # Create historical visualization
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Overall Condition", "Temperature", "Vibration"),
            vertical_spacing=0.1
        )
        
        # Add condition score
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['condition_score'],
                name="Condition Score",
                line=dict(color='rgba(100, 255, 200, 0.8)')
            ),
            row=1, col=1
        )
        
        # Add temperature
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['temperature'],
                name="Temperature",
                line=dict(color='rgba(255, 100, 100, 0.8)')
            ),
            row=2, col=1
        )
        
        # Add vibration
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['vibration'],
                name="Vibration",
                line=dict(color='rgba(100, 100, 255, 0.8)')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        
        stats_df = pd.DataFrame({
            'Metric': ['Condition Score', 'Temperature', 'Vibration'],
            'Mean': [
                historical_data['condition_score'].mean(),
                historical_data['temperature'].mean(),
                historical_data['vibration'].mean()
            ],
            'Std Dev': [
                historical_data['condition_score'].std(),
                historical_data['temperature'].std(),
                historical_data['vibration'].std()
            ],
            'Min': [
                historical_data['condition_score'].min(),
                historical_data['temperature'].min(),
                historical_data['vibration'].min()
            ],
            'Max': [
                historical_data['condition_score'].max(),
                historical_data['temperature'].max(),
                historical_data['vibration'].max()
            ]
        })
        
        st.dataframe(stats_df)

if __name__ == "__main__":
    render_tool_condition_monitoring() 