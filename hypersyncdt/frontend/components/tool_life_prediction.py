import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ToolLifeConfig:
    tool_id: str
    material: str
    operation_type: str
    cutting_parameters: Dict[str, float]
    expected_life: float
    maintenance_threshold: float
    replacement_threshold: float

class ToolLifePredictor:
    def __init__(self):
        self.data_path = Path("data/tool_life")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.setup_predictor()
    
    def setup_predictor(self):
        """Initialize prediction components"""
        if 'prediction_state' not in st.session_state:
            st.session_state.prediction_state = {
                'current_tool': None,
                'predictions': [],
                'history': []
            }
        
        # Initialize ML model for life prediction
        self.model = self._initialize_model()
    
    def _initialize_model(self) -> nn.Module:
        """Initialize the ML model for life prediction"""
        model = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Use MPS if available (for M1 Macs)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            model = model.to(self.device)
        else:
            self.device = torch.device("cpu")
            
        return model
    
    def predict_remaining_life(self, current_wear: float, 
                             cutting_speed: float,
                             feed_rate: float,
                             depth_of_cut: float,
                             material_hardness: float,
                             temperature: float,
                             vibration: float) -> Dict[str, float]:
        """Predict remaining tool life based on current conditions"""
        # Prepare input tensor
        inputs = torch.tensor([
            current_wear,
            cutting_speed,
            feed_rate,
            depth_of_cut,
            material_hardness,
            temperature,
            vibration
        ], dtype=torch.float32).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            remaining_hours = self.model(inputs.unsqueeze(0)).item()
            confidence = self._calculate_prediction_confidence(
                current_wear, cutting_speed
            )
        
        return {
            'remaining_hours': max(0, remaining_hours),
            'confidence': confidence,
            'optimal_parameters': self._optimize_cutting_parameters(current_wear)
        }
    
    def _calculate_prediction_confidence(self, wear: float, speed: float) -> float:
        """Calculate confidence score for the prediction"""
        # Simple confidence calculation based on wear and speed
        wear_factor = 1 - (wear / 1.0)  # Assume max wear is 1.0
        speed_factor = 1 - (abs(speed - 1000) / 1000)  # Assume optimal speed is 1000
        
        return min(1.0, 0.7 * wear_factor + 0.3 * speed_factor)
    
    def _optimize_cutting_parameters(self, current_wear: float) -> Dict[str, float]:
        """Optimize cutting parameters based on current wear"""
        # Simple optimization logic
        if current_wear < 0.3:
            speed_factor = 1.0
        elif current_wear < 0.6:
            speed_factor = 0.8
        else:
            speed_factor = 0.6
        
        return {
            'cutting_speed': 1000 * speed_factor,
            'feed_rate': 100 * speed_factor,
            'depth_of_cut': 2.0 * speed_factor
        }
    
    def save_config(self, config: ToolLifeConfig):
        """Save tool life prediction configuration"""
        config_dict = {
            'tool_id': config.tool_id,
            'material': config.material,
            'operation_type': config.operation_type,
            'cutting_parameters': config.cutting_parameters,
            'expected_life': config.expected_life,
            'maintenance_threshold': config.maintenance_threshold,
            'replacement_threshold': config.replacement_threshold
        }
        
        with open(self.data_path / f"{config.tool_id}_config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def load_config(self, tool_id: str) -> Optional[ToolLifeConfig]:
        """Load tool life prediction configuration"""
        config_file = self.data_path / f"{tool_id}_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
                return ToolLifeConfig(**config_dict)
        return None

def render_tool_life_prediction():
    """Render the Tool Life Prediction interface"""
    st.title("⏳ Tool Life Prediction")
    
    # Initialize predictor if not in session state
    if 'life_predictor' not in st.session_state:
        st.session_state.life_predictor = ToolLifePredictor()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "Real-time Prediction",
        "Life Analysis",
        "Parameter Optimization",
        "Configuration"
    ])
    
    with tab1:
        st.subheader("Real-time Tool Life Prediction")
        
        # Tool and material selection
        col1, col2 = st.columns(2)
        
        with col1:
            tool_id = st.text_input("Tool ID", "TOOL_001")
            material = st.selectbox(
                "Material",
                ["Steel", "Aluminum", "Titanium", "Stainless Steel"]
            )
        
        with col2:
            operation = st.selectbox(
                "Operation Type",
                ["Milling", "Turning", "Drilling", "Grinding"]
            )
            hardness = st.number_input("Material Hardness (HRC)", 20.0, 70.0, 45.0)
        
        # Current conditions
        st.subheader("Current Conditions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_wear = st.number_input("Current Wear (mm)", 0.0, 1.0, 0.2, 0.01)
            cutting_speed = st.number_input("Cutting Speed (m/min)", 100.0, 2000.0, 1000.0)
        
        with col2:
            feed_rate = st.number_input("Feed Rate (mm/rev)", 0.1, 1.0, 0.5, 0.1)
            depth_of_cut = st.number_input("Depth of Cut (mm)", 0.5, 5.0, 2.0)
        
        with col3:
            temperature = st.number_input("Temperature (°C)", 20.0, 200.0, 65.0)
            vibration = st.number_input("Vibration (mm/s)", 0.0, 2.0, 0.5, 0.1)
        
        # Predict remaining life
        if st.button("Predict Remaining Life", type="primary"):
            prediction = st.session_state.life_predictor.predict_remaining_life(
                current_wear, cutting_speed, feed_rate, depth_of_cut,
                hardness, temperature, vibration
            )
            
            # Display prediction results
            col1, col2 = st.columns(2)
            
            with col1:
                # Create gauge chart for remaining life
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction['remaining_hours'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'steps': [
                            {'range': [0, 20], 'color': "red"},
                            {'range': [20, 50], 'color': "yellow"},
                            {'range': [50, 100], 'color': "green"}
                        ]
                    },
                    title = {'text': "Remaining Hours"}
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric(
                    "Prediction Confidence",
                    f"{prediction['confidence']*100:.1f}%",
                    "High" if prediction['confidence'] > 0.8 else "Medium"
                )
                
                st.markdown("### Optimal Parameters")
                for param, value in prediction['optimal_parameters'].items():
                    st.metric(
                        param.replace('_', ' ').title(),
                        f"{value:.1f}",
                        "Optimal" if value > 0.8 * cutting_speed else "Adjust"
                    )
    
    with tab2:
        st.subheader("Tool Life Analysis")
        
        # Generate sample life data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
        life_data = pd.DataFrame({
            'timestamp': dates,
            'wear': np.cumsum(np.random.normal(0.01, 0.002, 100)),
            'temperature': np.random.normal(65, 5, 100),
            'cutting_speed': np.random.normal(1000, 50, 100)
        })
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Wear Progression", "Operating Parameters"),
            vertical_spacing=0.15
        )
        
        # Add wear progression
        fig.add_trace(
            go.Scatter(
                x=life_data['timestamp'],
                y=life_data['wear'],
                name="Tool Wear",
                line=dict(color='rgba(100, 255, 200, 0.8)')
            ),
            row=1, col=1
        )
        
        # Add temperature and speed
        fig.add_trace(
            go.Scatter(
                x=life_data['timestamp'],
                y=life_data['temperature'],
                name="Temperature",
                line=dict(color='rgba(255, 100, 100, 0.8)')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=life_data['timestamp'],
                y=life_data['cutting_speed'],
                name="Cutting Speed",
                line=dict(color='rgba(100, 100, 255, 0.8)')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Life statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Average Tool Life",
                "47.5 hours",
                "+2.5 hours vs. last batch"
            )
            
            st.metric(
                "Wear Rate",
                "0.021 mm/hour",
                "-0.002 mm/hour vs. nominal"
            )
        
        with col2:
            st.metric(
                "Prediction Accuracy",
                "94.2%",
                "+1.5% vs. last month"
            )
            
            st.metric(
                "Cost per Hour",
                "$12.50",
                "-$0.75 vs. target"
            )
    
    with tab3:
        st.subheader("Parameter Optimization")
        
        # Current vs. optimal parameters
        st.markdown("### Parameter Comparison")
        
        # Generate sample parameter data
        current_params = {
            'Cutting Speed': 1000,
            'Feed Rate': 0.5,
            'Depth of Cut': 2.0,
            'Tool Life': 45
        }
        
        optimal_params = {
            'Cutting Speed': 1200,
            'Feed Rate': 0.4,
            'Depth of Cut': 1.8,
            'Tool Life': 55
        }
        
        # Create comparison chart
        fig = go.Figure()
        
        # Add current parameters
        fig.add_trace(go.Scatterpolar(
            r=[current_params[p] for p in current_params.keys()],
            theta=list(current_params.keys()),
            fill='toself',
            name='Current'
        ))
        
        # Add optimal parameters
        fig.add_trace(go.Scatterpolar(
            r=[optimal_params[p] for p in optimal_params.keys()],
            theta=list(optimal_params.keys()),
            fill='toself',
            name='Optimal'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(current_params.values()),
                                max(optimal_params.values()))]
                )),
            showlegend=True,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization recommendations
        st.markdown("### Optimization Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            #### Parameter Adjustments
            1. Increase cutting speed by 20%
            2. Reduce feed rate by 0.1 mm/rev
            3. Decrease depth of cut by 0.2 mm
            """)
        
        with col2:
            st.success("""
            #### Expected Benefits
            - Tool life increase: +22%
            - Surface finish improvement: 15%
            - Cost reduction: 18%
            """)
    
    with tab4:
        st.subheader("Prediction Configuration")
        
        # Configuration form
        with st.form("life_prediction_config"):
            st.markdown("### Basic Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                config_tool_id = st.text_input("Tool ID", "TOOL_001", key="config_tool_id")
                config_material = st.selectbox(
                    "Material",
                    ["Steel", "Aluminum", "Titanium", "Stainless Steel"],
                    key="config_material"
                )
            
            with col2:
                config_operation = st.selectbox(
                    "Operation Type",
                    ["Milling", "Turning", "Drilling", "Grinding"],
                    key="config_operation"
                )
                expected_life = st.number_input("Expected Tool Life (hours)", 20.0, 100.0, 50.0)
            
            st.markdown("### Cutting Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                speed_min = st.number_input("Min Cutting Speed", 100.0, 500.0, 200.0)
                speed_max = st.number_input("Max Cutting Speed", 501.0, 2000.0, 1500.0)
            
            with col2:
                feed_min = st.number_input("Min Feed Rate", 0.1, 0.3, 0.1)
                feed_max = st.number_input("Max Feed Rate", 0.31, 1.0, 0.8)
            
            with col3:
                depth_min = st.number_input("Min Depth of Cut", 0.5, 1.0, 0.5)
                depth_max = st.number_input("Max Depth of Cut", 1.1, 5.0, 3.0)
            
            st.markdown("### Thresholds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                maintenance_threshold = st.number_input(
                    "Maintenance Threshold (hours)",
                    5.0, 20.0, 10.0
                )
            
            with col2:
                replacement_threshold = st.number_input(
                    "Replacement Threshold (hours)",
                    1.0, 10.0, 5.0
                )
            
            if st.form_submit_button("Save Configuration"):
                config = ToolLifeConfig(
                    tool_id=config_tool_id,
                    material=config_material,
                    operation_type=config_operation,
                    cutting_parameters={
                        'speed_range': (speed_min, speed_max),
                        'feed_range': (feed_min, feed_max),
                        'depth_range': (depth_min, depth_max)
                    },
                    expected_life=expected_life,
                    maintenance_threshold=maintenance_threshold,
                    replacement_threshold=replacement_threshold
                )
                
                st.session_state.life_predictor.save_config(config)
                st.success("Configuration saved successfully!")

if __name__ == "__main__":
    render_tool_life_prediction() 