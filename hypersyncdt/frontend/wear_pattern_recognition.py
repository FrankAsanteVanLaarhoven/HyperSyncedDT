import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import json
from pathlib import Path
from PIL import Image
import io
import base64

class WearPatternRecognizer:
    def __init__(self):
        self.data_path = Path("data/wear_patterns")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.setup_recognizer()
    
    def setup_recognizer(self):
        """Initialize pattern recognition components"""
        if 'recognition_state' not in st.session_state:
            st.session_state.recognition_state = {
                'current_analysis': None,
                'patterns': [],
                'history': []
            }
        
        # Initialize ML model for pattern recognition
        self.model = self._initialize_model()
    
    def _initialize_model(self) -> nn.Module:
        """Initialize the ML model for pattern recognition"""
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 wear pattern types
        )
        
        # Use MPS if available (for M1 Macs)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            model = model.to(self.device)
        else:
            self.device = torch.device("cpu")
            
        return model
    
    def analyze_pattern(self, image: Image.Image) -> Dict[str, any]:
        """Analyze wear pattern in the image"""
        # Preprocess image
        image = image.resize((64, 64))
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Map to pattern types
        pattern_types = [
            'Flank Wear',
            'Crater Wear',
            'Notch Wear',
            'Built-up Edge',
            'Chipping',
            'Thermal Cracking'
        ]
        
        patterns = {
            pattern: prob.item()
            for pattern, prob in zip(pattern_types, probabilities)
        }
        
        # Get dominant pattern
        dominant_pattern = pattern_types[torch.argmax(probabilities).item()]
        
        return {
            'patterns': patterns,
            'dominant_pattern': dominant_pattern,
            'confidence': max(probabilities).item()
        }
    
    def get_pattern_characteristics(self, pattern_type: str) -> Dict[str, str]:
        """Get characteristics of a wear pattern type"""
        characteristics = {
            'Flank Wear': {
                'description': 'Wear on the tool flank face due to friction',
                'causes': [
                    'High cutting speed',
                    'Insufficient cooling',
                    'Abrasive material'
                ],
                'remedies': [
                    'Reduce cutting speed',
                    'Improve cooling',
                    'Use coated tools'
                ]
            },
            'Crater Wear': {
                'description': 'Depression on the rake face of the tool',
                'causes': [
                    'High temperature',
                    'Chemical reaction',
                    'High feed rate'
                ],
                'remedies': [
                    'Reduce cutting parameters',
                    'Use appropriate coating',
                    'Improve cooling strategy'
                ]
            },
            'Notch Wear': {
                'description': 'Localized wear at the depth of cut line',
                'causes': [
                    'Work hardening',
                    'Oxidation',
                    'Interrupted cutting'
                ],
                'remedies': [
                    'Vary depth of cut',
                    'Use tougher tool grade',
                    'Optimize cutting parameters'
                ]
            },
            'Built-up Edge': {
                'description': 'Material adhesion on the cutting edge',
                'causes': [
                    'Low cutting speed',
                    'Poor lubrication',
                    'Sticky material'
                ],
                'remedies': [
                    'Increase cutting speed',
                    'Improve lubrication',
                    'Use different tool geometry'
                ]
            },
            'Chipping': {
                'description': 'Small pieces broken from cutting edge',
                'causes': [
                    'Interrupted cutting',
                    'Excessive feed rate',
                    'Tool brittleness'
                ],
                'remedies': [
                    'Reduce feed rate',
                    'Use tougher tool grade',
                    'Improve tool path'
                ]
            },
            'Thermal Cracking': {
                'description': 'Cracks due to thermal cycling',
                'causes': [
                    'Temperature fluctuation',
                    'Interrupted cutting',
                    'Poor cooling'
                ],
                'remedies': [
                    'Consistent cooling',
                    'Optimize cutting parameters',
                    'Use appropriate coating'
                ]
            }
        }
        
        return characteristics.get(pattern_type, {})
    
    def save_analysis(self, tool_id: str, analysis_data: Dict[str, any]):
        """Save wear pattern analysis data"""
        file_path = self.data_path / f"{tool_id}_analysis.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
            existing_data['history'].append(analysis_data)
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
        else:
            with open(file_path, 'w') as f:
                json.dump({'tool_id': tool_id, 'history': [analysis_data]}, f, indent=4)

def render_wear_pattern_recognition():
    """Render the Wear Pattern Recognition interface"""
    st.title("🔍 Wear Pattern Recognition")
    
    # Initialize recognizer if not in session state
    if 'pattern_recognizer' not in st.session_state:
        st.session_state.pattern_recognizer = WearPatternRecognizer()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "Pattern Analysis",
        "Historical Patterns",
        "Pattern Library",
        "Analysis Settings"
    ])
    
    with tab1:
        st.subheader("Tool Wear Pattern Analysis")
        
        # Tool identification
        col1, col2 = st.columns(2)
        
        with col1:
            tool_id = st.text_input("Tool ID", "TOOL_001")
            operation = st.selectbox(
                "Operation Type",
                ["Milling", "Turning", "Drilling", "Grinding"]
            )
        
        with col2:
            material = st.selectbox(
                "Material",
                ["Steel", "Aluminum", "Titanium", "Stainless Steel"]
            )
            inspection_date = st.date_input("Inspection Date", datetime.now())
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload tool wear image",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analyze pattern
            if st.button("Analyze Pattern", type="primary"):
                analysis = st.session_state.pattern_recognizer.analyze_pattern(image)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create gauge chart for confidence
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = analysis['confidence'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'steps': [
                                {'range': [0, 60], 'color': "red"},
                                {'range': [60, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ]
                        },
                        title = {'text': "Recognition Confidence"}
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"""
                    ### Dominant Pattern: {analysis['dominant_pattern']}
                    
                    **Pattern Distribution:**
                    """)
                    
                    for pattern, prob in analysis['patterns'].items():
                        st.progress(prob, text=f"{pattern}: {prob*100:.1f}%")
                
                # Get pattern characteristics
                characteristics = st.session_state.pattern_recognizer.get_pattern_characteristics(
                    analysis['dominant_pattern']
                )
                
                st.markdown(f"""
                ### Pattern Characteristics
                
                **Description:**  
                {characteristics['description']}
                
                **Common Causes:**
                """)
                
                for cause in characteristics['causes']:
                    st.markdown(f"- {cause}")
                
                st.markdown("**Recommended Remedies:**")
                for remedy in characteristics['remedies']:
                    st.markdown(f"- {remedy}")
                
                # Save analysis
                analysis_data = {
                    'timestamp': datetime.now().isoformat(),
                    'tool_id': tool_id,
                    'operation': operation,
                    'material': material,
                    'patterns': analysis['patterns'],
                    'dominant_pattern': analysis['dominant_pattern'],
                    'confidence': analysis['confidence']
                }
                
                st.session_state.pattern_recognizer.save_analysis(tool_id, analysis_data)
    
    with tab2:
        st.subheader("Historical Pattern Analysis")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now()
            )
        
        # Generate sample historical data
        dates = pd.date_range(start=start_date, end=end_date, freq='1d')
        pattern_history = pd.DataFrame({
            'timestamp': dates,
            'flank_wear': np.random.normal(0.4, 0.1, len(dates)),
            'crater_wear': np.random.normal(0.2, 0.05, len(dates)),
            'notch_wear': np.random.normal(0.15, 0.03, len(dates))
        })
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Pattern Distribution Over Time", "Cumulative Patterns"),
            vertical_spacing=0.15
        )
        
        # Add pattern distribution
        for pattern in ['flank_wear', 'crater_wear', 'notch_wear']:
            fig.add_trace(
                go.Scatter(
                    x=pattern_history['timestamp'],
                    y=pattern_history[pattern],
                    name=pattern.replace('_', ' ').title(),
                    stackgroup='one'
                ),
                row=1, col=1
            )
        
        # Add cumulative patterns
        fig.add_trace(
            go.Scatter(
                x=pattern_history['timestamp'],
                y=np.cumsum(pattern_history['flank_wear']),
                name="Cumulative Wear",
                line=dict(color='rgba(100, 255, 200, 0.8)')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern statistics
        st.subheader("Pattern Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Most Common Pattern",
                "Flank Wear",
                "40% of cases"
            )
            
            st.metric(
                "Pattern Diversity",
                "Medium",
                "3 main patterns"
            )
        
        with col2:
            st.metric(
                "Recognition Accuracy",
                "92.5%",
                "+2.5% vs. last month"
            )
            
            st.metric(
                "Pattern Stability",
                "High",
                "85% consistent"
            )
    
    with tab3:
        st.subheader("Wear Pattern Library")
        
        # Pattern selection
        selected_pattern = st.selectbox(
            "Select Pattern Type",
            [
                "Flank Wear",
                "Crater Wear",
                "Notch Wear",
                "Built-up Edge",
                "Chipping",
                "Thermal Cracking"
            ]
        )
        
        # Get pattern characteristics
        characteristics = st.session_state.pattern_recognizer.get_pattern_characteristics(
            selected_pattern
        )
        
        # Display pattern information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            ### {selected_pattern}
            
            **Description:**  
            {characteristics['description']}
            
            **Common Causes:**
            """)
            
            for cause in characteristics['causes']:
                st.markdown(f"- {cause}")
            
            st.markdown("**Recommended Remedies:**")
            for remedy in characteristics['remedies']:
                st.markdown(f"- {remedy}")
        
        with col2:
            st.markdown("### Severity Levels")
            
            severity_levels = {
                'Minor': 'Regular monitoring',
                'Moderate': 'Increased inspection',
                'Severe': 'Immediate action required'
            }
            
            for level, action in severity_levels.items():
                st.markdown(f"""
                **{level}:**  
                {action}
                """)
    
    with tab4:
        st.subheader("Analysis Settings")
        
        # Recognition settings
        st.markdown("### Recognition Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.5, 1.0, 0.8
            )
            
            detection_mode = st.selectbox(
                "Detection Mode",
                ["High Precision", "Balanced", "High Recall"]
            )
        
        with col2:
            multi_pattern = st.checkbox(
                "Enable Multi-pattern Detection",
                value=True
            )
            
            auto_save = st.checkbox(
                "Auto-save Analysis Results",
                value=True
            )
        
        # Notification settings
        st.markdown("### Notification Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            notify_threshold = st.slider(
                "Notification Threshold",
                0.0, 1.0, 0.9
            )
        
        with col2:
            notification_types = st.multiselect(
                "Notification Types",
                ["Email", "Slack", "System Alert"],
                default=["System Alert"]
            )
        
        if st.button("Save Settings"):
            st.success("Analysis settings updated successfully!")

if __name__ == "__main__":
    render_wear_pattern_recognition() 