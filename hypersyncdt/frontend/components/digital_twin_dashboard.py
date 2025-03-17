import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random
import json
from scipy import signal

# Custom CSS for better appearance - matching the dark theme in the screenshot
custom_css = """
<style>
    .main-header {color:#F0F2F6; text-align:center; font-size:2.5rem; font-weight:bold; margin-bottom:30px;}
    .sub-header {color:#F0F2F6; font-size:1.5rem; font-weight:bold; margin-top:20px; margin-bottom:10px;}
    .metric-container {background-color:#1E2130; border-radius:5px; padding:15px; margin:10px 0px;}
    .alert-box {background-color:#2C3454; border-left:5px solid #FF4B4B; padding:15px; margin:10px 0px;}
    .info-box {background-color:#2C3454; border-left:5px solid #4B8BFF; padding:15px; margin:10px 0px;}
    .digital-twin-container {background-color:#1E2130; border-radius:5px; padding:15px; margin:10px 0px;}
    
    /* Sidebar styling to match screenshot */
    .css-1d391kg {background-color: #111827;}
    .css-1lcbmhc {background-color: #111827;}
    
    /* Main area styling */
    .css-18e3th9 {background-color: #141b2d;}
    
    /* Custom header to match screenshot */
    .stApp header {background-color: #111827; color: white;}
    
    /* Dropdown styling */
    .stSelectbox {background-color: #1E2130;}
    
    /* Button styling */
    .stButton>button {
        background-color: #334155;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    }
    
    .stButton>button:hover {
        background-color: #4B5563;
    }
    
    /* Plot background */
    .js-plotly-plot {background-color: #1E2130;}
</style>
"""

# Load digital twin model (simplified version for demonstration)
class ToolDynamicsModel:
    def __init__(self):
        # Parameters for the digital twin model
        self.natural_frequency = 120.0  # Hz
        self.damping_ratio = 0.05
        self.stiffness_new = 1.0
        self.wear_coefficient = 0.002
        
    def predict_vibration(self, force, current_wear):
        # Simplified model: vibration increases with wear and applied force
        stiffness_current = self.stiffness_new * (1 - current_wear)
        natural_freq_current = self.natural_frequency * np.sqrt(stiffness_current)
        
        # Create a second-order system
        wn = 2 * np.pi * natural_freq_current
        num = [wn**2]
        den = [1, 2 * self.damping_ratio * wn, wn**2]
        
        # Apply force input to the system
        t = np.linspace(0, 0.1, 100)
        force_input = force * np.ones_like(t)
        # Fix: In newer versions of SciPy, lsim returns 3 values (t_out, y_out, x_out)
        # Use the second value (y_out) as our response
        lsim_result = signal.lsim((num, den), force_input, t)
        response = lsim_result[1]  # Get just the response values
        
        # Add noise and wear effect
        noise = np.random.normal(0, 0.1, response.shape)
        wear_effect = 2 * current_wear * response
        
        return response + noise + wear_effect
    
    def predict_temperature(self, cutting_speed, feed_rate, current_wear):
        # Simplified model: temperature increases with wear, speed and feed rate
        base_temp = 25.0  # ambient temperature
        speed_effect = 0.5 * cutting_speed
        feed_effect = 10.0 * feed_rate
        wear_effect = 100.0 * current_wear
        
        # Add some randomness
        noise = np.random.normal(0, 2.0)
        
        return base_temp + speed_effect + feed_effect + wear_effect + noise
    
    def predict_tool_life(self, current_wear, cutting_speed, material_hardness):
        # Taylor's tool life equation (simplified)
        v_c = cutting_speed
        n = 0.25  # Taylor's exponent (depends on tool material)
        C = 80  # Taylor's constant
        
        # Adjust for current wear and material
        remaining_life_minutes = (C / v_c)**n * (1 - current_wear) * (1 / material_hardness)
        
        return remaining_life_minutes

# Initialize the digital twin model
@st.cache_resource
def load_digital_twin():
    return ToolDynamicsModel()

digital_twin = load_digital_twin()

# Generate simulated manufacturing data
def generate_manufacturing_data(last_timestamp, current_state, n_samples=1, time_delta=5):
    """Generate synthetic manufacturing data with digital twin predictions"""
    digital_twin = load_digital_twin()
    
    # Create timestamps
    timestamps = [last_timestamp + timedelta(seconds=i*time_delta) for i in range(n_samples)]
    
    # Get current state parameters
    current_wear = current_state['wear']
    current_cutting_speed = current_state['cutting_speed']
    current_feed_rate = current_state['feed_rate']
    current_material = current_state['material']
    
    # Material properties lookup
    material_properties = {
        'Aluminum': {'hardness': 0.3, 'wear_factor': 0.5},
        'Steel': {'hardness': 0.7, 'wear_factor': 1.0},
        'Titanium': {'hardness': 0.9, 'wear_factor': 1.5},
        'Composite': {'hardness': 0.6, 'wear_factor': 1.2}
    }
    
    # Default values if material not found
    if current_material not in material_properties:
        current_material = 'Steel'
    
    material_hardness = material_properties[current_material]['hardness']
    wear_factor = material_properties[current_material]['wear_factor']
    
    # Arrays to store generated data
    wear_values = []
    vibration_values = []
    temperature_values = []
    force_values = []
    dt_vibration_values = []  # Digital twin predictions
    dt_temperature_values = []
    dt_wear_values = []
    
    for i in range(n_samples):
        # Calculate applied cutting force (simplified)
        force = current_cutting_speed * current_feed_rate * material_hardness * (1 + 0.5 * current_wear)
        force += np.random.normal(0, force * 0.05)  # Add noise
        
        # Update wear based on current conditions
        wear_increment = (
            0.0001 * wear_factor * 
            (current_cutting_speed / 100) * 
            (current_feed_rate / 0.1) * 
            (1 + 0.5 * current_wear)  # Wear accelerates as tool degrades
        )
        
        # Add randomness to wear progression
        wear_increment += np.random.normal(0, wear_increment * 0.1)
        
        # Update current wear
        current_wear += wear_increment
        
        # Get actual vibration (with some randomness to simulate real-world conditions)
        vibration_base = 0.5 + 5 * current_wear + 0.01 * force
        vibration = vibration_base + np.random.normal(0, vibration_base * 0.2)
        
        # Get actual temperature
        temperature_base = 25 + 0.2 * force + 100 * current_wear
        temperature = temperature_base + np.random.normal(0, 2)
        
        # Get digital twin predictions
        dt_vibration_raw = digital_twin.predict_vibration(force, current_wear)
        dt_vibration = np.mean(np.abs(dt_vibration_raw)) * 5  # Scale for visualization
        
        dt_temperature = digital_twin.predict_temperature(
            current_cutting_speed, current_feed_rate, current_wear
        )
        
        # Digital twin's wear prediction (slightly different from actual)
        dt_wear = current_wear * (1 + np.random.normal(0, 0.05))
        
        # Store values
        wear_values.append(current_wear)
        vibration_values.append(vibration)
        temperature_values.append(temperature)
        force_values.append(force)
        dt_vibration_values.append(dt_vibration)
        dt_temperature_values.append(dt_temperature)
        dt_wear_values.append(dt_wear)
    
    # Create DataFrame with all metrics
    data = pd.DataFrame({
        'timestamp': timestamps,
        'tool_wear': wear_values,
        'vibration': vibration_values,
        'temperature': temperature_values,
        'cutting_force': force_values,
        'dt_vibration': dt_vibration_values,
        'dt_temperature': dt_temperature_values,
        'dt_wear': dt_wear_values
    })
    
    # Update the session state with the new wear value
    st.session_state.current_state['wear'] = current_wear
    
    return data

def render_digital_twin_dashboard():
    """
    Main function to render the digital twin dashboard
    """
    st.title("Digital Twin Manufacturing Monitoring")
    
    # Force initial data generation when loaded from main app
    from_main_app = "main_app_integration" in st.session_state and st.session_state.main_app_integration
    
    # Initialize digital twin model
    digital_twin = load_digital_twin()
    
    # Initialize session state if needed
    if "manufacturing_data" not in st.session_state or len(st.session_state.manufacturing_data) <= 1:
        from datetime import datetime
        import pandas as pd
        st.session_state.manufacturing_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'tool_wear': [0.1],
            'vibration': [0.5],
            'temperature': [25.0],
            'cutting_force': [10.0],
            'dt_vibration': [0.5],
            'dt_temperature': [25.0],
            'dt_wear': [0.1]
        })
    
    if "current_state" not in st.session_state:
        st.session_state.current_state = {
            'wear': 0.1,
            'cutting_speed': 100,  # m/min
            'feed_rate': 0.1,      # mm/rev
            'material': 'Steel'
        }
    
    if "streaming_active" not in st.session_state:
        st.session_state.streaming_active = True
    
    if "update_frequency" not in st.session_state:
        st.session_state.update_frequency = 2
    
    # Auto-start data generation if coming from main app
    if from_main_app and len(st.session_state.manufacturing_data) <= 5:
        # Generate some initial data points for better visualization
        last_timestamp = st.session_state.manufacturing_data['timestamp'].iloc[-1]
        for _ in range(10):  # Generate 10 initial points
            new_data = generate_manufacturing_data(last_timestamp, st.session_state.current_state)
            st.session_state.manufacturing_data = pd.concat([st.session_state.manufacturing_data, new_data], ignore_index=True)
            last_timestamp = new_data['timestamp'].iloc[0]
    
    # Dashboard Layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.sidebar.header("Manufacturing Controls")
        
        # Material selection
        material = st.sidebar.selectbox(
            "Material",
            ["Steel", "Aluminum", "Titanium", "Composite"],
            index=["Steel", "Aluminum", "Titanium", "Composite"].index(st.session_state.current_state['material'])
        )
        
        # Cutting parameters
        cutting_speed = st.sidebar.slider(
            "Cutting Speed (m/min)",
            min_value=50, 
            max_value=200, 
            value=int(st.session_state.current_state['cutting_speed']),
            step=5
        )
        
        feed_rate = st.sidebar.slider(
            "Feed Rate (mm/rev)", 
            min_value=0.05, 
            max_value=0.5, 
            value=float(st.session_state.current_state['feed_rate']),
            step=0.01,
            format="%.2f"
        )
        
        # Update current state if parameters change
        if (material != st.session_state.current_state['material'] or
            cutting_speed != st.session_state.current_state['cutting_speed'] or
            feed_rate != st.session_state.current_state['feed_rate']):
            
            st.session_state.current_state = {
                'wear': st.session_state.current_state['wear'],
                'cutting_speed': cutting_speed,
                'feed_rate': feed_rate,
                'material': material
            }
        
        # Alert thresholds
        st.sidebar.subheader("Alert Thresholds")
        wear_threshold = st.sidebar.slider("Tool Wear Threshold", 0.1, 1.0, 0.7, 0.05)
        vibration_threshold = st.sidebar.slider("Vibration Threshold", 0.5, 5.0, 3.0, 0.1)
        temperature_threshold = st.sidebar.slider("Temperature Threshold (°C)", 25.0, 150.0, 100.0, 5.0)
        
        # Monitoring controls
        st.sidebar.subheader("Monitoring Controls")
        streaming_active = st.sidebar.toggle("Live Data Stream", st.session_state.streaming_active)
        
        if streaming_active != st.session_state.streaming_active:
            st.session_state.streaming_active = streaming_active
            
        update_frequency = st.sidebar.slider(
            "Update Frequency (seconds)",
            min_value=1,
            max_value=10,
            value=st.session_state.update_frequency,
            step=1
        )
        
        if update_frequency != st.session_state.update_frequency:
            st.session_state.update_frequency = update_frequency
        
        run_every = st.session_state.update_frequency
        
    # Force immediate update when loaded from main app
    if from_main_app:
        # Call update directly and set run_every to make sure fragment runs
        run_every = st.session_state.update_frequency

    @st.fragment(run_every=run_every)
    def update_monitoring_visualizations():
        # Get latest data
        latest_data = st.session_state.manufacturing_data.iloc[-1]
        latest_timestamp = latest_data['timestamp']
        
        # Generate new data point if streaming is active
        if st.session_state.streaming_active or (from_main_app and len(st.session_state.manufacturing_data) <= 15):
            new_data = generate_manufacturing_data(latest_timestamp, st.session_state.current_state)
            st.session_state.manufacturing_data = pd.concat([st.session_state.manufacturing_data, new_data], ignore_index=True)
            
            # Keep only last 100 records to avoid memory issues
            if len(st.session_state.manufacturing_data) > 100:
                st.session_state.manufacturing_data = st.session_state.manufacturing_data.iloc[-100:]
        
        # Filter data for display (last 30 seconds or minutes depending on scale)
        display_df = st.session_state.manufacturing_data
            
        # REAL-TIME MONITORING SECTION
        st.header("Real-Time Monitoring")
        
        # Tool Wear Monitoring
        col1, col2 = st.columns(2)
        
        with col1:
            # Tool Wear Chart with Future Prediction
            fig_wear = go.Figure()
            
            # Actual tool wear data
            fig_wear.add_trace(go.Scatter(
                x=display_df['timestamp'], 
                y=display_df['tool_wear'],
                mode='lines+markers',
                name='Actual Tool Wear',
                line=dict(color='blue')
            ))
            
            # Digital twin prediction
            fig_wear.add_trace(go.Scatter(
                x=display_df['timestamp'], 
                y=display_df['dt_wear'],
                mode='lines',
                name='Digital Twin Prediction',
                line=dict(color='red', dash='dot')
            ))
            
            # Add threshold line
            fig_wear.add_shape(
                type="line",
                x0=display_df['timestamp'].min(),
                y0=wear_threshold,
                x1=display_df['timestamp'].max(),
                y1=wear_threshold,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            fig_wear.update_layout(
                title='Tool Wear Monitoring',
                xaxis_title='Time',
                yaxis_title='Tool Wear (mm)',
                height=300,
                margin=dict(l=10, r=10, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_wear, use_container_width=True)
        
        with col2:
            # Combined Vibration and Temperature Chart (dual y-axis)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add vibration data to primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=display_df['timestamp'], 
                    y=display_df['vibration'],
                    mode='lines+markers',
                    name='Vibration',
                    line=dict(color='orange')
                ),
                secondary_y=False,
            )
            
            # Add temperature data to secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=display_df['timestamp'], 
                    y=display_df['temperature'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='red')
                ),
                secondary_y=True,
            )
            
            # Add digital twin predictions
            fig.add_trace(
                go.Scatter(
                    x=display_df['timestamp'], 
                    y=display_df['dt_vibration'],
                    mode='lines',
                    name='DT Vibration',
                    line=dict(color='orange', dash='dot')
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=display_df['timestamp'], 
                    y=display_df['dt_temperature'],
                    mode='lines',
                    name='DT Temperature',
                    line=dict(color='red', dash='dot')
                ),
                secondary_y=True,
            )
            
            # Add threshold lines
            fig.add_shape(
                type="line",
                x0=display_df['timestamp'].min(),
                y0=vibration_threshold,
                x1=display_df['timestamp'].max(),
                y1=vibration_threshold,
                line=dict(color="orange", width=2, dash="dash"),
                secondary_y=False,
            )
            
            fig.add_shape(
                type="line",
                x0=display_df['timestamp'].min(),
                y0=temperature_threshold,
                x1=display_df['timestamp'].max(),
                y1=temperature_threshold,
                line=dict(color="red", width=2, dash="dash"),
                secondary_y=True,
            )
            
            fig.update_layout(
                title='Vibration and Temperature Monitoring',
                xaxis_title='Time',
                height=300,
                margin=dict(l=10, r=10, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig.update_yaxes(title_text="Vibration (mm/s)", secondary_y=False)
            fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ALERTS SECTION
        st.subheader("System Alerts")
        
        # Get latest data for alerts
        latest = display_df.iloc[-1]
        
        # Check for threshold violations
        alert_col1, alert_col2, alert_col3 = st.columns(3)
        
        with alert_col1:
            if latest['tool_wear'] > wear_threshold:
                st.error(f"🔴 Tool Wear Alert: {latest['tool_wear']:.2f} mm (Threshold: {wear_threshold} mm)")
                st.info("Recommendation: Schedule tool replacement within next 2 hours")
            else:
                wear_percent = (latest['tool_wear'] / wear_threshold) * 100
                st.success(f"✅ Tool Wear: {latest['tool_wear']:.2f} mm ({wear_percent:.1f}% of threshold)")
        
        with alert_col2:
            if latest['vibration'] > vibration_threshold:
                st.error(f"🔴 Vibration Alert: {latest['vibration']:.2f} mm/s (Threshold: {vibration_threshold} mm/s)")
                st.info("Recommendation: Check machine alignment and workpiece fixturing")
            else:
                vibration_percent = (latest['vibration'] / vibration_threshold) * 100
                st.success(f"✅ Vibration: {latest['vibration']:.2f} mm/s ({vibration_percent:.1f}% of threshold)")
        
        with alert_col3:
            if latest['temperature'] > temperature_threshold:
                st.error(f"🔴 Temperature Alert: {latest['temperature']:.1f}°C (Threshold: {temperature_threshold}°C)")
                st.info("Recommendation: Increase coolant flow or reduce cutting parameters")
            else:
                temp_percent = (latest['temperature'] / temperature_threshold) * 100
                st.success(f"✅ Temperature: {latest['temperature']:.1f}°C ({temp_percent:.1f}% of threshold)")
        
        # METRICS SECTION
        st.subheader("Digital Twin Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="Digital Twin Accuracy (Vibration)",
                value=f"{100 - abs(latest['vibration'] - latest['dt_vibration']) / latest['vibration'] * 100:.1f}%",
                delta=f"{latest['dt_vibration'] - latest['vibration']:.2f} mm/s"
            )
        
        with metric_col2:
            st.metric(
                label="Digital Twin Accuracy (Temperature)",
                value=f"{100 - abs(latest['temperature'] - latest['dt_temperature']) / latest['temperature'] * 100:.1f}%",
                delta=f"{latest['dt_temperature'] - latest['temperature']:.1f}°C"
            )
        
        with metric_col3:
            st.metric(
                label="Digital Twin Accuracy (Tool Wear)",
                value=f"{100 - abs(latest['tool_wear'] - latest['dt_wear']) / latest['tool_wear'] * 100:.1f}%",
                delta=f"{latest['dt_wear'] - latest['tool_wear']:.3f} mm"
            )
        
        with metric_col4:
            # Predict remaining tool life
            material_hardness = {"Steel": 60, "Aluminum": 30, "Titanium": 90, "Composite": 45}[material]
            remaining_life = digital_twin.predict_tool_life(
                latest['tool_wear'], 
                cutting_speed, 
                material_hardness
            )
            
            st.metric(
                label="Estimated Remaining Tool Life",
                value=f"{remaining_life:.1f} minutes",
                delta=None
            )
        
        # PhD PROJECT INFO
        st.markdown("---")
        st.markdown("**PhD Project: Digital Twin Technology for Advanced Manufacturing**")
        st.markdown("This dashboard demonstrates real-time monitoring with digital twin prediction capabilities.")
    
    # Make sure the fragment runs immediately on load
    update_monitoring_visualizations()
    return

def main():
    render_digital_twin_dashboard()

if __name__ == "__main__":
    main() 