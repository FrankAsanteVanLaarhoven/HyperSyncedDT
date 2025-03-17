import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import signal

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

# Generate manufacturing data
def generate_manufacturing_data(last_timestamp, current_state, n_samples=1):
    """Generate synthetic manufacturing data with digital twin predictions"""
    digital_twin = load_digital_twin()
    
    # Create timestamps
    timestamps = [last_timestamp + timedelta(seconds=i*2) for i in range(n_samples)]
    
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

def render_live_dashboard():
    """Render the live manufacturing dashboard with real-time updates."""
    
    # Initialize session state for data storage if not already done
    if "manufacturing_data" not in st.session_state:
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
    
    # Load digital twin model
    digital_twin = load_digital_twin()
    
    # Container for the persistent dashboard
    st.markdown("## Real-Time Manufacturing Status")
    
    # Update data if streaming is active
    if st.session_state.streaming_active:
        # Get latest timestamp
        latest_timestamp = st.session_state.manufacturing_data['timestamp'].iloc[-1]
        
        # Generate new data
        new_data = generate_manufacturing_data(latest_timestamp, st.session_state.current_state)
        
        # Update the data in session state
        st.session_state.manufacturing_data = pd.concat([st.session_state.manufacturing_data, new_data], ignore_index=True)
        
        # Keep only the last 100 records
        if len(st.session_state.manufacturing_data) > 100:
            st.session_state.manufacturing_data = st.session_state.manufacturing_data.iloc[-100:]
    
    # Get the latest data point
    latest_data = st.session_state.manufacturing_data.iloc[-1]
    
    # Define thresholds for alerts
    wear_threshold = 0.7
    vibration_threshold = 3.0
    temperature_threshold = 100.0

    # Create side-by-side layout
    col1, col2 = st.columns([2, 1])
    
    # Add update fragment to refresh data periodically
    @st.fragment(run_every=st.session_state.update_frequency)
    def update_dashboard_data():
        # Get latest timestamp
        if len(st.session_state.manufacturing_data) > 0:
            latest_timestamp = st.session_state.manufacturing_data['timestamp'].iloc[-1]
            
            # Generate new data
            new_data = generate_manufacturing_data(latest_timestamp, st.session_state.current_state)
            
            # Update the data in session state
            st.session_state.manufacturing_data = pd.concat([st.session_state.manufacturing_data, new_data], ignore_index=True)
            
            # Keep only the last 100 records
            if len(st.session_state.manufacturing_data) > 100:
                st.session_state.manufacturing_data = st.session_state.manufacturing_data.iloc[-100:]
    
    # Run the fragment
    update_dashboard_data()

    # Left side: Monitoring charts
    with col1:
        # Temperature & Vibration Chart
        temp_vib_fig = go.Figure()
        
        # Get last 20 data points for display
        display_df = st.session_state.manufacturing_data.iloc[-20:]
        
        # Temperature line
        temp_vib_fig.add_trace(go.Scatter(
            x=display_df['timestamp'], 
            y=display_df['temperature'],
            name="Temperature",
            line=dict(color='#FF4B4B', width=2)
        ))
        
        # Vibration line
        temp_vib_fig.add_trace(go.Scatter(
            x=display_df['timestamp'], 
            y=display_df['vibration'],
            name="Vibration",
            line=dict(color='#4B8BFF', width=2)
        ))
        
        # Tool Wear line
        temp_vib_fig.add_trace(go.Scatter(
            x=display_df['timestamp'], 
            y=display_df['tool_wear']*100,  # Scale for better visibility
            name="Tool Wear (x100)",
            line=dict(color='#FFAA00', width=2)
        ))
        
        # Digital twin prediction
        temp_vib_fig.add_trace(go.Scatter(
            x=display_df['timestamp'], 
            y=display_df['dt_wear']*100,  # Scale for better visibility
            name="DT Wear Pred (x100)",
            line=dict(color='#FFAA00', width=1, dash='dash')
        ))
        
        # Add threshold lines
        temp_vib_fig.add_shape(
            type="line",
            x0=display_df['timestamp'].min(),
            y0=vibration_threshold,
            x1=display_df['timestamp'].max(),
            y1=vibration_threshold,
            line=dict(color="blue", width=1, dash="dash")
        )
        
        temp_vib_fig.add_shape(
            type="line",
            x0=display_df['timestamp'].min(),
            y0=temperature_threshold,
            x1=display_df['timestamp'].max(),
            y1=temperature_threshold,
            line=dict(color="red", width=1, dash="dash")
        )
        
        # Update layout
        temp_vib_fig.update_layout(
            title='Temperature, Vibration & Tool Wear',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Time",
                showgrid=True,
                gridcolor='rgba(80, 80, 80, 0.1)'
            ),
            yaxis=dict(
                title="Value",
                showgrid=True,
                gridcolor='rgba(80, 80, 80, 0.1)'
            )
        )
        
        # Show the plot
        st.plotly_chart(temp_vib_fig, use_container_width=True)
    
    # Right side: Metrics and status
    with col2:
        # Create metrics grid
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        # Tool wear metric
        with metric_col1:
            wear_value = latest_data['tool_wear']
            wear_percent = (wear_value / wear_threshold) * 100
            
            if wear_value > wear_threshold:
                wear_color = "red"
                wear_status = "Critical"
            elif wear_value > 0.7 * wear_threshold:
                wear_color = "orange"
                wear_status = "Warning"
            else:
                wear_color = "green"
                wear_status = "Normal"
            
            st.metric(
                label="Tool Wear",
                value=f"{wear_value:.3f} mm",
                delta=f"{wear_percent:.1f}%",
                delta_color="inverse"
            )
            st.caption(f":{'red' if wear_color == 'red' else 'orange' if wear_color == 'orange' else 'green'}_circle: {wear_status}")
        
        # Vibration metric
        with metric_col2:
            vibration_value = latest_data['vibration']
            vibration_percent = (vibration_value / vibration_threshold) * 100
            
            if vibration_value > vibration_threshold:
                vibration_color = "red"
                vibration_status = "Critical"
            elif vibration_value > 0.7 * vibration_threshold:
                vibration_color = "orange"
                vibration_status = "Warning"
            else:
                vibration_color = "green"
                vibration_status = "Normal"
            
            st.metric(
                label="Vibration",
                value=f"{vibration_value:.2f} mm/s",
                delta=f"{vibration_percent:.1f}%",
                delta_color="inverse"
            )
            st.caption(f":{'red' if vibration_color == 'red' else 'orange' if vibration_color == 'orange' else 'green'}_circle: {vibration_status}")
        
        # Temperature metric
        with metric_col3:
            temperature_value = latest_data['temperature']
            temperature_percent = (temperature_value / temperature_threshold) * 100
            
            if temperature_value > temperature_threshold:
                temperature_color = "red"
                temperature_status = "Critical"
            elif temperature_value > 0.7 * temperature_threshold:
                temperature_color = "orange"
                temperature_status = "Warning"
            else:
                temperature_color = "green"
                temperature_status = "Normal"
            
            st.metric(
                label="Temperature",
                value=f"{temperature_value:.1f} °C",
                delta=f"{temperature_percent:.1f}%", 
                delta_color="inverse"
            )
            st.caption(f":{'red' if temperature_color == 'red' else 'orange' if temperature_color == 'orange' else 'green'}_circle: {temperature_status}")
        
        # Additional info about the digital twin
        st.markdown("### Digital Twin Prediction")
        st.info(f"Estimated remaining tool life: {digital_twin.predict_tool_life(latest_data['tool_wear'], st.session_state.current_state['cutting_speed'], 0.7):.1f} minutes")
    
    # Add a horizontal rule for separation
    st.markdown("---")
    
    # Return height of the dashboard for spacing calculations
    return 400  # Approximate height in pixels

if __name__ == "__main__":
    # For testing the dashboard standalone
    render_live_dashboard() 