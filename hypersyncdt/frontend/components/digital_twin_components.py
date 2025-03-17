import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
import plotly.graph_objects as go
import logging
import torch
import torch.nn as nn
import streamlit as st

# Simplified classes for Streamlit Cloud

class MachineSpecs:
    """Machine specifications data class"""
    def __init__(self, machine_id: str, type: str, model: str, axes: int, max_speed: float, workspace: Tuple[float, float, float], api_endpoint: str, sensor_endpoints: Dict[str, str]):
        self.machine_id = machine_id
        self.type = type
        self.model = model
        self.axes = axes
        self.max_speed = max_speed
        self.workspace = workspace
        self.api_endpoint = api_endpoint
        self.sensor_endpoints = sensor_endpoints

class MachineConnector:
    """Handles connections to machine APIs and data streaming (simplified for Streamlit Cloud)"""
    
    def __init__(self):
        self.machine_specs = {}
        self.connections = {}
        self.data_buffer = {}
        
        # Initialize machine database
        self._init_machine_database()
    
    def _init_machine_database(self):
        """Initialize database of supported machines"""
        # Mitsubishi Heavy Industries machines
        self.machine_specs["MHI-M8"] = MachineSpecs(
            machine_id="MHI-M8",
            type="5-Axis Machining Center",
            model="M8-V Series",
            axes=5,
            max_speed=12000,
            workspace=(800, 1000, 850),
            api_endpoint="http://mhi-m8/api",
            sensor_endpoints={
                "camera_front": "http://mhi-m8/camera1",
                "camera_top": "http://mhi-m8/camera2",
                "camera_side": "http://mhi-m8/camera3"
            }
        )
        
        self.machine_specs["MHI-MVR"] = MachineSpecs(
            machine_id="MHI-MVR",
            type="Double-Column Machining Center",
            model="MVR-Ex Series",
            axes=5,
            max_speed=15000,
            workspace=(1600, 2000, 1000),
            api_endpoint="http://mhi-mvr/api",
            sensor_endpoints={
                "camera_front": "http://mhi-mvr/camera1",
                "camera_top": "http://mhi-mvr/camera2",
                "camera_side": "http://mhi-mvr/camera3"
            }
        )
        
        # Sandvik machines
        self.machine_specs["SANDVIK-CT50"] = MachineSpecs(
            machine_id="SANDVIK-CT50",
            type="Turning Center",
            model="CoroTurn Prime 50",
            axes=4,
            max_speed=4000,
            workspace=(500, 300, 400),
            api_endpoint="http://sandvik-ct50/api",
            sensor_endpoints={
                "camera_front": "http://sandvik-ct50/camera1",
                "camera_top": "http://sandvik-ct50/camera2"
            }
        )
        
        self.machine_specs["SANDVIK-CG3100"] = MachineSpecs(
            machine_id="SANDVIK-CG3100",
            type="Multi-Task Machining Center",
            model="Coromant Capto G3100",
            axes=5,
            max_speed=6000,
            workspace=(600, 400, 500),
            api_endpoint="http://sandvik-cg3100/api",
            sensor_endpoints={
                "camera_front": "http://sandvik-cg3100/camera1",
                "camera_top": "http://sandvik-cg3100/camera2",
                "vibration": "http://sandvik-cg3100/sensors/vibration"
            }
        )

class DigitalTwinVisualizer:
    """Visualizes digital twin data"""
    
    def __init__(self):
        self.visualizations = {}
        
    def create_visualization(self, twin_id: str, config: Dict[str, Any]) -> None:
        """Create a visualization for a digital twin"""
        self.visualizations[twin_id] = {
            "config": config,
            "plots": {}
        }
        
    def render_dashboard(self):
        """Render a simple dashboard with plotly"""
        
        st.subheader("Digital Twin Visualization")
        
        # Sample data
        time_range = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        temperature = 60 + 10 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 1, 100)
        pressure = 100 + 5 * np.cos(np.linspace(0, 3*np.pi, 100)) + np.random.normal(0, 0.5, 100)
        
        # Create figures
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=time_range, y=temperature, name="Temperature", line=dict(color='firebrick')))
        fig1.update_layout(title="Machine Temperature", xaxis_title="Time", yaxis_title="Temperature (°C)")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=time_range, y=pressure, name="Pressure", line=dict(color='royalblue')))
        fig2.update_layout(title="Machine Pressure", xaxis_title="Time", yaxis_title="Pressure (PSI)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
class CameraManager:
    """Manages camera streams (simplified for Streamlit Cloud)"""
    
    def __init__(self):
        self.cameras = {}
        
    def connect_camera(self, camera_id: str, stream_url: str):
        """Connect to a camera stream"""
        self.cameras[camera_id] = {"url": stream_url, "status": "connected"}
        
    def get_latest_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the latest frame from a camera"""
        # Simplified to return None since we're not actually connecting to cameras
        return None

class SensorProcessor:
    """Process sensor data (simplified for Streamlit Cloud)"""
    
    def __init__(self):
        self.sensor_data = {}
        
    def process_sensor_data(self, machine_id: str, sensor_data: Dict) -> Dict:
        """Process sensor data from a machine"""
        # Store the data
        if machine_id not in self.sensor_data:
            self.sensor_data[machine_id] = []
        
        self.sensor_data[machine_id].append({
            "timestamp": datetime.now(),
            "data": sensor_data
        })
        
        # Simple processing - just return the data with a calculated trend
        result = {
            "processed_data": sensor_data,
            "trends": {k: self._calculate_trend(k, machine_id) for k in sensor_data.keys()},
            "alerts": []
        }
        
        # Add a sample alert for demonstration
        if len(self.sensor_data[machine_id]) > 10:
            if "temperature" in sensor_data and sensor_data["temperature"] > 80:
                result["alerts"].append({
                    "type": "High Temperature",
                    "value": sensor_data["temperature"],
                    "threshold": 80,
                    "severity": "Warning"
                })
                
        return result
    
    def _calculate_trend(self, sensor: str, machine_id: str) -> float:
        """Calculate the trend for a sensor"""
        # Simplified trend calculation
        if machine_id not in self.sensor_data or len(self.sensor_data[machine_id]) < 2:
            return 0.0
            
        try:
            latest = self.sensor_data[machine_id][-1]["data"].get(sensor, 0)
            previous = self.sensor_data[machine_id][-2]["data"].get(sensor, 0)
            return latest - previous
        except (KeyError, IndexError):
            return 0.0
            
    def get_sensor_history(self, machine_id: str, sensor: str, timeframe: str = '1h') -> pd.DataFrame:
        """Get historical sensor data"""
        # Generate sample data for demonstration
        end_time = datetime.now()
        
        if timeframe == '1h':
            start_time = end_time - timedelta(hours=1)
            freq = '1min'
            periods = 60
        elif timeframe == '1d':
            start_time = end_time - timedelta(days=1)
            freq = '15min'
            periods = 96
        else:
            start_time = end_time - timedelta(hours=1)
            freq = '1min'
            periods = 60
            
        time_range = pd.date_range(start=start_time, end=end_time, periods=periods)
        
        # Generate different patterns based on sensor type
        if sensor == 'temperature':
            values = 65 + 10 * np.sin(np.linspace(0, 4*np.pi, len(time_range))) + np.random.normal(0, 1, len(time_range))
        elif sensor == 'pressure':
            values = 100 + 5 * np.cos(np.linspace(0, 6*np.pi, len(time_range))) + np.random.normal(0, 0.5, len(time_range))
        elif sensor == 'vibration':
            values = 0.2 + 0.1 * np.sin(np.linspace(0, 8*np.pi, len(time_range))) + np.abs(np.random.normal(0, 0.05, len(time_range)))
        else:
            values = 50 + 10 * np.random.randn(len(time_range))
            
        return pd.DataFrame({
            'timestamp': time_range,
            'value': values,
            'sensor': sensor,
            'machine_id': machine_id
        })

class SensorData:
    """Data model for sensor readings"""
    def __init__(self, timestamp: datetime, sensor_id: str, value: float, unit: str, status: str):
        self.timestamp = timestamp
        self.sensor_id = sensor_id
        self.value = value
        self.unit = unit
        self.status = status

class MachineState:
    """Data model for machine state"""
    def __init__(self, machine_id: str, status: str, temperature: float, pressure: float, 
                 vibration: float, power_consumption: float, last_maintenance: datetime, 
                 next_maintenance: datetime):
        self.machine_id = machine_id
        self.status = status
        self.temperature = temperature
        self.pressure = pressure
        self.vibration = vibration
        self.power_consumption = power_consumption
        self.last_maintenance = last_maintenance
        self.next_maintenance = next_maintenance

class SynchronizedDigitalTwin:
    """A digital twin synchronized with a physical asset"""
    
    def __init__(self):
        self.model = self._initialize_model()
        self.sensor_readings = {}
        self.machine_states = {}
        
    def _initialize_model(self) -> nn.Module:
        """Initialize a simple neural network model for the digital twin"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        return model
    
    def sync_with_physical_asset(self, machine_id: str) -> None:
        """Simulate synchronization with a physical asset"""
        # In a real implementation, this would connect to actual sensors
        # Here we generate a sample state
        self.machine_states[machine_id] = self._generate_sample_state(machine_id)
        
    def _generate_sample_state(self, machine_id: str) -> MachineState:
        """Generate a sample machine state for demonstration"""
        now = datetime.now()
        
        # Generate slightly different states for different machines
        if machine_id.startswith('MHI'):
            temperature = 65 + np.random.normal(0, 2)
            pressure = 100 + np.random.normal(0, 1)
            vibration = 0.2 + np.random.normal(0, 0.02)
            power = 15 + np.random.normal(0, 0.5)
        else:
            temperature = 60 + np.random.normal(0, 2)
            pressure = 95 + np.random.normal(0, 1)
            vibration = 0.15 + np.random.normal(0, 0.02)
            power = 12 + np.random.normal(0, 0.5)
            
        return MachineState(
            machine_id=machine_id,
            status="Operational",
            temperature=temperature,
            pressure=pressure,
            vibration=vibration,
            power_consumption=power,
            last_maintenance=now - timedelta(days=np.random.randint(10, 30)),
            next_maintenance=now + timedelta(days=np.random.randint(5, 15))
        )
        
    def get_machine_state(self, machine_id: str) -> MachineState:
        """Get the current state of a machine"""
        if machine_id not in self.machine_states:
            self.sync_with_physical_asset(machine_id)
        return self.machine_states[machine_id]
        
    def predict_future_state(self, machine_id: str, hours_ahead: int) -> Dict[str, Any]:
        """Predict the future state of a machine"""
        if machine_id not in self.machine_states:
            self.sync_with_physical_asset(machine_id)
            
        current_state = self.machine_states[machine_id]
        
        # Generate a simple prediction based on current state
        # In a real implementation, this would use the neural network model
        predictions = {
            'timestamps': [(datetime.now() + timedelta(hours=h)).isoformat() for h in range(hours_ahead + 1)],
            'temperature': [],
            'pressure': [],
            'vibration': [],
            'power_consumption': []
        }
        
        # Simple prediction logic with some randomness and trends
        temp_trend = np.random.choice([-0.1, 0, 0.1, 0.2])
        pressure_trend = np.random.choice([-0.2, -0.1, 0, 0.1])
        vibration_trend = np.random.choice([0, 0.001, 0.002, 0.003])
        power_trend = np.random.choice([-0.05, 0, 0.05, 0.1])
        
        for h in range(hours_ahead + 1):
            predictions['temperature'].append(current_state.temperature + h * temp_trend + np.random.normal(0, 0.5))
            predictions['pressure'].append(current_state.pressure + h * pressure_trend + np.random.normal(0, 0.2))
            predictions['vibration'].append(current_state.vibration + h * vibration_trend + np.random.normal(0, 0.005))
            predictions['power_consumption'].append(current_state.power_consumption + h * power_trend + np.random.normal(0, 0.1))
            
        return predictions
    
    def _encode_status(self, status: str) -> List[float]:
        """Encode a machine status into a numerical representation"""
        status_map = {"Operational": [1, 0, 0], "Maintenance": [0, 1, 0], "Error": [0, 0, 1]}
        return status_map.get(status, [0, 0, 0])
        
    def add_sensor_reading(self, reading: SensorData) -> None:
        """Add a sensor reading to the digital twin"""
        if reading.sensor_id not in self.sensor_readings:
            self.sensor_readings[reading.sensor_id] = []
            
        self.sensor_readings[reading.sensor_id].append(reading)
        
        # Keep only the latest 1000 readings
        if len(self.sensor_readings[reading.sensor_id]) > 1000:
            self.sensor_readings[reading.sensor_id] = self.sensor_readings[reading.sensor_id][-1000:]
            
    def get_sensor_history(self, sensor_id: str, limit: int = 100) -> List[SensorData]:
        """Get historical sensor readings"""
        if sensor_id not in self.sensor_readings:
            return []
            
        return self.sensor_readings[sensor_id][-limit:]
        
    def analyze_performance(self, machine_id: str) -> Dict[str, Any]:
        """Analyze the performance of a machine"""
        if machine_id not in self.machine_states:
            self.sync_with_physical_asset(machine_id)
            
        state = self.machine_states[machine_id]
        
        # Calculate performance metrics based on the current state
        performance = {
            'efficiency': min(100, max(50, 95 - state.vibration * 100 - (state.temperature - 60) * 0.5)),
            'health_score': min(100, max(50, 90 - state.vibration * 200 - abs(state.pressure - 100) * 0.2)),
            'energy_efficiency': min(100, max(50, 85 - state.power_consumption * 2)),
            'overall_score': 0  # Will be calculated as the average of the other scores
        }
        
        performance['overall_score'] = (
            performance['efficiency'] + 
            performance['health_score'] + 
            performance['energy_efficiency']
        ) / 3
        
        return performance
        
    def get_maintenance_recommendations(self, machine_id: str) -> List[Dict[str, Any]]:
        """Get maintenance recommendations for a machine"""
        if machine_id not in self.machine_states:
            self.sync_with_physical_asset(machine_id)
            
        state = self.machine_states[machine_id]
        recommendations = []
        
        # Check temperature
        if state.temperature > 70:
            recommendations.append({
                'component': 'Cooling System',
                'issue': 'High Temperature',
                'current_value': f"{state.temperature:.1f}°C",
                'threshold': '70°C',
                'priority': 'High' if state.temperature > 75 else 'Medium',
                'recommendation': 'Check cooling system and ensure proper ventilation'
            })
            
        # Check vibration
        if state.vibration > 0.25:
            recommendations.append({
                'component': 'Bearings',
                'issue': 'High Vibration',
                'current_value': f"{state.vibration:.3f} mm/s",
                'threshold': '0.25 mm/s',
                'priority': 'High' if state.vibration > 0.3 else 'Medium',
                'recommendation': 'Inspect bearings and alignment'
            })
            
        # Check pressure
        if abs(state.pressure - 100) > 10:
            recommendations.append({
                'component': 'Hydraulic System',
                'issue': 'Pressure Deviation',
                'current_value': f"{state.pressure:.1f} PSI",
                'threshold': '90-110 PSI',
                'priority': 'Medium',
                'recommendation': 'Check hydraulic system for leaks or restrictions'
            })
            
        # Check power consumption
        if state.power_consumption > 16:
            recommendations.append({
                'component': 'Drive System',
                'issue': 'High Power Consumption',
                'current_value': f"{state.power_consumption:.1f} kW",
                'threshold': '16 kW',
                'priority': 'Low',
                'recommendation': 'Optimize machining parameters to reduce power consumption'
            })
            
        # Add a routine maintenance recommendation if due soon
        days_to_maintenance = (state.next_maintenance - datetime.now()).days
        if days_to_maintenance < 7:
            recommendations.append({
                'component': 'All Systems',
                'issue': 'Scheduled Maintenance Due',
                'current_value': f"{days_to_maintenance} days",
                'threshold': '7 days',
                'priority': 'High' if days_to_maintenance <= 2 else 'Medium',
                'recommendation': 'Perform scheduled maintenance according to machine manual'
            })
            
        return recommendations
        
    def get_optimization_suggestions(self, machine_id: str) -> List[Dict[str, Any]]:
        """Get optimization suggestions for a machine"""
        if machine_id not in self.machine_states:
            self.sync_with_physical_asset(machine_id)
            
        state = self.machine_states[machine_id]
        suggestions = []
        
        # Suggest optimizations based on current state
        if state.temperature > 65:
            suggestions.append({
                'parameter': 'Spindle Speed',
                'current_setting': 'High',
                'suggested_setting': 'Medium',
                'expected_improvement': 'Reduce temperature by ~3°C',
                'impact': 'May increase cycle time by 5-10%'
            })
            
        if state.vibration > 0.2:
            suggestions.append({
                'parameter': 'Depth of Cut',
                'current_setting': 'Deep',
                'suggested_setting': 'Moderate',
                'expected_improvement': 'Reduce vibration by ~20%',
                'impact': 'May require additional passes'
            })
            
        if state.power_consumption > 14:
            suggestions.append({
                'parameter': 'Material Removal Rate',
                'current_setting': 'High',
                'suggested_setting': 'Medium',
                'expected_improvement': 'Reduce power consumption by ~15%',
                'impact': 'Increased cycle time'
            })
            
        return suggestions
