import numpy as np
import time
import threading
import random
from datetime import datetime, timedelta

class SensorSimulator:
    """
    Sensor data simulator for the HyperSyncedDT platform
    
    Simulates realistic sensor data for manufacturing equipment,
    including temperature, vibration, pressure, and other metrics.
    """
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.sensors = {
            'temperature': {
                'value': 600.0,
                'min': 0.0,
                'max': 1200.0,
                'noise': 5.0,
                'trend': 0.0,
                'unit': '°C'
            },
            'load': {
                'value': 500.0,
                'min': 0.0,
                'max': 1000.0,
                'noise': 10.0,
                'trend': 0.0,
                'unit': 'N'
            },
            'vibration': {
                'value': 0.5,
                'min': 0.0,
                'max': 10.0,
                'noise': 0.2,
                'trend': 0.0,
                'unit': 'mm/s'
            },
            'pressure': {
                'value': 100.0,
                'min': 0.0,
                'max': 200.0,
                'noise': 2.0,
                'trend': 0.0,
                'unit': 'kPa'
            },
            'rpm': {
                'value': 1000.0,
                'min': 0.0,
                'max': 2000.0,
                'noise': 20.0,
                'trend': 0.0,
                'unit': 'RPM'
            },
            'humidity': {
                'value': 45.0,
                'min': 10.0,
                'max': 90.0,
                'noise': 1.0,
                'trend': 0.0,
                'unit': '%'
            },
            'current': {
                'value': 15.0,
                'min': 0.0,
                'max': 30.0,
                'noise': 0.5,
                'trend': 0.0,
                'unit': 'A'
            },
            'voltage': {
                'value': 220.0,
                'min': 200.0,
                'max': 240.0,
                'noise': 1.0,
                'trend': 0.0,
                'unit': 'V'
            }
        }
        
        # Callbacks for when sensor data is updated
        self.callbacks = []
        
        # Historical data
        self.history = {sensor: [] for sensor in self.sensors}
        self.max_history_points = 1000
        
    def start(self, interval=1.0):
        """Start the sensor simulator"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the sensor simulator"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _run(self, interval):
        """Run the sensor simulator"""
        while self.running:
            self._update_sensors()
            time.sleep(interval)
            
    def _update_sensors(self):
        """Update all sensor values"""
        timestamp = datetime.now().isoformat()
        
        for sensor_id, sensor in self.sensors.items():
            # Apply trend
            sensor['value'] += sensor['trend']
            
            # Apply random noise
            noise = np.random.normal(0, sensor['noise'])
            sensor['value'] += noise
            
            # Ensure value is within bounds
            sensor['value'] = max(sensor['min'], min(sensor['max'], sensor['value']))
            
            # Add to history
            self.history[sensor_id].append({
                'timestamp': timestamp,
                'value': sensor['value']
            })
            
            # Trim history if needed
            if len(self.history[sensor_id]) > self.max_history_points:
                self.history[sensor_id] = self.history[sensor_id][-self.max_history_points:]
            
            # Call callbacks
            for callback in self.callbacks:
                callback(sensor_id, sensor['value'], timestamp)
                
    def set_sensor_trend(self, sensor_id, trend):
        """Set the trend for a sensor"""
        if sensor_id in self.sensors:
            self.sensors[sensor_id]['trend'] = trend
            
    def set_sensor_value(self, sensor_id, value):
        """Set the value for a sensor"""
        if sensor_id in self.sensors:
            self.sensors[sensor_id]['value'] = max(
                self.sensors[sensor_id]['min'],
                min(self.sensors[sensor_id]['max'], value)
            )
            
    def get_sensor_value(self, sensor_id):
        """Get the current value for a sensor"""
        if sensor_id in self.sensors:
            return self.sensors[sensor_id]['value']
        return None
        
    def get_all_sensor_values(self):
        """Get all current sensor values"""
        return {
            sensor_id: {
                'value': sensor['value'],
                'unit': sensor['unit']
            }
            for sensor_id, sensor in self.sensors.items()
        }
        
    def get_sensor_history(self, sensor_id, limit=100):
        """Get historical data for a sensor"""
        if sensor_id in self.history:
            return self.history[sensor_id][-limit:]
        return []
        
    def add_callback(self, callback):
        """Add a callback for sensor updates"""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback):
        """Remove a callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def simulate_anomaly(self, sensor_id, duration=60, magnitude=2.0):
        """Simulate an anomaly in a sensor"""
        if sensor_id not in self.sensors:
            return False
            
        # Create a separate thread to handle the anomaly
        def _anomaly_thread():
            # Save original trend
            original_trend = self.sensors[sensor_id]['trend']
            original_noise = self.sensors[sensor_id]['noise']
            
            # Apply anomaly
            self.sensors[sensor_id]['trend'] = magnitude * original_trend + magnitude
            self.sensors[sensor_id]['noise'] = magnitude * original_noise
            
            # Wait for duration
            time.sleep(duration)
            
            # Restore original values
            self.sensors[sensor_id]['trend'] = original_trend
            self.sensors[sensor_id]['noise'] = original_noise
            
        thread = threading.Thread(target=_anomaly_thread)
        thread.daemon = True
        thread.start()
        
        return True
        
    def generate_historical_data(self, days=7, interval_minutes=15):
        """Generate historical data for all sensors"""
        # Clear existing history
        self.history = {sensor: [] for sensor in self.sensors}
        
        # Calculate number of data points
        points_per_day = 24 * 60 // interval_minutes
        total_points = days * points_per_day
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = [
            (start_time + timedelta(minutes=i*interval_minutes)).isoformat()
            for i in range(total_points)
        ]
        
        # Generate data for each sensor
        for sensor_id, sensor in self.sensors.items():
            base_value = (sensor['min'] + sensor['max']) / 2
            
            # Generate random walk
            values = [base_value]
            for i in range(1, total_points):
                # Add some randomness
                noise = np.random.normal(0, sensor['noise'])
                
                # Add daily patterns (higher during day, lower at night)
                hour = (start_time + timedelta(minutes=i*interval_minutes)).hour
                daily_factor = np.sin(hour / 24 * 2 * np.pi) * 0.2
                
                # Add weekly patterns
                day = (start_time + timedelta(minutes=i*interval_minutes)).weekday()
                weekly_factor = (1 - 0.3 * (day >= 5))  # Lower on weekends
                
                new_value = values[-1] + noise + daily_factor * sensor['max'] * 0.1
                new_value *= weekly_factor
                
                # Ensure value is within bounds
                new_value = max(sensor['min'], min(sensor['max'], new_value))
                values.append(new_value)
            
            # Store in history
            self.history[sensor_id] = [
                {'timestamp': ts, 'value': val}
                for ts, val in zip(timestamps, values)
            ]
            
        return True

# Create a singleton instance
sensor_simulator = SensorSimulator() 