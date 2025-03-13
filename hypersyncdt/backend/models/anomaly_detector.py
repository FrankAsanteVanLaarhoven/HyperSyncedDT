import numpy as np
from scipy import stats
import threading
import time
from datetime import datetime, timedelta
import json

class AnomalyDetector:
    """
    Anomaly detection system for the HyperSyncedDT platform
    
    Uses statistical methods and machine learning to detect anomalies
    in sensor data and predict potential failures.
    """
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.sensor_baselines = {}
        self.anomalies = []
        self.max_anomalies = 100
        self.callbacks = []
        
        # Detection parameters
        self.z_score_threshold = 3.0  # Number of standard deviations for anomaly
        self.window_size = 60  # Number of data points to consider
        self.min_data_points = 30  # Minimum data points needed for detection
        
    def start(self, interval=5.0):
        """Start the anomaly detector"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the anomaly detector"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _run(self, interval):
        """Run the anomaly detector"""
        while self.running:
            self._detect_anomalies()
            time.sleep(interval)
            
    def _detect_anomalies(self):
        """Detect anomalies in sensor data"""
        # This would typically use the sensor simulator to get data
        # For now, we'll just simulate some anomalies
        pass
        
    def add_sensor_data(self, sensor_id, value, timestamp):
        """Add new sensor data and check for anomalies"""
        # Initialize baseline if needed
        if sensor_id not in self.sensor_baselines:
            self.sensor_baselines[sensor_id] = {
                'values': [],
                'timestamps': [],
                'mean': None,
                'std': None
            }
            
        baseline = self.sensor_baselines[sensor_id]
        
        # Add new data point
        baseline['values'].append(value)
        baseline['timestamps'].append(timestamp)
        
        # Trim to window size
        if len(baseline['values']) > self.window_size:
            baseline['values'] = baseline['values'][-self.window_size:]
            baseline['timestamps'] = baseline['timestamps'][-self.window_size:]
            
        # Update statistics
        if len(baseline['values']) >= self.min_data_points:
            baseline['mean'] = np.mean(baseline['values'])
            baseline['std'] = np.std(baseline['values'])
            
            # Check for anomaly
            if baseline['std'] > 0:  # Avoid division by zero
                z_score = abs(value - baseline['mean']) / baseline['std']
                
                if z_score > self.z_score_threshold:
                    self._record_anomaly(sensor_id, value, timestamp, z_score)
                    
    def _record_anomaly(self, sensor_id, value, timestamp, score):
        """Record an anomaly"""
        anomaly = {
            'sensor_id': sensor_id,
            'value': value,
            'timestamp': timestamp,
            'score': score,
            'detected_at': datetime.now().isoformat()
        }
        
        self.anomalies.append(anomaly)
        
        # Trim anomalies list if needed
        if len(self.anomalies) > self.max_anomalies:
            self.anomalies = self.anomalies[-self.max_anomalies:]
            
        # Call callbacks
        for callback in self.callbacks:
            callback(anomaly)
            
    def get_recent_anomalies(self, limit=10):
        """Get recent anomalies"""
        return self.anomalies[-limit:]
        
    def get_sensor_anomalies(self, sensor_id, limit=10):
        """Get anomalies for a specific sensor"""
        sensor_anomalies = [a for a in self.anomalies if a['sensor_id'] == sensor_id]
        return sensor_anomalies[-limit:]
        
    def add_callback(self, callback):
        """Add a callback for anomaly detection"""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback):
        """Remove a callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def set_detection_parameters(self, z_score_threshold=None, window_size=None, min_data_points=None):
        """Set detection parameters"""
        if z_score_threshold is not None:
            self.z_score_threshold = z_score_threshold
        if window_size is not None:
            self.window_size = window_size
        if min_data_points is not None:
            self.min_data_points = min_data_points
            
    def get_detection_parameters(self):
        """Get current detection parameters"""
        return {
            'z_score_threshold': self.z_score_threshold,
            'window_size': self.window_size,
            'min_data_points': self.min_data_points
        }
        
    def predict_failures(self, sensor_id, horizon_hours=24):
        """
        Predict potential failures based on anomaly patterns
        
        This is a simplified implementation. A real system would use
        more sophisticated machine learning models.
        """
        # Get anomalies for this sensor
        sensor_anomalies = [a for a in self.anomalies if a['sensor_id'] == sensor_id]
        
        if len(sensor_anomalies) < 5:
            return {
                'prediction': 'Insufficient data',
                'probability': 0.0,
                'confidence': 0.0
            }
            
        # Calculate anomaly frequency
        now = datetime.now()
        one_day_ago = now - timedelta(days=1)
        one_day_ago_iso = one_day_ago.isoformat()
        
        recent_anomalies = [a for a in sensor_anomalies if a['timestamp'] > one_day_ago_iso]
        anomaly_frequency = len(recent_anomalies) / 24.0  # Anomalies per hour
        
        # Simple prediction model
        if anomaly_frequency > 0.5:  # More than one anomaly every 2 hours
            failure_probability = min(0.9, anomaly_frequency * 0.2)
            confidence = min(0.8, 0.4 + anomaly_frequency * 0.1)
            prediction = 'High risk of failure'
        elif anomaly_frequency > 0.2:  # More than one anomaly every 5 hours
            failure_probability = min(0.7, anomaly_frequency * 0.15)
            confidence = min(0.7, 0.3 + anomaly_frequency * 0.1)
            prediction = 'Moderate risk of failure'
        elif anomaly_frequency > 0.05:  # More than one anomaly per day
            failure_probability = min(0.3, anomaly_frequency * 0.1)
            confidence = min(0.6, 0.2 + anomaly_frequency * 0.1)
            prediction = 'Low risk of failure'
        else:
            failure_probability = max(0.01, anomaly_frequency * 0.05)
            confidence = min(0.5, 0.1 + anomaly_frequency * 0.1)
            prediction = 'Very low risk of failure'
            
        return {
            'prediction': prediction,
            'probability': failure_probability,
            'confidence': confidence,
            'horizon_hours': horizon_hours,
            'anomaly_frequency': anomaly_frequency
        }

# Create a singleton instance
anomaly_detector = AnomalyDetector() 