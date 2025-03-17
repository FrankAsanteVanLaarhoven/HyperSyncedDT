import h5py
import numpy as np
import os
import time
from datetime import datetime
import json

class HDF5QuantumMesh:
    """
    HDF5 Quantum Mesh for high-performance data storage and retrieval
    
    This class provides an advanced data storage solution using HDF5 format
    with optimized data structures for manufacturing digital twin applications.
    """
    
    def __init__(self, file_path="database/quantum_mesh_data.h5"):
        self.file_path = file_path
        self._ensure_file_exists()
        
    def _ensure_file_exists(self):
        """Ensure the HDF5 file exists and has the correct structure"""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        # Create file if it doesn't exist
        if not os.path.exists(self.file_path):
            with h5py.File(self.file_path, 'w') as f:
                # Create main groups
                f.create_group('sensor_data')
                f.create_group('predictions')
                f.create_group('system_metrics')
                f.create_group('metadata')
                
                # Add metadata
                metadata = f['metadata']
                metadata.attrs['created_at'] = datetime.now().isoformat()
                metadata.attrs['version'] = '1.0.0'
                metadata.attrs['description'] = 'HDF5 Quantum Mesh for HyperSyncedDT'
    
    def store_prediction(self, temp, load, prediction, metadata=None):
        """Store a tool wear prediction in the quantum mesh"""
        with h5py.File(self.file_path, 'a') as f:
            predictions = f['predictions']
            
            # Create a timestamp-based group for this prediction
            timestamp = datetime.now().isoformat()
            pred_group = predictions.create_group(timestamp)
            
            # Store prediction data
            pred_group.create_dataset('temperature', data=temp)
            pred_group.create_dataset('load', data=load)
            pred_group.create_dataset('prediction', data=prediction)
            
            # Store metadata if provided
            if metadata:
                meta_group = pred_group.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value
                    else:
                        # For complex types, store as JSON
                        meta_group.attrs[key] = json.dumps(value)
            
            # Update last_updated in metadata
            f['metadata'].attrs['last_updated'] = timestamp
            
            return timestamp
    
    def get_recent_predictions(self, limit=10):
        """Get the most recent predictions from the quantum mesh"""
        with h5py.File(self.file_path, 'r') as f:
            predictions = f['predictions']
            
            # Get all prediction timestamps
            timestamps = list(predictions.keys())
            timestamps.sort(reverse=True)
            
            # Limit the number of results
            timestamps = timestamps[:limit]
            
            results = []
            for ts in timestamps:
                pred_group = predictions[ts]
                
                # Extract prediction data
                temp = float(pred_group['temperature'][()])
                load = float(pred_group['load'][()])
                prediction = float(pred_group['prediction'][()])
                
                # Extract metadata if available
                metadata = {}
                if 'metadata' in pred_group:
                    meta_group = pred_group['metadata']
                    for key in meta_group.attrs:
                        value = meta_group.attrs[key]
                        # Try to parse JSON for complex types
                        if isinstance(value, str) and value.startswith('{'):
                            try:
                                value = json.loads(value)
                            except:
                                pass
                        metadata[key] = value
                
                results.append({
                    'timestamp': ts,
                    'temperature': temp,
                    'load': load,
                    'prediction': prediction,
                    'metadata': metadata
                })
            
            return results
    
    def store_sensor_data(self, sensor_id, value, timestamp=None):
        """Store sensor data in the quantum mesh"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        with h5py.File(self.file_path, 'a') as f:
            sensor_data = f['sensor_data']
            
            # Create sensor group if it doesn't exist
            if sensor_id not in sensor_data:
                sensor_data.create_group(sensor_id)
            
            sensor_group = sensor_data[sensor_id]
            
            # Store the data point
            data_point = sensor_group.create_dataset(timestamp, data=value)
            data_point.attrs['timestamp'] = timestamp
            
            return timestamp
    
    def get_sensor_data(self, sensor_id, start_time=None, end_time=None, limit=100):
        """Get sensor data from the quantum mesh within a time range"""
        with h5py.File(self.file_path, 'r') as f:
            sensor_data = f['sensor_data']
            
            if sensor_id not in sensor_data:
                return []
                
            sensor_group = sensor_data[sensor_id]
            
            # Get all timestamps
            timestamps = list(sensor_group.keys())
            
            # Filter by time range if specified
            if start_time:
                timestamps = [ts for ts in timestamps if ts >= start_time]
            if end_time:
                timestamps = [ts for ts in timestamps if ts <= end_time]
                
            # Sort timestamps
            timestamps.sort(reverse=True)
            
            # Limit the number of results
            timestamps = timestamps[:limit]
            
            results = []
            for ts in timestamps:
                value = float(sensor_group[ts][()])
                results.append({
                    'timestamp': ts,
                    'value': value
                })
                
            return results
    
    def store_system_metric(self, metric_name, value):
        """Store a system metric in the quantum mesh"""
        timestamp = datetime.now().isoformat()
        
        with h5py.File(self.file_path, 'a') as f:
            system_metrics = f['system_metrics']
            
            # Create metric group if it doesn't exist
            if metric_name not in system_metrics:
                system_metrics.create_group(metric_name)
            
            metric_group = system_metrics[metric_name]
            
            # Store the metric
            metric_group.create_dataset(timestamp, data=value)
            
            return timestamp
    
    def get_system_metrics(self, metric_name, limit=100):
        """Get system metrics from the quantum mesh"""
        with h5py.File(self.file_path, 'r') as f:
            system_metrics = f['system_metrics']
            
            if metric_name not in system_metrics:
                return []
                
            metric_group = system_metrics[metric_name]
            
            # Get all timestamps
            timestamps = list(metric_group.keys())
            
            # Sort timestamps
            timestamps.sort(reverse=True)
            
            # Limit the number of results
            timestamps = timestamps[:limit]
            
            results = []
            for ts in timestamps:
                value = float(metric_group[ts][()])
                results.append({
                    'timestamp': ts,
                    'value': value
                })
                
            return results
    
    def get_mesh_info(self):
        """Get information about the quantum mesh"""
        with h5py.File(self.file_path, 'r') as f:
            metadata = f['metadata']
            
            info = {
                'created_at': metadata.attrs.get('created_at', 'Unknown'),
                'version': metadata.attrs.get('version', 'Unknown'),
                'description': metadata.attrs.get('description', 'Unknown'),
                'last_updated': metadata.attrs.get('last_updated', 'Never')
            }
            
            # Count items in each group
            info['prediction_count'] = len(f['predictions'].keys())
            info['sensor_count'] = len(f['sensor_data'].keys())
            info['metric_count'] = len(f['system_metrics'].keys())
            
            # Calculate file size
            info['file_size_mb'] = os.path.getsize(self.file_path) / (1024 * 1024)
            
            return info

# Create a singleton instance
quantum_mesh = HDF5QuantumMesh() 