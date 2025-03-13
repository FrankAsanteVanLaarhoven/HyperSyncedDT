from .qpiagn import get_qpiagn_model
from .hdf5_quantum_mesh import quantum_mesh
from .sensor_simulator import sensor_simulator
from .anomaly_detector import anomaly_detector

# Initialize the sensor simulator and anomaly detector
def initialize_models():
    # Generate historical data
    sensor_simulator.generate_historical_data(days=7)
    
    # Connect sensor simulator to quantum mesh and anomaly detector
    def sensor_callback(sensor_id, value, timestamp):
        # Store in quantum mesh
        quantum_mesh.store_sensor_data(sensor_id, value, timestamp)
        
        # Check for anomalies
        anomaly_detector.add_sensor_data(sensor_id, value, timestamp)
    
    # Add callback to sensor simulator
    sensor_simulator.add_callback(sensor_callback)
    
    # Start the sensor simulator and anomaly detector
    sensor_simulator.start()
    anomaly_detector.start()
    
    return {
        'sensor_simulator': sensor_simulator,
        'anomaly_detector': anomaly_detector,
        'quantum_mesh': quantum_mesh
    } 