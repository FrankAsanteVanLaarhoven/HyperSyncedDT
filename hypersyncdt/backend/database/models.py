from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .config import db_config

Base = db_config.Base

class Machine(Base):
    """Machine entity representing physical equipment."""
    __tablename__ = "machines"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    type = Column(String)
    location = Column(String)
    manufacturer = Column(String)
    installation_date = Column(DateTime, default=datetime.utcnow)
    last_maintenance = Column(DateTime)
    status = Column(String)  # operational, maintenance, offline
    configuration = Column(JSON)
    
    # Relationships
    sensors = relationship("Sensor", back_populates="machine")
    maintenance_records = relationship("MaintenanceRecord", back_populates="machine")
    anomalies = relationship("Anomaly", back_populates="machine")

class Sensor(Base):
    """Sensor entity for data collection."""
    __tablename__ = "sensors"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"))
    name = Column(String)
    type = Column(String)  # temperature, pressure, vibration, etc.
    unit = Column(String)
    location = Column(String)
    calibration_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    configuration = Column(JSON)
    
    # Relationships
    machine = relationship("Machine", back_populates="sensors")
    readings = relationship("SensorReading", back_populates="sensor")

class SensorReading(Base):
    """Sensor reading data."""
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(Integer, ForeignKey("sensors.id"))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    value = Column(Float)
    quality = Column(Float)  # Data quality score
    metadata = Column(JSON)
    
    # Relationships
    sensor = relationship("Sensor", back_populates="readings")

class MaintenanceRecord(Base):
    """Maintenance activity records."""
    __tablename__ = "maintenance_records"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"))
    maintenance_type = Column(String)  # preventive, corrective, predictive
    description = Column(String)
    technician = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    cost = Column(Float)
    parts_replaced = Column(JSON)
    notes = Column(String)
    
    # Relationships
    machine = relationship("Machine", back_populates="maintenance_records")

class Anomaly(Base):
    """Detected anomalies in machine operation."""
    __tablename__ = "anomalies"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"))
    detection_time = Column(DateTime, default=datetime.utcnow)
    anomaly_type = Column(String)
    severity = Column(Float)
    description = Column(String)
    resolved = Column(Boolean, default=False)
    resolution_time = Column(DateTime)
    resolution_notes = Column(String)
    sensor_data = Column(JSON)  # Relevant sensor readings at time of anomaly
    
    # Relationships
    machine = relationship("Machine", back_populates="anomalies")

class DigitalTwin(Base):
    """Digital twin model configuration and state."""
    __tablename__ = "digital_twins"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"), unique=True)
    model_version = Column(String)
    last_sync = Column(DateTime)
    state = Column(JSON)  # Current state of the digital twin
    parameters = Column(JSON)  # Model parameters
    performance_metrics = Column(JSON)  # Accuracy, latency, etc.
    calibration_history = Column(JSON)
    
    # Relationships
    machine = relationship("Machine")
    simulations = relationship("Simulation", back_populates="digital_twin")

class Simulation(Base):
    """Simulation runs and results."""
    __tablename__ = "simulations"
    
    id = Column(Integer, primary_key=True, index=True)
    digital_twin_id = Column(Integer, ForeignKey("digital_twins.id"))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    scenario = Column(JSON)  # Simulation parameters and conditions
    results = Column(JSON)  # Simulation outputs
    metrics = Column(JSON)  # Performance metrics
    status = Column(String)  # running, completed, failed
    
    # Relationships
    digital_twin = relationship("DigitalTwin", back_populates="simulations")

class MLModel(Base):
    """Machine learning model metadata and tracking."""
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    version = Column(String)
    type = Column(String)  # predictive_maintenance, anomaly_detection, etc.
    creation_date = Column(DateTime, default=datetime.utcnow)
    last_training = Column(DateTime)
    performance_metrics = Column(JSON)
    hyperparameters = Column(JSON)
    feature_importance = Column(JSON)
    deployment_status = Column(String)  # development, staging, production
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model")

class Prediction(Base):
    """Model predictions and actual outcomes."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_data = Column(JSON)
    prediction = Column(JSON)
    actual_outcome = Column(JSON)
    confidence = Column(Float)
    performance_metrics = Column(JSON)
    
    # Relationships
    model = relationship("MLModel", back_populates="predictions")

# Create all tables
def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=db_config.engine)

# Example usage
if __name__ == "__main__":
    create_tables() 