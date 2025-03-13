from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
import sqlite3
import os
from typing import Annotated, Dict, List, Optional, Any
import numpy as np
from datetime import datetime

# Import our advanced models
from models import get_qpiagn_model, quantum_mesh, sensor_simulator, anomaly_detector, initialize_models

app = FastAPI(
    title="HyperSyncedDT API",
    description="Advanced Digital Twin API for Manufacturing",
    version="2.0.0"
)

# Database connection function
def get_db():
    db_path = "database/hyper_synced_dt.db"
    # Ensure the database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Tool Wear Prediction Model Input
class ToolWearPrediction(BaseModel):
    temp: float
    load: float

# Sensor Data Model
class SensorData(BaseModel):
    sensor_id: str
    value: float
    timestamp: Optional[str] = None

# Anomaly Simulation Model
class AnomalySimulation(BaseModel):
    sensor_id: str
    duration: int = 60  # seconds
    magnitude: float = 2.0

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    # Initialize models
    initialize_models()
    
    # Load Q-PIAGN model
    get_qpiagn_model()

# Health Check
@app.get("/")
def read_root():
    return {
        "message": "HyperSyncedDT Backend is running!",
        "version": "2.0.0",
        "features": [
            "Q-PIAGN Prediction Model",
            "HDF5 Quantum Mesh",
            "Real-time Sensor Simulation",
            "Anomaly Detection",
            "Failure Prediction"
        ]
    }

# Tool Wear Prediction with Q-PIAGN
@app.post("/predict-tool-wear")
def predict_tool_wear(data: ToolWearPrediction, conn: Annotated[sqlite3.Connection, Depends(get_db)]):
    # Get the Q-PIAGN model
    model = get_qpiagn_model()
    
    # Make prediction
    X = np.array([[data.temp, data.load]])
    prediction = float(model.predict(X)[0])
    
    # Store in SQLite database
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tool_wear (temp, load, prediction) VALUES (?, ?, ?)",
        (data.temp, data.load, prediction)
    )
    conn.commit()
    
    # Store in Quantum Mesh
    quantum_mesh.store_prediction(
        data.temp, 
        data.load, 
        prediction,
        metadata={
            "model": "Q-PIAGN",
            "version": "1.0.0"
        }
    )
    
    return {
        "prediction": prediction,
        "model": "Q-PIAGN",
        "confidence": 0.95,
        "timestamp": datetime.now().isoformat()
    }

# Get historical predictions from SQLite
@app.get("/history")
def get_history(conn: Annotated[sqlite3.Connection, Depends(get_db)], limit: int = 10):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tool_wear ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    return {"history": [dict(row) for row in rows]}

# Get historical predictions from Quantum Mesh
@app.get("/quantum-history")
def get_quantum_history(limit: int = 10):
    predictions = quantum_mesh.get_recent_predictions(limit)
    return {"predictions": predictions}

# Get all sensor data
@app.get("/sensors")
def get_sensors():
    return {"sensors": sensor_simulator.get_all_sensor_values()}

# Get specific sensor data
@app.get("/sensors/{sensor_id}")
def get_sensor(sensor_id: str, limit: int = 100):
    value = sensor_simulator.get_sensor_value(sensor_id)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
        
    history = sensor_simulator.get_sensor_history(sensor_id, limit)
    
    return {
        "sensor_id": sensor_id,
        "current_value": value,
        "history": history
    }

# Set sensor value
@app.post("/sensors/{sensor_id}")
def set_sensor(sensor_id: str, data: SensorData):
    if sensor_id != data.sensor_id:
        raise HTTPException(status_code=400, detail="Sensor ID in path must match body")
        
    success = sensor_simulator.set_sensor_value(sensor_id, data.value)
    if not success:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
        
    return {"success": True, "sensor_id": sensor_id, "value": data.value}

# Simulate anomaly
@app.post("/simulate-anomaly")
def simulate_anomaly(data: AnomalySimulation):
    success = sensor_simulator.simulate_anomaly(
        data.sensor_id, 
        duration=data.duration,
        magnitude=data.magnitude
    )
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Sensor {data.sensor_id} not found")
        
    return {
        "success": True,
        "message": f"Anomaly simulation started for {data.sensor_id}",
        "duration": data.duration,
        "magnitude": data.magnitude
    }

# Get recent anomalies
@app.get("/anomalies")
def get_anomalies(limit: int = 10):
    return {"anomalies": anomaly_detector.get_recent_anomalies(limit)}

# Get anomalies for a specific sensor
@app.get("/anomalies/{sensor_id}")
def get_sensor_anomalies(sensor_id: str, limit: int = 10):
    return {"anomalies": anomaly_detector.get_sensor_anomalies(sensor_id, limit)}

# Predict failures
@app.get("/predict-failure/{sensor_id}")
def predict_failure(sensor_id: str, horizon_hours: int = 24):
    prediction = anomaly_detector.predict_failures(sensor_id, horizon_hours)
    return prediction

# Get system metrics
@app.get("/system-metrics")
def get_system_metrics():
    # Get information about the quantum mesh
    mesh_info = quantum_mesh.get_mesh_info()
    
    # Get anomaly detector parameters
    detector_params = anomaly_detector.get_detection_parameters()
    
    return {
        "quantum_mesh": mesh_info,
        "anomaly_detector": detector_params,
        "sensor_count": len(sensor_simulator.sensors),
        "timestamp": datetime.now().isoformat()
    }
