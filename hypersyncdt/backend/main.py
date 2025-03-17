from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from datetime import datetime, timedelta

from database.config import db_config
from database.models import (
    Machine, Sensor, SensorReading, MaintenanceRecord,
    Anomaly, DigitalTwin, Simulation, MLModel, Prediction
)
from app.ml.predictive_maintenance import PredictiveMaintenanceModel
from app.ml.anomaly_detector import AnomalyDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HyperSyncDT API",
    description="Digital Twin System API for Industrial Equipment Monitoring",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML models
predictive_model = PredictiveMaintenanceModel()
anomaly_detector = AnomalyDetector()

# Dependency to get database session
def get_db():
    db = db_config.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "HyperSyncDT API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "database": db_config.verify_connection()
    }

# Machine endpoints
@app.get("/machines", response_model=List[dict])
async def get_machines(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None
):
    """Get list of machines with optional filtering."""
    query = db.query(Machine)
    if status:
        query = query.filter(Machine.status == status)
    return query.offset(skip).limit(limit).all()

@app.get("/machines/{machine_id}")
async def get_machine(machine_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific machine."""
    machine = db.query(Machine).filter(Machine.id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine

@app.get("/machines/{machine_id}/sensors")
async def get_machine_sensors(machine_id: int, db: Session = Depends(get_db)):
    """Get all sensors for a specific machine."""
    machine = db.query(Machine).filter(Machine.id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine.sensors

@app.get("/machines/{machine_id}/readings")
async def get_machine_readings(
    machine_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get sensor readings for a specific machine within a time range."""
    if not start_time:
        start_time = datetime.utcnow() - timedelta(hours=24)
    if not end_time:
        end_time = datetime.utcnow()
    
    readings = (
        db.query(SensorReading)
        .join(Sensor)
        .filter(Sensor.machine_id == machine_id)
        .filter(SensorReading.timestamp.between(start_time, end_time))
        .all()
    )
    return readings

# Digital Twin endpoints
@app.get("/digital-twins/{machine_id}")
async def get_digital_twin(machine_id: int, db: Session = Depends(get_db)):
    """Get digital twin information for a specific machine."""
    digital_twin = (
        db.query(DigitalTwin)
        .filter(DigitalTwin.machine_id == machine_id)
        .first()
    )
    if not digital_twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    return digital_twin

@app.post("/digital-twins/{machine_id}/simulate")
async def run_simulation(
    machine_id: int,
    scenario: dict,
    db: Session = Depends(get_db)
):
    """Run a simulation on a digital twin."""
    digital_twin = (
        db.query(DigitalTwin)
        .filter(DigitalTwin.machine_id == machine_id)
        .first()
    )
    if not digital_twin:
        raise HTTPException(status_code=404, detail="Digital twin not found")
    
    # Create simulation record
    simulation = Simulation(
        digital_twin_id=digital_twin.id,
        start_time=datetime.utcnow(),
        scenario=scenario,
        status="running"
    )
    db.add(simulation)
    db.commit()
    
    try:
        # Run simulation logic here
        # This is a placeholder for actual simulation
        results = {"status": "success", "data": {"predicted_lifetime": 365}}
        
        # Update simulation record
        simulation.end_time = datetime.utcnow()
        simulation.results = results
        simulation.status = "completed"
        db.commit()
        
        return results
    except Exception as e:
        simulation.status = "failed"
        simulation.results = {"error": str(e)}
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

# Predictive Maintenance endpoints
@app.post("/predict/maintenance/{machine_id}")
async def predict_maintenance(
    machine_id: int,
    db: Session = Depends(get_db)
):
    """Predict maintenance needs for a specific machine."""
    # Get recent sensor readings
    readings = (
        db.query(SensorReading)
        .join(Sensor)
        .filter(Sensor.machine_id == machine_id)
        .filter(SensorReading.timestamp >= datetime.utcnow() - timedelta(hours=24))
        .all()
    )
    
    if not readings:
        raise HTTPException(status_code=404, detail="No recent sensor data found")
    
    try:
        # Prepare data for prediction
        data = {
            "timestamp": [r.timestamp for r in readings],
            "value": [r.value for r in readings],
            "sensor_id": [r.sensor_id for r in readings]
        }
        
        # Make prediction
        prediction = predictive_model.predict(data)
        
        # Store prediction
        ml_model = db.query(MLModel).filter(MLModel.type == "predictive_maintenance").first()
        pred_record = Prediction(
            model_id=ml_model.id,
            input_data=data,
            prediction=prediction.tolist(),
            confidence=0.95  # Example confidence score
        )
        db.add(pred_record)
        db.commit()
        
        return {
            "prediction": prediction.tolist(),
            "timestamp": datetime.utcnow(),
            "confidence": 0.95
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Anomaly Detection endpoints
@app.post("/detect/anomalies/{machine_id}")
async def detect_anomalies(
    machine_id: int,
    db: Session = Depends(get_db)
):
    """Detect anomalies for a specific machine."""
    # Get recent sensor readings
    readings = (
        db.query(SensorReading)
        .join(Sensor)
        .filter(Sensor.machine_id == machine_id)
        .filter(SensorReading.timestamp >= datetime.utcnow() - timedelta(hours=24))
        .all()
    )
    
    if not readings:
        raise HTTPException(status_code=404, detail="No recent sensor data found")
    
    try:
        # Prepare data for anomaly detection
        data = {
            "timestamp": [r.timestamp for r in readings],
            "value": [r.value for r in readings],
            "sensor_id": [r.sensor_id for r in readings]
        }
        
        # Detect anomalies
        results = anomaly_detector.predict(data)
        
        # Store detected anomalies
        for i, is_anomaly in enumerate(results['anomalies']):
            if is_anomaly:
                anomaly = Anomaly(
                    machine_id=machine_id,
                    detection_time=data['timestamp'][i],
                    anomaly_type="sensor_reading",
                    severity=results['reconstruction_errors'][i],
                    description="Anomalous sensor reading detected",
                    sensor_data={
                        "value": data['value'][i],
                        "sensor_id": data['sensor_id'][i]
                    }
                )
                db.add(anomaly)
        db.commit()
        
        return {
            "anomalies_detected": sum(results['anomalies']),
            "timestamp": datetime.utcnow(),
            "details": results
        }
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
