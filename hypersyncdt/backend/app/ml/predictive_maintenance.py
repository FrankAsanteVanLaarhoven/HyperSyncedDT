import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class ToolWearPredictor(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class PhysicsInformedNN(nn.Module):
    def __init__(self, input_size: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Archard's wear law implementation"""
        pressure = x[:, 0]
        velocity = x[:, 1]
        hardness = x[:, 2]
        
        # Physics-based wear rate
        theoretical_wear = pressure * velocity / hardness
        return nn.MSELoss()(y_pred, theoretical_wear)

class MaintenancePredictor:
    def __init__(self):
        self.tool_wear_model = ToolWearPredictor()
        self.physics_model = PhysicsInformedNN()
        self.scaler = StandardScaler()
        
    def predict_maintenance_needs(self, 
                                sensor_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Predict maintenance needs based on sensor data"""
        # Prepare input data
        X = self._prepare_input_data(sensor_data)
        
        # Make predictions
        with torch.no_grad():
            wear_pred = self.tool_wear_model(X)
            physics_pred = self.physics_model(X)
            
        # Combine predictions
        final_prediction = 0.7 * wear_pred + 0.3 * physics_pred
        
        return {
            'predicted_wear': float(final_prediction),
            'maintenance_due_in_hours': self._calculate_maintenance_window(final_prediction),
            'confidence_score': self._calculate_confidence(wear_pred, physics_pred)
        }
    
    def _prepare_input_data(self, sensor_data: Dict[str, np.ndarray]) -> torch.Tensor:
        """Prepare input data for model prediction"""
        features = np.column_stack([
            sensor_data['temperature'],
            sensor_data['vibration'],
            sensor_data['pressure'],
            sensor_data['velocity'],
            sensor_data['acoustic_emission']
        ])
        
        scaled_features = self.scaler.fit_transform(features)
        return torch.FloatTensor(scaled_features).unsqueeze(0)
    
    def _calculate_maintenance_window(self, wear_prediction: torch.Tensor) -> float:
        """Calculate time until maintenance is needed"""
        WEAR_THRESHOLD = 0.8
        current_wear = float(wear_prediction)
        wear_rate = current_wear / 100  # wear per hour
        
        return (WEAR_THRESHOLD - current_wear) / wear_rate
    
    def _calculate_confidence(self, 
                            wear_pred: torch.Tensor, 
                            physics_pred: torch.Tensor) -> float:
        """Calculate confidence score based on model agreement"""
        difference = abs(float(wear_pred - physics_pred))
        return max(0, 1 - difference/0.5)  # Scale confidence between 0 and 1