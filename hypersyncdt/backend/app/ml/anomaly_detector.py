import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoEncoder(nn.Module):
    """Neural autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class AnomalyDetector:
    """Advanced anomaly detection system combining multiple approaches."""
    
    def __init__(self, contamination: float = 0.1):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.autoencoder = None
        self.reconstruction_threshold = None
        self.feature_statistics = {}
    
    def _initialize_autoencoder(self, input_dim: int):
        """Initialize the autoencoder with the correct input dimension."""
        self.autoencoder = AutoEncoder(input_dim=input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters())
        self.criterion = nn.MSELoss()
    
    def _train_autoencoder(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the autoencoder on normal data."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                reconstructed, _ = self.autoencoder(batch[0])
                loss = self.criterion(reconstructed, batch[0])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
    
    def _compute_reconstruction_threshold(self, X: np.ndarray, percentile: float = 95):
        """Compute the reconstruction error threshold for anomaly detection."""
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed, _ = self.autoencoder(X_tensor)
            reconstruction_errors = torch.mean(torch.pow(X_tensor - reconstructed, 2), dim=1)
            self.reconstruction_threshold = np.percentile(
                reconstruction_errors.cpu().numpy(), 
                percentile
            )
    
    def fit(self, data: pd.DataFrame):
        """Train the anomaly detection model on normal data."""
        logger.info("Starting anomaly detector training...")
        
        # Prepare features
        numeric_features = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        self.isolation_forest.fit(X_scaled)
        
        # Initialize and train autoencoder
        self._initialize_autoencoder(X_scaled.shape[1])
        self._train_autoencoder(X_scaled)
        self._compute_reconstruction_threshold(X_scaled)
        
        # Compute feature statistics
        self.feature_statistics = {
            'mean': data[numeric_features].mean().to_dict(),
            'std': data[numeric_features].std().to_dict(),
            'q1': data[numeric_features].quantile(0.25).to_dict(),
            'q3': data[numeric_features].quantile(0.75).to_dict()
        }
        
        logger.info("Anomaly detector training completed successfully")
    
    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Detect anomalies using multiple methods."""
        numeric_features = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_features].values
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest predictions
        if_predictions = self.isolation_forest.predict(X_scaled)
        if_anomalies = if_predictions == -1
        
        # Autoencoder predictions
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructed, latent = self.autoencoder(X_tensor)
            reconstruction_errors = torch.mean(torch.pow(X_tensor - reconstructed, 2), dim=1)
            ae_anomalies = reconstruction_errors.cpu().numpy() > self.reconstruction_threshold
        
        # Statistical anomalies
        stat_anomalies = np.zeros(len(data), dtype=bool)
        for feature in numeric_features:
            mean, std = self.feature_statistics['mean'][feature], self.feature_statistics['std'][feature]
            z_scores = np.abs((data[feature] - mean) / std)
            stat_anomalies |= z_scores > 3  # Mark as anomaly if more than 3 standard deviations
        
        # Combine predictions
        combined_anomalies = (if_anomalies.astype(int) + 
                            ae_anomalies.astype(int) + 
                            stat_anomalies.astype(int)) >= 2  # Majority voting
        
        return {
            'anomalies': combined_anomalies,
            'isolation_forest_anomalies': if_anomalies,
            'autoencoder_anomalies': ae_anomalies,
            'statistical_anomalies': stat_anomalies,
            'reconstruction_errors': reconstruction_errors.cpu().numpy(),
            'latent_space': latent.cpu().numpy()
        }
    
    def save_model(self, path: str):
        """Save the trained model."""
        model_state = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'autoencoder_state': self.autoencoder.state_dict(),
            'reconstruction_threshold': self.reconstruction_threshold,
            'feature_statistics': self.feature_statistics
        }
        torch.save(model_state, path)
        logger.info(f"Model saved successfully to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        model_state = torch.load(path, map_location=self.device)
        self.isolation_forest = model_state['isolation_forest']
        self.scaler = model_state['scaler']
        self._initialize_autoencoder(len(self.feature_statistics['mean']))
        self.autoencoder.load_state_dict(model_state['autoencoder_state'])
        self.reconstruction_threshold = model_state['reconstruction_threshold']
        self.feature_statistics = model_state['feature_statistics']
        logger.info(f"Model loaded successfully from {path}")

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = pd.DataFrame({
        'temperature': np.random.normal(60, 5, n_samples),
        'vibration': np.random.normal(0.5, 0.1, n_samples),
        'pressure': np.random.normal(100, 10, n_samples),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    })
    
    # Add some anomalies
    anomaly_data = pd.DataFrame({
        'temperature': np.random.normal(85, 5, 50),
        'vibration': np.random.normal(1.2, 0.1, 50),
        'pressure': np.random.normal(150, 10, 50),
        'timestamp': pd.date_range(start='2024-02-01', periods=50, freq='H')
    })
    
    # Combine data
    data = pd.concat([normal_data, anomaly_data], ignore_index=True)
    
    # Initialize and train detector
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(normal_data)
    
    # Detect anomalies
    results = detector.predict(data)
    print("\nNumber of anomalies detected:", sum(results['anomalies']))
    print("Anomaly detection methods agreement rate:", 
          sum((results['isolation_forest_anomalies'] & 
               results['autoencoder_anomalies'] & 
               results['statistical_anomalies'])) / sum(results['anomalies'])) 