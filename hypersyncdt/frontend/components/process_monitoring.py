import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ProcessMonitor:
    def __init__(self):
        self.metrics = {}
        self.history = pd.DataFrame()
        self.alerts = []
        
    def update_metrics(self, new_metrics: dict):
        self.metrics.update(new_metrics)
        self._update_history()
        
    def _update_history(self):
        current_time = datetime.now()
        new_row = pd.DataFrame([{
            'timestamp': current_time,
            **self.metrics
        }])
        self.history = pd.concat([self.history, new_row], ignore_index=True)
        
    def get_current_metrics(self):
        return self.metrics
        
    def get_history(self, timeframe: str = '1h'):
        if self.history.empty:
            return pd.DataFrame()
            
        now = datetime.now()
        if timeframe == '1h':
            cutoff = now - timedelta(hours=1)
        elif timeframe == '24h':
            cutoff = now - timedelta(days=1)
        elif timeframe == '7d':
            cutoff = now - timedelta(days=7)
        else:
            return self.history
            
        return self.history[self.history['timestamp'] >= cutoff]
        
    def add_alert(self, message: str, severity: str = 'info'):
        self.alerts.append({
            'timestamp': datetime.now(),
            'message': message,
            'severity': severity
        })
        
    def get_alerts(self, max_alerts: int = 10):
        return sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:max_alerts] 