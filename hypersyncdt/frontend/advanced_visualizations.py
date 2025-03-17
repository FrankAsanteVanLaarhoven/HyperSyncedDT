# Import this module directly rather than from hypersyncdt.frontend.components
# This is a simplified version for direct import in app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Optional

class MultiModalVisualizer:
    """Class for creating multi-modal visualizations for the digital twin"""
    
    def __init__(self):
        """Initialize the visualizer with default settings"""
        self.theme = "dark"
        self.animation_frame = 0
        self.last_update = datetime.now()
        
    def create_multi_modal_chart(self, 
                             data: pd.DataFrame, 
                             x_column: str,
                             metrics: List[str],
                             title: str = "Multi-Modal Analysis",
                             height: int = 500) -> go.Figure:
        """
        Create a multi-modal chart combining multiple visualization types
        
        Args:
            data: DataFrame containing the data
            x_column: Column to use for x-axis
            metrics: List of metrics to visualize
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        # Create subplots with 2 rows and shared x-axis
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f"{metrics[0]} vs {metrics[1]}", 
                f"{metrics[2]} Analysis"
            )
        )
        
        # Add scatter plot to first subplot
        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=data[metrics[0]],
                mode='lines+markers',
                name=metrics[0],
                line=dict(color='#00CED1', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add bar chart on same subplot
        fig.add_trace(
            go.Bar(
                x=data[x_column],
                y=data[metrics[1]],
                name=metrics[1],
                marker_color='rgba(255, 87, 51, 0.7)'
            ),
            row=1, col=1
        )
        
        # Add area chart to second subplot
        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=data[metrics[2]],
                mode='lines',
                fill='tozeroy',
                name=metrics[2],
                line=dict(color='#9370DB', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig
    
    def generate_sample_data(self, n_points: int = 50) -> pd.DataFrame:
        """
        Generate sample data for visualization
        
        Args:
            n_points: Number of data points to generate
            
        Returns:
            DataFrame with sample data
        """
        now = datetime.now()
        timestamps = [now - timedelta(minutes=i) for i in range(n_points)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'temperature': 70 + np.sin(np.linspace(0, 6*np.pi, n_points)) * 10 + np.random.normal(0, 1, n_points),
            'pressure': 100 + np.cos(np.linspace(0, 4*np.pi, n_points)) * 15 + np.random.normal(0, 2, n_points),
            'flow_rate': 50 + np.sin(np.linspace(0, 8*np.pi, n_points)) * 8 + np.random.normal(0, 1, n_points),
            'vibration': 0.5 + 0.2 * np.sin(np.linspace(0, 10*np.pi, n_points)) + np.random.normal(0, 0.05, n_points),
            'power': 120 + np.sin(np.linspace(0, 3*np.pi, n_points)) * 20 + np.random.normal(0, 3, n_points),
            'efficiency': 85 + np.cos(np.linspace(0, 5*np.pi, n_points)) * 8 + np.random.normal(0, 1, n_points),
            'quality': 95 + np.sin(np.linspace(0, 7*np.pi, n_points)) * 3 + np.random.normal(0, 0.5, n_points)
        }) 