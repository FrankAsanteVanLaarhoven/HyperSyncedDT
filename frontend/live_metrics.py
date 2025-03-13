import streamlit as st
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import random
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional

class LiveMetricsManager:
    """Manages live updating metrics with realistic variations and trends."""
    
    def __init__(self):
        """Initialize the metrics manager with default values."""
        # Base metrics with initial values and variation parameters
        self.metrics = {
            "efficiency": {
                "value": 83.3,
                "min": 78.0,
                "max": 92.0,
                "volatility": 0.2,  # How much it can change per update
                "trend": 0.02,      # Small upward trend
                "format": "{:.1f}%"
            },
            "quality_score": {
                "value": 99.6,
                "min": 98.0,
                "max": 99.9,
                "volatility": 0.05,
                "trend": 0.01,
                "format": "{:.1f}"
            },
            "throughput": {
                "value": 739,
                "min": 680,
                "max": 820,
                "volatility": 3.0,
                "trend": 0.5,
                "format": "{} units"
            },
            "utilization": {
                "value": 80.7,
                "min": 75.0,
                "max": 90.0,
                "volatility": 0.3,
                "trend": 0.04,
                "format": "{:.1f}%"
            },
            "current": {
                "value": 46.2,
                "min": 40.0,
                "max": 55.0,
                "volatility": 0.4,
                "trend": 0.0,  # No trend
                "format": "{:.1f}%"
            },
            "avg_response_time": {
                "value": 118,
                "min": 90,
                "max": 150,
                "volatility": 1.0,
                "trend": -0.02,  # Slight improvement trend
                "format": "{} ms"
            },
            "daily_throughput": {
                "value": 18134,
                "min": 17000,
                "max": 20000,
                "volatility": 15.0,
                "trend": 2.0,
                "format": "{:,}"
            },
            "system_health": {
                "value": 91.0,
                "min": 85.0,
                "max": 98.0,
                "volatility": 0.2,
                "trend": 0.01,
                "format": "{}%"
            }
        }
        
        # History for charting
        self.history = {key: [] for key in self.metrics.keys()}
        self.timestamps = []
        
        # System uptime tracking
        self.start_time = datetime.now() - timedelta(days=18, hours=20)
        
        # Initialize with some historical data
        for i in range(100):
            self.update_metrics(save_history=True)
    
    def update_metrics(self, save_history: bool = False) -> None:
        """Update all metrics with realistic variations."""
        current_time = datetime.now()
        
        for key, metric in self.metrics.items():
            # Apply random variation based on volatility
            change = np.random.normal(0, metric["volatility"])
            
            # Apply trend
            change += metric["trend"]
            
            # Update value with constraints
            new_value = metric["value"] + change
            new_value = max(metric["min"], min(metric["max"], new_value))
            
            # Update the metric
            metric["value"] = new_value
            
            # Save to history if requested
            if save_history:
                self.history[key].append(new_value)
        
        # Save timestamp for history
        if save_history:
            self.timestamps.append(current_time)
            
            # Limit history size
            max_history = 1000
            if len(self.timestamps) > max_history:
                self.timestamps = self.timestamps[-max_history:]
                for key in self.history:
                    self.history[key] = self.history[key][-max_history:]
    
    def get_formatted_value(self, key: str) -> str:
        """Get a formatted string value for the specified metric."""
        if key not in self.metrics:
            return "N/A"
        
        metric = self.metrics[key]
        return metric["format"].format(int(metric["value"]) if metric["format"] == "{}" or metric["format"] == "{:,}" else metric["value"])
    
    def get_system_uptime(self) -> str:
        """Get the formatted system uptime."""
        uptime = datetime.now() - self.start_time
        days = uptime.days
        hours = uptime.seconds // 3600
        return f"{days}d {hours}h"
    
    def get_delta(self, key: str) -> float:
        """Calculate the delta/change for the specified metric."""
        if key not in self.metrics or len(self.history[key]) < 10:
            return 0.0
        
        # Calculate change over last 10 points
        recent = self.history[key][-10:]
        if len(recent) > 1:
            change = recent[-1] - recent[0]
            return change
        return 0.0
    
    def get_chart_data(self, key: str, points: int = 100) -> Dict[str, List]:
        """Get chart data for the specified metric."""
        if key not in self.history or not self.timestamps:
            return {"x": [], "y": []}
        
        # Get the most recent points
        data_points = min(points, len(self.timestamps))
        times = self.timestamps[-data_points:]
        values = self.history[key][-data_points:]
        
        return {"x": times, "y": values}
    
    def generate_spark_chart(self, key: str, width: int = 100, height: int = 30, color: str = "#00BFFF") -> go.Figure:
        """Generate a small spark line chart for the specified metric."""
        data = self.get_chart_data(key, 50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["x"], 
            y=data["y"],
            mode='lines',
            line=dict(color=color, width=1.5),
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
        ))
        
        # Remove all axes, grid, and legend
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False)
        )
        
        return fig

def render_live_metrics():
    """Render a dashboard with live, ticking metrics."""
    # Ensure we have the metrics manager in session state
    if "live_metrics_manager" not in st.session_state:
        st.session_state.live_metrics_manager = LiveMetricsManager()
    
    # Get manager from session state
    manager = st.session_state.live_metrics_manager
    
    # Update metrics on each rerun
    manager.update_metrics(save_history=True)
    
    # Create the dashboard layout
    st.markdown("""
    <style>
    .metric-container {
        background: rgba(25, 30, 50, 0.6);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(100, 120, 200, 0.2);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        backdrop-filter: blur(5px);
    }
    .metric-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        color: rgba(200, 220, 255, 0.9);
        border-bottom: 1px solid rgba(100, 120, 200, 0.2);
        padding-bottom: 8px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
    }
    .metric-value-large {
        font-size: 2.2rem;
        font-weight: 700;
        color: rgba(240, 240, 255, 0.95);
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: rgba(180, 190, 220, 0.8);
    }
    .metric-delta-positive {
        font-size: 0.9rem;
        color: rgba(100, 220, 120, 0.9);
        display: inline-block;
        margin-left: 5px;
    }
    .metric-delta-negative {
        font-size: 0.9rem;
        color: rgba(220, 100, 100, 0.9);
        display: inline-block;
        margin-left: 5px;
    }
    .system-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 15px;
    }
    .stat-item {
        flex: 1;
        min-width: 120px;
        padding: 10px;
        background: rgba(40, 45, 70, 0.7);
        border-radius: 8px;
        text-align: center;
    }
    .stat-icon {
        font-size: 1.2rem;
        margin-right: 5px;
        opacity: 0.9;
    }
    .sparkline-container {
        height: 30px;
        margin-top: 5px;
    }
    .realtime-label {
        background: rgba(100, 180, 220, 0.2);
        color: rgba(180, 220, 255, 0.9);
        font-size: 0.7rem;
        padding: 3px 8px;
        border-radius: 10px;
        display: inline-block;
        margin-left: 10px;
        border: 1px solid rgba(100, 180, 220, 0.3);
    }
    .monitoring-container {
        background: rgba(30, 35, 55, 0.6);
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        border: 1px solid rgba(100, 120, 200, 0.2);
    }
    .monitor-values {
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
        font-size: 0.85rem;
    }
    .monitor-title {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: rgba(80, 220, 100, 0.8);
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main metrics section
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="metric-title">Advanced Metrics <span class="realtime-label">LIVE</span></h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="metric-value-large">{manager.get_formatted_value("efficiency")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Efficiency</div>', unsafe_allow_html=True)
        
        # Add small sparkline chart
        st.plotly_chart(manager.generate_spark_chart("efficiency", color="#4CAF50"), use_container_width=True)
        
        st.markdown(f'<div class="metric-value-large">{manager.get_formatted_value("throughput")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Throughput</div>', unsafe_allow_html=True)
        
        # Add small sparkline chart
        st.plotly_chart(manager.generate_spark_chart("throughput", color="#FF9800"), use_container_width=True)
        
    with col2:
        st.markdown(f'<div class="metric-value-large">{manager.get_formatted_value("quality_score")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Quality Score</div>', unsafe_allow_html=True)
        
        # Add small sparkline chart
        st.plotly_chart(manager.generate_spark_chart("quality_score", color="#2196F3"), use_container_width=True)
        
        st.markdown(f'<div class="metric-value-large">{manager.get_formatted_value("utilization")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Utilization</div>', unsafe_allow_html=True)
        
        # Add small sparkline chart
        st.plotly_chart(manager.generate_spark_chart("utilization", color="#9C27B0"), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time monitoring section
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="monitor-title"><div class="live-indicator"></div>Real-Time Monitoring</div>', unsafe_allow_html=True)
    
    # Current value
    current_value = manager.get_formatted_value("current")
    current_delta = manager.get_delta("current")
    delta_class = "metric-delta-positive" if current_delta >= 0 else "metric-delta-negative"
    delta_symbol = "+" if current_delta >= 0 else ""
    
    st.markdown(f'<div class="metric-value-large">{current_value} <span class="{delta_class}">{delta_symbol}{current_delta:.1f}%</span></div>', unsafe_allow_html=True)
    
    # Real-time chart for current value
    current_data = manager.get_chart_data("current", 100)
    
    # Create real-time chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=current_data["x"], 
        y=current_data["y"],
        mode='lines',
        line=dict(color='rgba(100, 220, 255, 0.8)', width=2),
        fill='tozeroy',
        fillcolor='rgba(100, 220, 255, 0.1)'
    ))
    
    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        height=180,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(100, 120, 200, 0.1)',
            showline=False,
            tickvals=[manager.metrics["current"]["min"], 
                     (manager.metrics["current"]["min"] + manager.metrics["current"]["max"])/2, 
                     manager.metrics["current"]["max"]],
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show min, avg, max values
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="stat-item"><div class="metric-label">Average</div><div class="metric-value-large" style="font-size:1.5rem;">44.9%</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="stat-item"><div class="metric-label">Minimum</div><div class="metric-value-large" style="font-size:1.5rem;">35.2%</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="stat-item"><div class="metric-label">Maximum</div><div class="metric-value-large" style="font-size:1.5rem;">53.5%</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key Performance Indicators
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="metric-title">Key Performance Indicators</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="stat-item"><div class="stat-icon">⏱️</div> <div class="metric-value-large" style="font-size:1.2rem;">{manager.get_system_uptime()}</div><div class="metric-label">System Uptime</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="stat-item"><div class="stat-icon">⚡</div> <div class="metric-value-large" style="font-size:1.2rem;">{manager.get_formatted_value("avg_response_time")}</div><div class="metric-label">Avg Response Time</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="stat-item"><div class="stat-icon">🔄</div> <div class="metric-value-large" style="font-size:1.2rem;">{manager.get_formatted_value("daily_throughput")}</div><div class="metric-label">Daily Throughput</div></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="stat-item"><div class="stat-icon">❤️</div> <div class="metric-value-large" style="font-size:1.2rem;">{manager.get_formatted_value("system_health")}</div><div class="metric-label">System Health</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Set up auto-refresh using JavaScript
    st.markdown("""
    <script>
        // Auto-refresh the page every 3 seconds
        setTimeout(function(){
            window.location.reload();
        }, 3000);
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Live Metrics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    render_live_metrics() 