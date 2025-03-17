import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .advanced_visualizations import MultiModalVisualizer

def render_advanced_visualization_page():
    """Render the advanced visualization page with interactive components."""
    st.title("Advanced Visualizations")
    
    # Initialize visualizer
    visualizer = MultiModalVisualizer()
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Process Uncertainty",
        "3D Analysis",
        "Process Capability",
        "Correlation Analysis"
    ])
    
    with tab1:
        st.subheader("Process Uncertainty Analysis")
        
        # Parameters for uncertainty visualization
        n_points = st.slider("Number of data points", 10, 100, 50)
        noise_level = st.slider("Noise level", 0.1, 2.0, 0.5)
        
        # Generate sample data
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_points)]
        mean = np.sin(np.linspace(0, 4*np.pi, n_points)) * 10 + 100
        std = np.ones_like(mean) * noise_level
        
        # Create and display visualization
        fig = visualizer.render_uncertainty_visualization(
            timestamps,
            mean.tolist(),
            std.tolist(),
            "Process Uncertainty Visualization"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("3D Process Analysis")
        
        # Parameters for 3D visualization
        n_points_3d = st.slider("Number of 3D points", 100, 1000, 500)
        
        # Generate sample 3D data with meaningful patterns
        t = np.linspace(0, 4*np.pi, n_points_3d)
        points = np.column_stack([
            np.sin(t) * np.cos(t),  # X coordinate
            np.sin(t) * np.sin(t),  # Y coordinate
            np.cos(t)               # Z coordinate
        ])
        
        # Generate color values based on the distance from origin
        colors = np.linalg.norm(points, axis=1)  # This will be a 1D array
        
        # Create and display 3D visualization
        fig = visualizer.render_3d_point_cloud(
            points,
            colors,
            "3D Process Parameter Space"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Process Capability Analysis")
        
        # Parameters for process capability
        n_samples = st.slider("Number of samples", 100, 1000, 500)
        target = st.number_input("Target value", 95.0, 105.0, 100.0)
        lsl = st.number_input("Lower Specification Limit", 90.0, 99.0, 94.0)
        usl = st.number_input("Upper Specification Limit", 101.0, 110.0, 106.0)
        
        # Generate sample process data
        data = np.random.normal(target, 2, n_samples)
        
        # Create and display process capability chart
        fig = visualizer.render_process_capability(
            data.tolist(),
            lsl,
            usl,
            target,
            "Process Capability Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Correlation Analysis")
        
        # Generate sample process data with correlations
        n_samples = 100
        temperature = np.random.normal(100, 5, n_samples)
        pressure = temperature * 0.5 + np.random.normal(50, 2, n_samples)
        flow_rate = pressure * 0.3 + np.random.normal(75, 3, n_samples)
        quality = -0.2 * temperature + 0.4 * pressure + 0.3 * flow_rate + np.random.normal(95, 1, n_samples)
        
        process_data = {
            "Temperature": temperature,
            "Pressure": pressure,
            "Flow Rate": flow_rate,
            "Quality": quality
        }
        
        # Create and display correlation heatmap
        fig = visualizer.render_correlation_heatmap(
            process_data,
            "Process Parameter Correlations"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display real-time dashboard
        st.subheader("Real-time Process Monitoring")
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]
        fig = visualizer.render_real_time_dashboard(
            timestamps,
            process_data,
            "Real-time Process Parameters"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_advanced_visualization_page() 