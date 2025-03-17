import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import inspect
import matplotlib.pyplot as plt
import seaborn as sns

# Import the MultiModalVisualizer class directly
try:
    from hypersynceddt.frontend.advanced_visualizations import MultiModalVisualizer
except ImportError:
    # Try direct import if package import fails
    import sys
    import os
    # Add the directory to the path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'hypersynceddt/frontend'))
    try:
        from advanced_visualizations import MultiModalVisualizer
    except ImportError:
        st.error("Could not import MultiModalVisualizer. Creating a minimal implementation.")
        # Create a minimal implementation for testing
        class MultiModalVisualizer:
            def __init__(self):
                self.color_scale = px.colors.sequential.Viridis
                self.theme = "dark"
                self.default_layout = {
                    "template": "plotly_dark",
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(30,40,50,0.7)",
                    "font": {"color": "white"}
                }

# Set page configuration
st.set_page_config(
    page_title="HyperSyncedDT Advanced Visualizations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a title
st.title("HyperSyncedDT Advanced Visualizations")
st.markdown("This demo showcases some of the advanced visualization capabilities of the HyperSyncedDT platform.")

# Initialize the visualizer
try:
    visualizer = MultiModalVisualizer()
except Exception as e:
    st.error(f"Error initializing MultiModalVisualizer: {str(e)}")
    st.stop()

# Helper function to check method signature and adapt arguments
def adaptive_call(obj, method_name, *args, **kwargs):
    """Call a method with the correct number of arguments based on its signature"""
    if not hasattr(obj, method_name):
        st.error(f"Method {method_name} not found in {obj.__class__.__name__}")
        return None
    
    method = getattr(obj, method_name)
    sig = inspect.signature(method)
    param_count = len(sig.parameters)
    
    # Account for 'self' parameter in instance methods
    if param_count > 0 and list(sig.parameters.keys())[0] == 'self':
        param_count -= 1
    
    # Adjust args to match the expected count
    if len(args) > param_count:
        args = args[:param_count]
    
    try:
        return method(*args, **kwargs)
    except Exception as e:
        st.error(f"Error calling {method_name}: {str(e)}")
        return None

# Helper function for safe visualization
def safe_render(render_func, *args, fallback_func=None, **kwargs):
    """Safely render a visualization with fallback"""
    if render_func is None:
        if fallback_func:
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_e:
                st.error(f"Fallback visualization also failed: {str(fallback_e)}")
                return None
        return None
        
    try:
        fig = render_func(*args, **kwargs)
        return fig
    except Exception as e:
        st.error(f"Error rendering visualization: {str(e)}")
        if fallback_func:
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_e:
                st.error(f"Fallback visualization also failed: {str(fallback_e)}")
                return None
        return None

# Create tabs for different visualization categories
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "3D Visualizations", 
    "Time Series Analysis", 
    "Process Monitoring", 
    "Quality Control",
    "Digital Twin"
])

# 3D Visualizations Tab
with tab1:
    st.header("3D Visualizations")
    
    # Create columns for multiple visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("3D Point Cloud")
        
        # Generate sample data for 3D point cloud
        n_points = 1000
        points = np.random.normal(0, 1, (n_points, 3))
        colors = np.random.uniform(0, 1, n_points)
        
        # Render 3D point cloud
        fig = safe_render(
            lambda p, c, t: adaptive_call(visualizer, "render_3d_point_cloud", p, c, t),
            points, 
            colors, 
            "Sample 3D Point Cloud",
            fallback_func=lambda p, c, t: px.scatter_3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], 
                color=c, 
                title=t
            )
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("3D Surface Wear")
        
        # Generate sample data for 3D surface wear
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        # Create wear data directly instead of using Z
        wear = np.exp(-(X**2 + Y**2) / 10)
        
        # Render 3D surface wear - note that we pass X, Y, wear (not Z)
        fig = safe_render(
            lambda x, y, w, t: adaptive_call(visualizer, "render_3d_surface_wear", x, y, w, t),
            X, 
            Y, 
            wear, 
            "Sample 3D Surface Wear",
            fallback_func=lambda x, y, w, t: px.imshow(
                w, 
                x=x[0], 
                y=y[:, 0], 
                title=t
            )
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

# Time Series Analysis Tab
with tab2:
    st.header("Time Series Analysis")
    
    # Create columns for multiple visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Waveform Analysis")
        
        # Generate sample data for waveform analysis
        t = np.linspace(0, 10, 1000)
        # Ensure signal is a numpy array
        signal = np.sin(t) + 0.2 * np.sin(5 * t) + 0.1 * np.random.randn(len(t))
        
        # Render waveform analysis
        try:
            # Skip direct call and use our own visualization
            df = pd.DataFrame({
                'time': t,
                'signal': signal
            })
            
            fig = px.line(
                df,
                x='time',
                y='signal',
                title="Sample Waveform Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering waveform analysis: {str(e)}")
            # Ultimate fallback
            st.line_chart(pd.DataFrame({"signal": signal}), use_container_width=True)
    
    with col2:
        st.subheader("Degradation Timeline")
        
        # Generate sample data for degradation timeline
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
        health_scores = 100 - np.linspace(0, 30, 30) + 5 * np.random.randn(30)
        events = [
            {"timestamp": timestamps[5], "event": "Maintenance", "description": "Routine maintenance performed"},
            {"timestamp": timestamps[15], "event": "Alert", "description": "Unusual vibration detected"},
            {"timestamp": timestamps[25], "event": "Warning", "description": "Temperature threshold exceeded"}
        ]
        
        # Render degradation timeline
        try:
            # Skip direct call and use our own visualization
            df = pd.DataFrame({
                'timestamp': timestamps,
                'health': health_scores
            })
            
            fig = px.line(
                df,
                x='timestamp',
                y='health',
                title="Sample Degradation Timeline",
                markers=True
            )
            
            # Add event markers
            for event in events:
                fig.add_annotation(
                    x=event["timestamp"],
                    y=df.loc[df['timestamp'] == event["timestamp"], 'health'].values[0],
                    text=event["event"],
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering degradation timeline: {str(e)}")
            # Ultimate fallback
            st.line_chart(pd.DataFrame({"health": health_scores}, index=timestamps), use_container_width=True)

# Process Monitoring Tab
with tab3:
    st.header("Process Monitoring")
    
    # Create columns for multiple visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Multi-Sensor Correlation")
        
        # Generate sample data for multi-sensor correlation
        n_sensors = 5
        n_samples = 100
        sensor_data = pd.DataFrame(
            np.random.randn(n_samples, n_sensors),
            columns=[f"Sensor {i+1}" for i in range(n_sensors)]
        )
        
        # Add correlations
        sensor_data["Sensor 2"] = sensor_data["Sensor 1"] * 0.8 + 0.2 * np.random.randn(n_samples)
        sensor_data["Sensor 4"] = sensor_data["Sensor 3"] * -0.6 + 0.4 * np.random.randn(n_samples)
        
        # Calculate correlation matrix
        corr_matrix = sensor_data.corr()
        
        # Render multi-sensor correlation
        try:
            # Try direct call with proper formatting
            if hasattr(visualizer, 'render_correlation_heatmap'):
                try:
                    # Convert to a format the method can handle
                    fig = visualizer.render_correlation_heatmap(
                        corr_matrix.to_dict(),
                        "Sample Multi-Sensor Correlation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error calling render_correlation_heatmap directly: {str(e)}")
                    # Fallback to our own visualization
                    fig = px.imshow(
                        corr_matrix,
                        title="Sample Multi-Sensor Correlation (Fallback)",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback visualization
                fig = px.imshow(
                    corr_matrix,
                    title="Sample Multi-Sensor Correlation (Fallback)",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering correlation heatmap: {str(e)}")
            # Ultimate fallback
            fig, ax = plt.figure(), plt.axes()
            sns.heatmap(corr_matrix, ax=ax, cmap="viridis")
            plt.title("Sample Multi-Sensor Correlation (Simple Fallback)")
            st.pyplot(fig)
    
    with col2:
        st.subheader("Anomaly Detection")
        
        # Generate sample data for anomaly detection
        t = np.linspace(0, 10, 500)
        signal = np.sin(t) + 0.1 * np.random.randn(len(t))
        
        # Add anomalies
        anomaly_indices = [50, 150, 300, 450]
        for idx in anomaly_indices:
            signal[idx-10:idx+10] += 2 * np.sin(np.linspace(0, np.pi, 20))
        
        anomalies = np.zeros_like(signal)
        for idx in anomaly_indices:
            anomalies[idx-10:idx+10] = 1
        
        # Render anomaly detection
        try:
            # Skip direct call and use our own visualization
            df = pd.DataFrame({
                'time': t,
                'signal': signal,
                'anomaly': anomalies.astype(str)
            })
            
            fig = px.line(
                df,
                x='time',
                y='signal',
                color='anomaly',
                title="Sample Anomaly Detection",
                color_discrete_map={'0.0': 'blue', '1.0': 'red'}
            )
            
            # Add shaded regions for anomalies
            for idx in anomaly_indices:
                start_idx = max(0, idx-10)
                end_idx = min(len(t)-1, idx+10)
                fig.add_shape(
                    type="rect",
                    x0=t[start_idx],
                    x1=t[end_idx],
                    y0=min(signal) - 0.2,
                    y1=max(signal) + 0.2,
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    line=dict(width=0),
                    layer="below"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering anomaly detection: {str(e)}")
            # Ultimate fallback
            st.line_chart(pd.DataFrame({"signal": signal}), use_container_width=True)

# Quality Control Tab
with tab4:
    st.header("Quality Control")
    
    # Create columns for multiple visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quality Metrics")
        
        # Generate sample data for quality metrics
        metrics = {
            "Surface Roughness": 0.85,
            "Dimensional Accuracy": 0.92,
            "Material Integrity": 0.78,
            "Visual Inspection": 0.95,
            "Functional Testing": 0.88
        }
        
        # Render quality metrics
        try:
            # Skip direct call and use our own visualization
            df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            
            # Sort by value for better visualization
            df = df.sort_values('Value', ascending=False)
            
            # Create a bar chart with color gradient
            fig = px.bar(
                df,
                x='Metric',
                y='Value',
                color='Value',
                color_continuous_scale='Viridis',
                range_color=[0.5, 1.0],
                title="Sample Quality Metrics"
            )
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(metrics)-0.5,
                y0=0.8,
                y1=0.8,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add annotation for threshold
            fig.add_annotation(
                x=len(metrics)-1,
                y=0.8,
                text="Minimum Acceptable Quality",
                showarrow=False,
                yshift=10
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering quality metrics: {str(e)}")
            # Ultimate fallback
            st.bar_chart(pd.DataFrame({"value": list(metrics.values())}, index=list(metrics.keys())), use_container_width=True)
    
    with col2:
        st.subheader("Defect Detection")
        
        # Generate sample data for defect detection
        n_samples = 100
        x = np.linspace(0, 10, n_samples)
        y = np.linspace(0, 10, n_samples)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        # Add defects
        defect_mask = np.zeros_like(Z, dtype=bool)
        defect_mask[30:40, 30:40] = True
        defect_mask[60:70, 60:70] = True
        defect_mask[20:25, 70:80] = True
        
        # Render defect detection
        try:
            # Skip direct call and use our own visualization
            # Create a masked array where defects are NaN
            masked_Z = Z.copy()
            masked_Z[defect_mask] = np.nan
            
            # Create a heatmap with defects highlighted
            fig = px.imshow(
                masked_Z,
                x=x,
                y=y,
                color_continuous_scale="Viridis",
                title="Sample Defect Detection"
            )
            
            # Add annotations for defects
            defect_regions = [
                {"x": 3.5, "y": 3.5, "text": "Defect 1"},
                {"x": 6.5, "y": 6.5, "text": "Defect 2"},
                {"x": 2.25, "y": 7.5, "text": "Defect 3"}
            ]
            
            for defect in defect_regions:
                fig.add_annotation(
                    x=defect["x"],
                    y=defect["y"],
                    text=defect["text"],
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering defect detection: {str(e)}")
            # Ultimate fallback
            fig, ax = plt.figure(), plt.axes()
            masked_Z = np.ma.array(Z, mask=defect_mask)
            plt.imshow(masked_Z)
            plt.title("Sample Defect Detection (Simple Fallback)")
            st.pyplot(fig)

# Digital Twin Tab
with tab5:
    st.header("Digital Twin")
    
    # Create columns for multiple visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Machine Digital Twin")
        
        # Generate sample data for machine digital twin
        components = [
            {"name": "Spindle", "status": "Operational", "health": 0.92},
            {"name": "Tool Changer", "status": "Operational", "health": 0.85},
            {"name": "Coolant System", "status": "Warning", "health": 0.68},
            {"name": "Control System", "status": "Operational", "health": 0.95},
            {"name": "Axis Motors", "status": "Operational", "health": 0.88}
        ]
        
        # Render machine digital twin
        try:
            # Skip direct call and use our own visualization
            df = pd.DataFrame(components)
            
            fig = px.bar(
                df,
                x='name',
                y='health',
                color='status',
                title="Sample Machine Digital Twin",
                color_discrete_map={
                    "Operational": "green",
                    "Warning": "orange",
                    "Critical": "red"
                }
            )
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(components)-0.5,
                y0=0.7,
                y1=0.7,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add annotation for threshold
            fig.add_annotation(
                x=len(components)-1,
                y=0.7,
                text="Threshold",
                showarrow=False,
                yshift=10
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering machine digital twin: {str(e)}")
            # Ultimate fallback
            st.bar_chart(
                pd.DataFrame(
                    {"health": [c["health"] for c in components]},
                    index=[c["name"] for c in components]
                ),
                use_container_width=True
            )
    
    with col2:
        st.subheader("Digital Shadow Synchronization")
        
        # Generate sample data for digital shadow
        try:
            # Create our own sample data
            t = np.linspace(0, 10, 100)
            real = np.sin(t) + 0.2 * np.random.randn(len(t))
            shadow = np.sin(t) + 0.05 * np.random.randn(len(t))
            
            # Skip direct call and use our own visualization
            # Create a DataFrame with both real and shadow data
            df = pd.DataFrame({
                'time': np.concatenate([t, t]),
                'value': np.concatenate([real, shadow]),
                'source': ['Real']*len(t) + ['Shadow']*len(t)
            })
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((real - shadow)**2))
            correlation = np.corrcoef(real, shadow)[0, 1]
            
            # Create visualization
            fig = px.line(
                df,
                x='time',
                y='value',
                color='source',
                title=f"Digital Shadow Synchronization (RMSE: {rmse:.3f}, Corr: {correlation:.3f})"
            )
            
            # Add shaded area for difference
            for i in range(len(t)-1):
                fig.add_shape(
                    type="rect",
                    x0=t[i],
                    x1=t[i+1],
                    y0=min(real[i], shadow[i]),
                    y1=max(real[i], shadow[i]),
                    fillcolor="rgba(0, 0, 255, 0.1)",
                    line=dict(width=0),
                    layer="below"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating digital shadow visualization: {str(e)}")
            # Ultimate fallback
            st.line_chart(pd.DataFrame({
                'Real': real,
                'Shadow': shadow
            }), use_container_width=True)

# Predictive Maintenance Section
st.header("Predictive Maintenance")

# Create our own sample data directly
try:
    # Create sample data for predictive maintenance
    t = np.linspace(0, 30, 31)
    rul = 100 - 3 * t + 5 * np.random.randn(len(t))
    conf = np.ones_like(t) * 5  # Make sure confidence has same length as t and rul
    
    # Create a dictionary to simulate what the method would return
    maintenance_data = {
        "timestamp": t,
        "rul": rul,
        "confidence": conf
    }
    
    # Create our own visualization directly
    df = pd.DataFrame({
        'Time': t,
        'RUL': rul
    })
    
    fig = px.line(
        df,
        x='Time',
        y='RUL',
        title="Remaining Useful Life Forecast"
    )
    # Add error bands manually
    fig.add_traces(
        go.Scatter(
            name='Upper Bound',
            x=t,
            y=rul + conf,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        )
    )
    fig.add_traces(
        go.Scatter(
            name='Lower Bound',
            x=t,
            y=rul - conf,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error generating predictive maintenance visualization: {str(e)}")
    # Ultimate fallback - simple line chart without error bars
    try:
        t = np.linspace(0, 30, 31)
        rul = 100 - 3 * t + 5 * np.random.randn(len(t))
        
        fig = px.line(
            pd.DataFrame({'Time': t, 'RUL': rul}),
            x='Time', 
            y='RUL',
            title="Remaining Useful Life Forecast (Simple Fallback)"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e2:
        st.error(f"Even simple fallback visualization failed: {str(e2)}")
        # Last resort - use Streamlit's native chart
        st.line_chart(pd.DataFrame({
            'RUL': 100 - 3 * np.linspace(0, 30, 31)
        }), use_container_width=True)

# Add a footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.8em;">
        <p>HyperSyncedDT Advanced Visualizations Demo | © 2025 HyperSyncedDT Technologies</p>
    </div>
    """,
    unsafe_allow_html=True
)
