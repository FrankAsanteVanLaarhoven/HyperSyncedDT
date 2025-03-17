import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
import io
import sys
from contextlib import contextmanager
import traceback
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

class EDAWorkspace:
    def __init__(self):
        self.digital_twin = SynchronizedDigitalTwin()
        self.visualizer = MultiModalVisualizer()
        self.setup_workspace()
        
    def setup_workspace(self):
        """Initialize workspace components and state."""
        if 'code_editor_content' not in st.session_state:
            st.session_state.code_editor_content = ""
        if 'execution_output' not in st.session_state:
            st.session_state.execution_output = ""
        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}
        if 'plot_history' not in st.session_state:
            st.session_state.plot_history = []
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
    
    @contextmanager
    def capture_output(self):
        """Capture stdout and stderr."""
        new_out, new_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    
    def execute_code(self, code: str) -> str:
        """Execute code and return output."""
        with self.capture_output() as (out, err):
            try:
                # Create a local namespace with common data science libraries
                namespace = {
                    'pd': pd,
                    'np': np,
                    'plt': plt,
                    'sns': sns,
                    'go': go,
                    'px': px,
                    'make_subplots': make_subplots,
                    'stats': stats,
                    'StandardScaler': StandardScaler,
                    'PCA': PCA,
                    'KMeans': KMeans,
                    'torch': torch,
                    'nn': nn,
                    'st': st
                }
                
                # Add data from cache to namespace
                namespace.update(st.session_state.data_cache)
                
                # Execute code
                exec(code, namespace)
                
                # Capture any generated figures
                if 'fig' in namespace:
                    st.session_state.plot_history.append(namespace['fig'])
                
                return out.getvalue() + err.getvalue()
            except Exception as e:
                return f"Error: {str(e)}\n{traceback.format_exc()}"
    
    def render_code_editor(self):
        """Render the code editor interface."""
        st.markdown("### 📝 Code Editor")
        
        # Code editor with syntax highlighting
        code = st.text_area(
            "Python Code",
            value=st.session_state.code_editor_content,
            height=300,
            key="code_editor"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Run Code", type="primary"):
                output = self.execute_code(code)
                st.session_state.execution_output = output
                st.session_state.code_editor_content = code
        
        with col2:
            if st.button("Clear Output"):
                st.session_state.execution_output = ""
        
        with col3:
            if st.button("Save Script"):
                self.save_script(code)
        
        # Display execution output
        if st.session_state.execution_output:
            st.code(st.session_state.execution_output, language="python")
    
    def render_data_viewer(self):
        """Render the data viewer interface."""
        st.markdown("### 📊 Data Viewer")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Data File", type=['csv', 'xlsx', 'json'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                else:
                    data = pd.read_json(uploaded_file)
                
                # Store data in cache
                st.session_state.data_cache[uploaded_file.name] = data
                
                # Display data preview
                st.markdown(f"#### Preview of {uploaded_file.name}")
                st.dataframe(data.head())
                
                # Display data info
                st.markdown("#### Data Info")
                buffer = io.StringIO()
                data.info(buf=buffer)
                st.text(buffer.getvalue())
                
                # Display basic statistics
                st.markdown("#### Basic Statistics")
                st.dataframe(data.describe())
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    def render_visualization_tools(self):
        """Render the visualization tools interface."""
        st.markdown("### 📈 Visualization Tools")
        
        if not st.session_state.data_cache:
            st.warning("Please upload data to use visualization tools.")
            return
        
        # Select dataset
        dataset_name = st.selectbox("Select Dataset", list(st.session_state.data_cache.keys()))
        data = st.session_state.data_cache[dataset_name]
        
        # Visualization type selector
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Line Plot", "Scatter Plot", "Bar Plot", "Box Plot", "Histogram", 
             "Heatmap", "3D Scatter", "Parallel Coordinates"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Common parameters
            x_col = st.selectbox("X-axis", data.columns)
            y_col = st.selectbox("Y-axis", data.columns)
            
            if viz_type in ["Scatter Plot", "3D Scatter"]:
                color_col = st.selectbox("Color by", ["None"] + list(data.columns))
                if viz_type == "3D Scatter":
                    z_col = st.selectbox("Z-axis", data.columns)
        
        with col2:
            # Additional parameters
            if viz_type == "Histogram":
                bins = st.slider("Number of bins", 5, 100, 30)
            elif viz_type == "Box Plot":
                group_col = st.selectbox("Group by", ["None"] + list(data.columns))
            elif viz_type == "Heatmap":
                corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
        
        # Create visualization
        fig = None
        
        if viz_type == "Line Plot":
            fig = px.line(data, x=x_col, y=y_col)
        elif viz_type == "Scatter Plot":
            if color_col != "None":
                fig = px.scatter(data, x=x_col, y=y_col, color=color_col)
            else:
                fig = px.scatter(data, x=x_col, y=y_col)
        elif viz_type == "Bar Plot":
            fig = px.bar(data, x=x_col, y=y_col)
        elif viz_type == "Box Plot":
            if group_col != "None":
                fig = px.box(data, x=group_col, y=y_col)
            else:
                fig = px.box(data, y=y_col)
        elif viz_type == "Histogram":
            fig = px.histogram(data, x=x_col, nbins=bins)
        elif viz_type == "Heatmap":
            corr = data.corr(method=corr_method)
            fig = px.imshow(corr, color_continuous_scale="RdBu_r")
        elif viz_type == "3D Scatter":
            if color_col != "None":
                fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, color=color_col)
            else:
                fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col)
        elif viz_type == "Parallel Coordinates":
            fig = px.parallel_coordinates(data)
        
        if fig:
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.plot_history.append(fig)
    
    def render_analysis_tools(self):
        """Render the analysis tools interface."""
        st.markdown("### 🔍 Analysis Tools")
        
        if not st.session_state.data_cache:
            st.warning("Please upload data to use analysis tools.")
            return
        
        # Select dataset
        dataset_name = st.selectbox(
            "Select Dataset",
            list(st.session_state.data_cache.keys()),
            key="analysis_dataset"
        )
        data = st.session_state.data_cache[dataset_name]
        
        # Analysis type selector
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Statistical Tests", "Dimensionality Reduction", "Clustering",
             "Time Series Analysis", "Feature Analysis"]
        )
        
        if analysis_type == "Statistical Tests":
            self.render_statistical_tests(data)
        elif analysis_type == "Dimensionality Reduction":
            self.render_dimensionality_reduction(data)
        elif analysis_type == "Clustering":
            self.render_clustering(data)
        elif analysis_type == "Time Series Analysis":
            self.render_time_series_analysis(data)
        elif analysis_type == "Feature Analysis":
            self.render_feature_analysis(data)
    
    def render_statistical_tests(self, data: pd.DataFrame):
        """Render statistical tests interface."""
        test_type = st.selectbox(
            "Select Test Type",
            ["Normality Test", "Correlation Test", "T-Test", "ANOVA"]
        )
        
        if test_type == "Normality Test":
            col = st.selectbox("Select Column", data.select_dtypes(include=[np.number]).columns)
            stat, p_value = stats.normaltest(data[col].dropna())
            
            st.markdown(f"#### Normality Test Results for {col}")
            st.write(f"Statistic: {stat:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            
            # Visualize distribution
            fig = make_subplots(rows=1, cols=2)
            fig.add_trace(go.Histogram(x=data[col], name="Histogram"), row=1, col=1)
            fig.add_trace(go.Box(y=data[col], name="Box Plot"), row=1, col=2)
            fig.update_layout(title=f"Distribution of {col}", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        elif test_type == "Correlation Test":
            col1 = st.selectbox("Select First Column", data.select_dtypes(include=[np.number]).columns)
            col2 = st.selectbox("Select Second Column", data.select_dtypes(include=[np.number]).columns)
            
            corr, p_value = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
            
            st.markdown(f"#### Correlation Test Results")
            st.write(f"Correlation coefficient: {corr:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            
            # Visualize correlation
            fig = px.scatter(data, x=col1, y=col2, trendline="ols")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_dimensionality_reduction(self, data: pd.DataFrame):
        """Render dimensionality reduction interface."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Parameter selection
        n_components = st.slider("Number of Components", 2, min(10, len(numeric_cols)), 2)
        
        # Prepare data
        X = data[numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Display results
        st.markdown("#### PCA Results")
        
        # Explained variance ratio
        fig = px.bar(
            x=range(1, n_components + 1),
            y=pca.explained_variance_ratio_,
            labels={"x": "Principal Component", "y": "Explained Variance Ratio"}
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # 2D or 3D visualization of first components
        if n_components >= 2:
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                labels={"x": "PC1", "y": "PC2"}
            )
            if n_components >= 3:
                fig = px.scatter_3d(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    z=X_pca[:, 2],
                    labels={"x": "PC1", "y": "PC2", "z": "PC3"}
                )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_clustering(self, data: pd.DataFrame):
        """Render clustering interface."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Parameter selection
        features = st.multiselect("Select Features", numeric_cols, default=list(numeric_cols)[:2])
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        
        if len(features) < 2:
            st.warning("Please select at least 2 features for clustering.")
            return
        
        # Prepare data
        X = data[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Visualize results
        if len(features) == 2:
            fig = px.scatter(
                x=X[features[0]],
                y=X[features[1]],
                color=clusters,
                labels={"x": features[0], "y": features[1]}
            )
        else:
            # Use PCA for visualization if more than 2 features
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                color=clusters,
                labels={"x": "PC1", "y": "PC2"}
            )
        
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_time_series_analysis(self, data: pd.DataFrame):
        """Render time series analysis interface."""
        # Check for datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) == 0:
            st.warning("No datetime columns found in the dataset.")
            return
        
        # Parameter selection
        time_col = st.selectbox("Select Time Column", datetime_cols)
        value_col = st.selectbox("Select Value Column", data.select_dtypes(include=[np.number]).columns)
        
        # Set time column as index
        ts_data = data.set_index(time_col)[value_col]
        
        # Time series decomposition
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(ts_data, period=30)
            
            fig = make_subplots(rows=4, cols=1, subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
            fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, name='Original'), row=1, col=1)
            fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend, name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
            fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.resid, name='Residual'), row=4, col=1)
            
            fig.update_layout(height=800, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error performing time series decomposition: {str(e)}")
    
    def render_feature_analysis(self, data: pd.DataFrame):
        """Render feature analysis interface."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Feature statistics
        st.markdown("#### Feature Statistics")
        stats_df = data[numeric_cols].describe()
        st.dataframe(stats_df)
        
        # Correlation matrix
        st.markdown("#### Correlation Matrix")
        corr = data[numeric_cols].corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.markdown("#### Feature Distributions")
        selected_feature = st.selectbox("Select Feature", numeric_cols)
        
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Histogram(x=data[selected_feature], name="Distribution"), row=1, col=1)
        fig.add_trace(go.Box(y=data[selected_feature], name="Box Plot"), row=1, col=2)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    def save_script(self, code: str):
        """Save the current script to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"script_{timestamp}.py"
        
        try:
            with open(filename, 'w') as f:
                f.write(code)
            st.success(f"Script saved as {filename}")
        except Exception as e:
            st.error(f"Error saving script: {str(e)}")

    def load_sample_data(self) -> pd.DataFrame:
        """Load sample data for EDA."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'temperature': np.random.normal(25, 5, n_samples),
            'pressure': np.random.normal(100, 10, n_samples),
            'flow_rate': np.random.normal(50, 8, n_samples),
            'quality_score': np.random.uniform(85, 100, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['anomaly'] = df['temperature'].apply(
            lambda x: 'Yes' if x > 30 or x < 20 else 'No'
        )
        return df

def render_eda_workspace():
    """Render the EDA workspace with interactive analysis tools."""
    st.header("Exploratory Data Analysis Workspace")
    
    # Initialize workspace
    workspace = EDAWorkspace()
    
    # Sidebar controls
    st.sidebar.subheader("Data Settings")
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Sample Data", "Upload Data", "Live Data"]
    )
    
    # Load data based on selection
    if data_source == "Sample Data":
        data = workspace.load_sample_data()
    elif data_source == "Upload Data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file")
            return
    else:  # Live Data
        st.info("Live data connection will be implemented soon")
        return
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Overview",
        "Statistical Analysis",
        "Visualization",
        "Feature Engineering"
    ])
    
    with tab1:
        st.subheader("Data Overview")
        
        # Basic data info
        st.write("### Data Shape")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        
        # Display first few rows
        st.write("### Sample Data")
        st.dataframe(data.head())
        
        # Data types and missing values
        st.write("### Data Types and Missing Values")
        info_df = pd.DataFrame({
            'Data Type': data.dtypes,
            'Missing Values': data.isnull().sum(),
            'Missing %': (data.isnull().sum() / len(data) * 100).round(2)
        })
        st.dataframe(info_df)
    
    with tab2:
        st.subheader("Statistical Analysis")
        
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        selected_cols = st.multiselect(
            "Select Columns for Analysis",
            numerical_cols,
            default=list(numerical_cols)[:3]
        )
        
        if selected_cols:
            # Summary statistics
            st.write("### Summary Statistics")
            st.dataframe(data[selected_cols].describe())
            
            # Correlation analysis
            st.write("### Correlation Matrix")
            corr_matrix = data[selected_cols].corr()
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution analysis
            st.write("### Distribution Analysis")
            for col in selected_cols:
                fig = px.histogram(
                    data,
                    x=col,
                    title=f"Distribution of {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Visualization")
        
        # Plot type selector
        plot_type = st.selectbox(
            "Plot Type",
            ["Scatter Plot", "Line Plot", "Box Plot", "Violin Plot"]
        )
        
        # Column selectors
        if plot_type == "Scatter Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", numerical_cols)
            with col2:
                y_col = st.selectbox("Y-axis", numerical_cols)
            with col3:
                color_col = st.selectbox(
                    "Color by",
                    [None] + list(data.columns)
                )
            
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{x_col} vs {y_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Line Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", data.columns)
            with col2:
                y_cols = st.multiselect(
                    "Y-axis",
                    numerical_cols,
                    default=[numerical_cols[0]]
                )
            
            fig = go.Figure()
            for col in y_cols:
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[col],
                    name=col
                ))
            fig.update_layout(title=f"Time Series Plot")
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type in ["Box Plot", "Violin Plot"]:
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Value Column", numerical_cols)
            with col2:
                group_col = st.selectbox(
                    "Group by",
                    [None] + list(data.select_dtypes(exclude=[np.number]).columns)
                )
            
            if plot_type == "Box Plot":
                fig = px.box(
                    data,
                    y=y_col,
                    x=group_col,
                    title=f"Box Plot of {y_col}"
                )
            else:
                fig = px.violin(
                    data,
                    y=y_col,
                    x=group_col,
                    title=f"Violin Plot of {y_col}"
                )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Engineering")
        
        # Feature engineering options
        operation = st.selectbox(
            "Operation",
            ["Normalization", "Binning", "Log Transform", "Custom Formula"]
        )
        
        if operation == "Normalization":
            col = st.selectbox("Select Column", numerical_cols)
            method = st.selectbox(
                "Normalization Method",
                ["Min-Max", "Standard", "Robust"]
            )
            
            if st.button("Apply Normalization"):
                if method == "Min-Max":
                    normalized = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                elif method == "Standard":
                    normalized = (data[col] - data[col].mean()) / data[col].std()
                else:  # Robust
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    normalized = (data[col] - data[col].median()) / iqr
                
                # Display results
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data[col],
                    name="Original"
                ))
                fig.add_trace(go.Histogram(
                    x=normalized,
                    name="Normalized"
                ))
                fig.update_layout(
                    title=f"Original vs Normalized Distribution of {col}",
                    barmode='overlay'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif operation == "Binning":
            col = st.selectbox("Select Column", numerical_cols)
            num_bins = st.slider("Number of Bins", 2, 20, 5)
            
            if st.button("Apply Binning"):
                binned = pd.qcut(data[col], num_bins, labels=False)
                
                fig = px.histogram(
                    x=binned,
                    title=f"Distribution of Binned {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif operation == "Log Transform":
            col = st.selectbox("Select Column", numerical_cols)
            
            if st.button("Apply Log Transform"):
                if (data[col] <= 0).any():
                    st.warning("Cannot apply log transform to non-positive values")
                else:
                    log_transformed = np.log(data[col])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=data[col],
                        name="Original"
                    ))
                    fig.add_trace(go.Histogram(
                        x=log_transformed,
                        name="Log Transformed"
                    ))
                    fig.update_layout(
                        title=f"Original vs Log Transformed Distribution of {col}",
                        barmode='overlay'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:  # Custom Formula
            st.write("### Custom Formula")
            col1 = st.selectbox("Select Column 1", numerical_cols)
            operation = st.selectbox("Operation", ["+", "-", "*", "/"])
            col2 = st.selectbox("Select Column 2", numerical_cols)
            
            if st.button("Apply Formula"):
                if operation == "+":
                    result = data[col1] + data[col2]
                elif operation == "-":
                    result = data[col1] - data[col2]
                elif operation == "*":
                    result = data[col1] * data[col2]
                else:  # division
                    if (data[col2] == 0).any():
                        st.warning("Cannot divide by zero")
                        return
                    result = data[col1] / data[col2]
                
                fig = px.histogram(
                    x=result,
                    title=f"Distribution of {col1} {operation} {col2}"
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_eda_workspace() 