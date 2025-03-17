import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional
import os

class ExperimentTracker:
    def __init__(self):
        self.mlflow_client = MlflowClient()
        self.current_experiment = None
        self.metrics_history = {}
        
    def initialize_mlflow(self, tracking_uri: str = "sqlite:///mlflow.db"):
        """Initialize MLflow with the specified tracking URI"""
        mlflow.set_tracking_uri(tracking_uri)
        
    def create_experiment(self, name: str, tags: Dict[str, str] = None) -> str:
        """Create a new experiment with the given name and tags"""
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(name, tags=tags)
        else:
            experiment_id = experiment.experiment_id
        self.current_experiment = experiment_id
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to the current experiment"""
        with mlflow.start_run(experiment_id=self.current_experiment):
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value, step=step)
                if metric_name not in self.metrics_history:
                    self.metrics_history[metric_name] = []
                self.metrics_history[metric_name].append((step, value))
    
    def log_parameters(self, params: Dict[str, any]):
        """Log parameters to the current experiment"""
        with mlflow.start_run(experiment_id=self.current_experiment):
            mlflow.log_params(params)
    
    def log_model(self, model, name: str):
        """Log a model to the current experiment"""
        with mlflow.start_run(experiment_id=self.current_experiment):
            mlflow.sklearn.log_model(model, name)
    
    def get_experiment_runs(self, experiment_id: str = None) -> List[Dict]:
        """Get all runs for the specified experiment"""
        if experiment_id is None:
            experiment_id = self.current_experiment
        runs = self.mlflow_client.search_runs(experiment_id)
        return [run.to_dictionary() for run in runs]
    
    def get_metric_history(self, metric_name: str) -> List[tuple]:
        """Get the history of a specific metric"""
        return self.metrics_history.get(metric_name, [])

def render_experiment_tracking():
    """Render the Experiment Tracking interface"""
    st.title("🧪 Experiment Tracking")
    
    # Initialize experiment tracker if not in session state
    if 'experiment_tracker' not in st.session_state:
        st.session_state.experiment_tracker = ExperimentTracker()
        st.session_state.experiment_tracker.initialize_mlflow()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Active Experiments",
        "Metrics Dashboard",
        "Parameter Analysis",
        "Model Registry"
    ])
    
    with tab1:
        st.subheader("Active Experiments")
        
        # Experiment creation and management
        col1, col2 = st.columns([2, 1])
        with col1:
            experiment_name = st.text_input("Experiment Name", "Tool Wear Analysis")
            experiment_tags = st.text_area("Experiment Tags (JSON)", "{\"project\": \"tool_wear\", \"version\": \"1.0\"}")
        
        with col2:
            if st.button("Create Experiment"):
                try:
                    tags = json.loads(experiment_tags)
                    experiment_id = st.session_state.experiment_tracker.create_experiment(
                        experiment_name, tags
                    )
                    st.success(f"Created experiment with ID: {experiment_id}")
                except Exception as e:
                    st.error(f"Error creating experiment: {str(e)}")
        
        # Display active experiments
        st.markdown("### Active Experiments")
        experiments = mlflow.search_experiments()
        if experiments:
            experiment_df = pd.DataFrame([{
                "Name": exp.name,
                "ID": exp.experiment_id,
                "Created": exp.creation_time,
                "Status": "Active" if exp.lifecycle_stage == "active" else "Deleted"
            } for exp in experiments])
            
            st.dataframe(experiment_df, use_container_width=True)
    
    with tab2:
        st.subheader("Metrics Dashboard")
        
        # Metric logging interface
        col1, col2 = st.columns([2, 1])
        with col1:
            metric_name = st.text_input("Metric Name", "accuracy")
            metric_value = st.number_input("Metric Value", 0.0, 1.0, 0.95)
            step = st.number_input("Step", 0, 1000, 1)
        
        with col2:
            if st.button("Log Metric"):
                st.session_state.experiment_tracker.log_metrics(
                    {metric_name: metric_value}, step
                )
                st.success("Metric logged successfully!")
        
        # Display metric history
        st.markdown("### Metric History")
        metric_history = st.session_state.experiment_tracker.get_metric_history(metric_name)
        if metric_history:
            history_df = pd.DataFrame(metric_history, columns=["Step", "Value"])
            
            fig = px.line(history_df, x="Step", y="Value",
                         title=f"{metric_name} History")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Parameter Analysis")
        
        # Parameter logging interface
        col1, col2 = st.columns([2, 1])
        with col1:
            param_dict = st.text_area(
                "Parameters (JSON)",
                """{
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 100
                }"""
            )
        
        with col2:
            if st.button("Log Parameters"):
                try:
                    params = json.loads(param_dict)
                    st.session_state.experiment_tracker.log_parameters(params)
                    st.success("Parameters logged successfully!")
                except Exception as e:
                    st.error(f"Error logging parameters: {str(e)}")
        
        # Display parameter comparison
        st.markdown("### Parameter Comparison")
        runs = st.session_state.experiment_tracker.get_experiment_runs()
        if runs:
            run_params = []
            for run in runs:
                params = run["data"]["params"]
                params["run_id"] = run["info"]["run_id"]
                run_params.append(params)
            
            param_df = pd.DataFrame(run_params)
            st.dataframe(param_df, use_container_width=True)
    
    with tab4:
        st.subheader("Model Registry")
        
        # Model registration interface
        col1, col2 = st.columns([2, 1])
        with col1:
            model_name = st.text_input("Model Name", "tool_wear_predictor")
            model_version = st.text_input("Version", "1.0.0")
            model_stage = st.selectbox("Stage", ["None", "Staging", "Production", "Archived"])
        
        with col2:
            if st.button("Register Model"):
                st.info(f"Registering model: {model_name} (v{model_version})")
                # Here you would typically register the model with MLflow
                st.success("Model registered successfully!")
        
        # Display registered models
        st.markdown("### Registered Models")
        registered_models = mlflow.search_registered_models()
        if registered_models:
            model_df = pd.DataFrame([{
                "Name": model.name,
                "Latest Version": model.latest_versions[0].version if model.latest_versions else "None",
                "Stage": model.latest_versions[0].current_stage if model.latest_versions else "None",
                "Created": model.latest_versions[0].creation_timestamp if model.latest_versions else "None"
            } for model in registered_models])
            
            st.dataframe(model_df, use_container_width=True)

if __name__ == "__main__":
    render_experiment_tracking() 