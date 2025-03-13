import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import plotly.express as px
from advanced_visualizations import MultiModalVisualizer

class ModelPerformanceTracker:
    def __init__(self):
        self.visualizer = MultiModalVisualizer()
        
    def render_model_metrics(self, model_metrics: Dict[str, Dict[str, float]]):
        """Render model performance metrics in a modern card layout."""
        for model_name, metrics in model_metrics.items():
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="color: rgba(100, 255, 200, 0.9);">{model_name}</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            """, unsafe_allow_html=True)
            
            for metric_name, value in metrics.items():
                st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px;">
                        <h4 style="color: rgba(100, 255, 200, 0.7);">{metric_name}</h4>
                        <p style="font-size: 1.5rem; margin: 0;">{value:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    def render_training_history(self, history: Dict[str, List[float]], model_name: str):
        """Render training history visualization."""
        fig = go.Figure()
        
        for metric, values in history.items():
            fig.add_trace(go.Scatter(
                y=values,
                name=metric,
                mode='lines',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"{model_name} Training History",
            xaxis_title="Epoch",
            yaxis_title="Value",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor='rgba(255,255,255,0.1)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_confusion_matrix(self, confusion_matrix: np.ndarray, labels: List[str]):
        """Render confusion matrix visualization."""
        fig = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="Actual"),
            x=labels,
            y=labels,
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            title="Confusion Matrix",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_importance(self, features: List[str], importance: List[float]):
        """Render feature importance visualization."""
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color='rgba(100, 255, 200, 0.6)',
                line=dict(color='rgba(100, 255, 200, 0.8)', width=2)
            )
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_prediction_analysis(self, actual: List[float], predicted: List[float]):
        """Render prediction vs actual analysis."""
        fig = go.Figure()
        
        # Scatter plot of predictions
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(
                color='rgba(100, 255, 200, 0.6)',
                size=8,
                line=dict(color='rgba(100, 255, 200, 0.8)', width=1)
            )
        ))
        
        # Perfect prediction line
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash')
        ))
        
        fig.update_layout(
            title="Prediction Analysis",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_model_performance():
    """Render the Model Performance page with comprehensive analytics."""
    st.title("Model Performance Analytics")
    
    # Initialize performance tracker
    tracker = ModelPerformanceTracker()
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Overview",
        "Training History",
        "Model Evaluation",
        "Feature Analysis"
    ])
    
    with tab1:
        st.header("Model Performance Overview")
        
        # Sample model metrics
        model_metrics = {
            "Quantum-Enhanced Neural Network": {
                "Accuracy": 0.956,
                "Precision": 0.943,
                "Recall": 0.967,
                "F1 Score": 0.955,
                "ROC AUC": 0.989
            },
            "Hybrid LSTM-GNN": {
                "Accuracy": 0.934,
                "Precision": 0.921,
                "Recall": 0.945,
                "F1 Score": 0.933,
                "ROC AUC": 0.978
            }
        }
        
        tracker.render_model_metrics(model_metrics)
    
    with tab2:
        st.header("Training History Analysis")
        
        # Sample training history
        history = {
            "loss": [0.5, 0.3, 0.2, 0.15, 0.12],
            "accuracy": [0.8, 0.85, 0.9, 0.92, 0.95],
            "val_loss": [0.55, 0.35, 0.25, 0.18, 0.14],
            "val_accuracy": [0.75, 0.82, 0.87, 0.90, 0.93]
        }
        
        model_selection = st.selectbox(
            "Select Model",
            ["Quantum-Enhanced Neural Network", "Hybrid LSTM-GNN"]
        )
        
        tracker.render_training_history(history, model_selection)
    
    with tab3:
        st.header("Model Evaluation")
        
        # Sample confusion matrix data
        labels = ["Class A", "Class B", "Class C", "Class D"]
        confusion_matrix = np.array([
            [45, 2, 1, 0],
            [1, 42, 2, 1],
            [2, 1, 43, 2],
            [0, 1, 2, 45]
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            tracker.render_confusion_matrix(confusion_matrix, labels)
        
        with col2:
            # Sample prediction analysis data
            actual = np.random.normal(0, 1, 100)
            predicted = actual + np.random.normal(0, 0.2, 100)
            tracker.render_prediction_analysis(actual.tolist(), predicted.tolist())
    
    with tab4:
        st.header("Feature Analysis")
        
        # Sample feature importance data
        features = [
            "Temperature",
            "Pressure",
            "Flow Rate",
            "Vibration",
            "Power Load",
            "Humidity",
            "Speed",
            "Torque"
        ]
        importance = [0.85, 0.76, 0.72, 0.68, 0.65, 0.58, 0.52, 0.48]
        
        tracker.render_feature_importance(features, importance)
        
        # Add feature correlation analysis
        st.subheader("Feature Correlations")
        
        # Generate sample correlation data
        n_samples = 100
        feature_data = pd.DataFrame({
            feature: np.random.normal(0, 1, n_samples) for feature in features
        })
        
        # Add some correlations
        feature_data["Pressure"] = feature_data["Temperature"] * 0.7 + np.random.normal(0, 0.3, n_samples)
        feature_data["Flow Rate"] = feature_data["Pressure"] * 0.6 + np.random.normal(0, 0.4, n_samples)
        
        correlation_matrix = feature_data.corr()
        
        fig = px.imshow(
            correlation_matrix,
            labels=dict(x="Features", y="Features"),
            x=features,
            y=features,
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_model_performance() 