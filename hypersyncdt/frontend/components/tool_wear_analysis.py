import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ToolWearAnalysis:
    def __init__(self):
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'tool_wear_data' not in st.session_state:
            # Generate sample data if none exists
            st.session_state.tool_wear_data = self._generate_sample_data()
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
            
    def _generate_sample_data(self):
        """Generate sample tool wear data"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'timestamp': [datetime.now() - timedelta(hours=x) for x in range(n_samples)],
            'cutting_speed': np.random.normal(100, 10, n_samples),
            'feed_rate': np.random.normal(0.2, 0.05, n_samples),
            'cutting_depth': np.random.normal(2, 0.3, n_samples),
            'vibration': np.random.normal(0.5, 0.1, n_samples),
            'acoustic_emission': np.random.normal(70, 5, n_samples),
            'temperature': np.random.normal(150, 15, n_samples),
            'tool_wear': np.zeros(n_samples)
        }
        
        # Generate realistic tool wear progression
        base_wear = np.linspace(0, 1, n_samples)
        noise = np.random.normal(0, 0.05, n_samples)
        data['tool_wear'] = base_wear + noise
        data['tool_wear'] = np.clip(data['tool_wear'], 0, 1)
        
        return pd.DataFrame(data)

    def render_real_time_monitoring(self):
        """Render real-time tool wear monitoring interface"""
        st.markdown("""
        <style>
        .metric-card {
            background: rgba(22, 26, 30, 0.95);
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        }
        .wear-indicator {
            height: 8px;
            background: rgba(30, 35, 40, 0.8);
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        }
        .wear-progress {
            height: 100%;
            transition: width 0.5s ease;
        }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("Real-time Tool Wear Monitoring")
        
        # Current tool status
        current_data = st.session_state.tool_wear_data.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            wear_level = current_data['tool_wear']
            color = 'rgb(74, 222, 128)' if wear_level < 0.7 else 'rgb(251, 191, 36)' if wear_level < 0.9 else 'rgb(239, 68, 68)'
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: rgb(156, 163, 175);">Tool Wear Level</div>
                <div style="font-size: 1.5em; color: {color};">{wear_level:.1%}</div>
                <div class="wear-indicator">
                    <div class="wear-progress" style="width: {wear_level*100}%; background: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.metric("Cutting Speed", f"{current_data['cutting_speed']:.1f} m/min")
        with col3:
            st.metric("Temperature", f"{current_data['temperature']:.1f}°C")
        with col4:
            st.metric("Vibration", f"{current_data['vibration']:.3f} mm/s")
            
        # Real-time trends
        st.subheader("Parameter Trends")
        recent_data = st.session_state.tool_wear_data.tail(100)
        
        fig = go.Figure()
        
        # Tool wear trend
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['tool_wear'],
            name='Tool Wear',
            line=dict(color='rgb(74, 222, 128)')
        ))
        
        # Add other parameters
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['temperature'] / recent_data['temperature'].max(),
            name='Temperature (normalized)',
            line=dict(color='rgb(251, 191, 36)', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['vibration'] / recent_data['vibration'].max(),
            name='Vibration (normalized)',
            line=dict(color='rgb(147, 51, 234)', dash='dot')
        ))
        
        fig.update_layout(
            title='Real-time Parameter Trends',
            xaxis_title='Time',
            yaxis_title='Value',
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(22, 26, 30, 0.95)',
            font=dict(color='rgb(209, 213, 219)')
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_predictive_analysis(self):
        """Render predictive analysis interface"""
        st.subheader("Predictive Analysis")
        
        # Model configuration
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Model Configuration")
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            
        with col2:
            st.markdown("### Feature Selection")
            features = st.multiselect(
                "Select Features for Prediction",
                ['cutting_speed', 'feed_rate', 'cutting_depth', 'vibration', 'acoustic_emission', 'temperature'],
                ['cutting_speed', 'feed_rate', 'cutting_depth', 'vibration']
            )
        
        # Train model button
        if st.button("Train Predictive Model"):
            with st.spinner("Training model..."):
                # Prepare data
                X = st.session_state.tool_wear_data[features]
                y = st.session_state.tool_wear_data['tool_wear']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Train model
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Store results
                st.session_state.trained_model = model
                st.session_state.predictions = {
                    'true': y_test,
                    'pred': y_pred,
                    'features': features
                }
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.success("Model trained successfully!")
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col2:
                    st.metric("R² Score", f"{r2:.4f}")
        
        # Show predictions if model is trained
        if st.session_state.predictions is not None:
            st.subheader("Model Performance")
            
            # Prediction vs Actual plot
            fig = go.Figure()
            
            # Convert pandas Series to numpy arrays
            true_values = np.array(st.session_state.predictions['true'])
            pred_values = np.array(st.session_state.predictions['pred'])
            x_values = np.arange(len(true_values))
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=true_values,
                name='Actual',
                line=dict(color='rgb(74, 222, 128)')
            ))
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=pred_values,
                name='Predicted',
                line=dict(color='rgb(251, 191, 36)', dash='dash')
            ))
            
            fig.update_layout(
                title='Actual vs Predicted Tool Wear',
                xaxis_title='Sample',
                yaxis_title='Tool Wear',
                template='plotly_dark',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(22, 26, 30, 0.95)',
                font=dict(color='rgb(209, 213, 219)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if st.session_state.trained_model is not None:
                importances = st.session_state.trained_model.feature_importances_
                feat_imp = pd.DataFrame({
                    'Feature': st.session_state.predictions['features'],
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=feat_imp['Importance'],
                    y=feat_imp['Feature'],
                    orientation='h',
                    marker_color='rgb(74, 222, 128)'
                ))
                
                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance',
                    yaxis_title='Feature',
                    template='plotly_dark',
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(22, 26, 30, 0.95)',
                    font=dict(color='rgb(209, 213, 219)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Prediction interface
        st.subheader("Make Predictions")
        if st.session_state.trained_model is not None:
            col1, col2, col3 = st.columns(3)
            input_values = {}
            
            for i, feature in enumerate(st.session_state.predictions['features']):
                with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                    input_values[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        value=float(st.session_state.tool_wear_data[feature].mean()),
                        format="%.3f"
                    )
            
            if st.button("Predict Tool Wear"):
                input_data = pd.DataFrame([input_values])
                prediction = st.session_state.trained_model.predict(input_data)[0]
                
                # Display prediction with appropriate styling
                wear_level = prediction
                color = 'rgb(74, 222, 128)' if wear_level < 0.7 else 'rgb(251, 191, 36)' if wear_level < 0.9 else 'rgb(239, 68, 68)'
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: rgb(156, 163, 175);">Predicted Tool Wear</div>
                    <div style="font-size: 1.5em; color: {color};">{wear_level:.1%}</div>
                    <div class="wear-indicator">
                        <div class="wear-progress" style="width: {wear_level*100}%; background: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_tool_wear_analysis():
    """Render the Tool Wear Analysis page"""
    st.title("Tool Wear Analysis")
    
    # Initialize tool wear analysis if not in session state
    if 'tool_wear_analyzer' not in st.session_state:
        st.session_state.tool_wear_analyzer = ToolWearAnalysis()
    
    # Create tabs for different analysis views
    tab1, tab2 = st.tabs(["Real-time Monitoring", "Predictive Analysis"])
    
    with tab1:
        st.session_state.tool_wear_analyzer.render_real_time_monitoring()
    
    with tab2:
        st.session_state.tool_wear_analyzer.render_predictive_analysis()

if __name__ == "__main__":
    render_tool_wear_analysis() 