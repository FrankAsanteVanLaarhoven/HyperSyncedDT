import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

def generate_sample_parameter_data() -> Dict[str, pd.DataFrame]:
    """Generate sample process parameter data."""
    np.random.seed(42)
    
    # Generate current parameter settings
    parameters = pd.DataFrame({
        'parameter_id': [
            'Temperature', 'Pressure', 'Flow Rate',
            'Speed', 'Feed Rate', 'Coolant Level'
        ],
        'current_value': [
            185.5, 2.3, 45.0,
            1200, 15.5, 85.0
        ],
        'unit': [
            '°C', 'bar', 'L/min',
            'RPM', 'mm/min', '%'
        ],
        'min_limit': [
            150, 1.8, 35,
            800, 10, 60
        ],
        'max_limit': [
            200, 2.8, 55,
            1500, 20, 100
        ],
        'optimal_value': [
            180, 2.4, 48,
            1250, 15, 90
        ]
    })
    
    # Generate historical data
    timestamps = pd.date_range(
        start='2024-03-17 00:00:00',
        end='2024-03-17 23:59:59',
        freq='5min'
    )
    
    historical_data = []
    for param in parameters.itertuples():
        base_value = param.current_value
        variation = (param.max_limit - param.min_limit) * 0.1
        values = base_value + np.random.normal(0, variation, len(timestamps))
        
        for ts, val in zip(timestamps, values):
            historical_data.append({
                'timestamp': ts,
                'parameter_id': param.parameter_id,
                'value': val,
                'unit': param.unit
            })
    
    historical_df = pd.DataFrame(historical_data)
    
    return {
        'parameters': parameters,
        'historical': historical_df
    }

def render_process_parameters():
    """Render the process parameters dashboard."""
    st.header("Process Parameters Dashboard")
    
    # Initialize components
    digital_twin = SynchronizedDigitalTwin()
    visualizer = MultiModalVisualizer()
    
    # Sidebar controls
    st.sidebar.subheader("Parameter Settings")
    parameter_group = st.sidebar.selectbox(
        "Parameter Group",
        ["All Parameters", "Temperature Control", "Pressure Control", "Flow Control"]
    )
    
    view_mode = st.sidebar.selectbox(
        "View Mode",
        ["Real-time", "Historical", "Statistical"]
    )
    
    # Generate sample data
    param_data = generate_sample_parameter_data()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Parameter Overview",
        "Trend Analysis",
        "Control Limits",
        "Settings"
    ])
    
    with tab1:
        st.subheader("Current Parameter Status")
        
        # Parameter status cards
        for _, param in param_data['parameters'].iterrows():
            # Calculate status color based on limits
            value = param['current_value']
            if value < param['min_limit'] or value > param['max_limit']:
                status_color = 'red'
            elif abs(value - param['optimal_value']) > (param['max_limit'] - param['min_limit']) * 0.1:
                status_color = 'orange'
            else:
                status_color = 'green'
            
            # Calculate deviation from optimal
            deviation = ((value - param['optimal_value']) / param['optimal_value']) * 100
            
            st.markdown(f"""
            <div style='padding: 10px; border-left: 5px solid {status_color}; margin: 10px 0;'>
                <strong>{param['parameter_id']}</strong><br>
                Current Value: {value:.1f} {param['unit']}<br>
                Optimal Value: {param['optimal_value']:.1f} {param['unit']}<br>
                Deviation: {deviation:+.1f}%<br>
                Limits: [{param['min_limit']:.1f}, {param['max_limit']:.1f}] {param['unit']}
            </div>
            """, unsafe_allow_html=True)
        
        # Parameter correlation heatmap
        st.subheader("Parameter Correlations")
        historical_pivot = param_data['historical'].pivot(
            index='timestamp',
            columns='parameter_id',
            values='value'
        )
        correlation_matrix = historical_pivot.corr()
        
        fig = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            title="Parameter Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Parameter Trends")
        
        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 4 Hours", "Last 8 Hours", "Last 24 Hours"]
        )
        
        # Parameter selector
        selected_params = st.multiselect(
            "Select Parameters",
            param_data['parameters']['parameter_id'].tolist(),
            default=param_data['parameters']['parameter_id'].tolist()[:3]
        )
        
        if selected_params:
            # Filter historical data
            filtered_data = param_data['historical'][
                param_data['historical']['parameter_id'].isin(selected_params)
            ]
            
            # Create trend chart
            fig = go.Figure()
            
            for param_id in selected_params:
                param_data_filtered = filtered_data[
                    filtered_data['parameter_id'] == param_id
                ]
                
                fig.add_trace(go.Scatter(
                    x=param_data_filtered['timestamp'],
                    y=param_data_filtered['value'],
                    name=param_id,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Parameter Trends Over Time",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Control Limits Analysis")
        
        # Parameter selector for control limits
        selected_param = st.selectbox(
            "Select Parameter",
            param_data['parameters']['parameter_id'].tolist()
        )
        
        if selected_param:
            # Get parameter details
            param_details = param_data['parameters'][
                param_data['parameters']['parameter_id'] == selected_param
            ].iloc[0]
            
            # Get historical data for selected parameter
            historical_param = param_data['historical'][
                param_data['historical']['parameter_id'] == selected_param
            ]
            
            # Calculate statistics
            mean_value = historical_param['value'].mean()
            std_value = historical_param['value'].std()
            ucl = mean_value + 3 * std_value
            lcl = mean_value - 3 * std_value
            
            # Create control chart
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=historical_param['timestamp'],
                y=historical_param['value'],
                name='Actual Value',
                mode='lines'
            ))
            
            # Add control limits
            fig.add_trace(go.Scatter(
                x=historical_param['timestamp'],
                y=[ucl] * len(historical_param),
                name='Upper Control Limit',
                line=dict(dash='dash', color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=historical_param['timestamp'],
                y=[lcl] * len(historical_param),
                name='Lower Control Limit',
                line=dict(dash='dash', color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=historical_param['timestamp'],
                y=[mean_value] * len(historical_param),
                name='Mean',
                line=dict(dash='dot', color='green')
            ))
            
            fig.update_layout(
                title=f"Control Chart for {selected_param}",
                xaxis_title="Time",
                yaxis_title=f"Value ({param_details['unit']})",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{mean_value:.2f} {param_details['unit']}")
            with col2:
                st.metric("Standard Deviation", f"{std_value:.2f} {param_details['unit']}")
            with col3:
                violations = len(historical_param[
                    (historical_param['value'] > ucl) |
                    (historical_param['value'] < lcl)
                ])
                st.metric("Control Violations", str(violations))
    
    with tab4:
        st.subheader("Parameter Settings")
        
        # Parameter configuration table
        st.write("### Parameter Configuration")
        
        edited_df = st.data_editor(
            param_data['parameters'],
            hide_index=True,
            use_container_width=True,
            disabled=["parameter_id"]
        )
        
        # Save changes button
        if st.button("Save Parameter Changes"):
            st.success("Parameter changes saved successfully!")
            
            # Show confirmation of changes
            st.write("### Changes Summary")
            changes = edited_df.compare(param_data['parameters'])
            if not changes.empty:
                st.dataframe(changes)
            else:
                st.info("No changes detected") 