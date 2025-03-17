import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

def generate_sample_experiment_data() -> Dict[str, pd.DataFrame]:
    """Generate sample experiment design data."""
    np.random.seed(42)
    
    # Generate experiment configurations
    experiments = pd.DataFrame({
        'experiment_id': [f'EXP-{i:03d}' for i in range(1, 21)],
        'title': [
            'Temperature Impact Analysis',
            'Pressure Optimization Study',
            'Flow Rate Evaluation',
            'Material Properties Test',
            'Process Speed Analysis'
        ] * 4,
        'status': np.random.choice(
            ['Planned', 'In Progress', 'Completed', 'On Hold'],
            20
        ),
        'priority': np.random.choice(
            ['High', 'Medium', 'Low'],
            20
        ),
        'start_date': pd.date_range(
            start='2024-03-01',
            periods=20,
            freq='D'
        ),
        'duration_days': np.random.randint(5, 30, 20)
    })
    
    # Generate factors data
    factors = []
    for exp_id in experiments['experiment_id']:
        num_factors = np.random.randint(2, 5)
        for i in range(num_factors):
            factors.append({
                'experiment_id': exp_id,
                'factor_name': np.random.choice([
                    'Temperature',
                    'Pressure',
                    'Flow Rate',
                    'Speed',
                    'Concentration'
                ]),
                'low_level': np.random.uniform(10, 50),
                'high_level': np.random.uniform(51, 100),
                'units': np.random.choice([
                    '°C',
                    'bar',
                    'L/min',
                    'RPM',
                    '%'
                ])
            })
    
    factors_df = pd.DataFrame(factors)
    
    # Generate response variables
    responses = []
    for exp_id in experiments['experiment_id']:
        num_responses = np.random.randint(1, 4)
        for i in range(num_responses):
            responses.append({
                'experiment_id': exp_id,
                'response_name': np.random.choice([
                    'Quality Score',
                    'Yield',
                    'Defect Rate',
                    'Processing Time',
                    'Energy Consumption'
                ]),
                'target_value': np.random.uniform(80, 100),
                'tolerance': np.random.uniform(1, 5),
                'units': np.random.choice([
                    '%',
                    'kg/h',
                    'ppm',
                    'min',
                    'kWh'
                ])
            })
    
    responses_df = pd.DataFrame(responses)
    
    return {
        'experiments': experiments,
        'factors': factors_df,
        'responses': responses_df
    }

def render_experiment_design():
    """Render the experiment design dashboard."""
    st.header("Experiment Design Dashboard")
    
    # Initialize components
    digital_twin = SynchronizedDigitalTwin()
    visualizer = MultiModalVisualizer()
    
    # Sidebar controls
    st.sidebar.subheader("Experiment Settings")
    experiment_type = st.sidebar.selectbox(
        "Experiment Type",
        ["Factorial Design", "Response Surface", "Screening", "Optimization"]
    )
    
    view_mode = st.sidebar.selectbox(
        "View Mode",
        ["Design", "Analysis", "Results", "Documentation"]
    )
    
    # Generate sample data
    experiment_data = generate_sample_experiment_data()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Experiment Overview",
        "Design Matrix",
        "Analysis",
        "Results"
    ])
    
    with tab1:
        st.subheader("Experiment Overview")
        
        # Add new experiment button
        if st.button("Create New Experiment"):
            st.info("Creating new experiment...")
        
        # Experiment status summary
        status_counts = experiment_data['experiments']['status'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Experiments", len(experiment_data['experiments']))
        with col2:
            st.metric("In Progress", len(experiment_data['experiments'][
                experiment_data['experiments']['status'] == 'In Progress'
            ]))
        with col3:
            st.metric("Completed", len(experiment_data['experiments'][
                experiment_data['experiments']['status'] == 'Completed'
            ]))
        with col4:
            st.metric("Planned", len(experiment_data['experiments'][
                experiment_data['experiments']['status'] == 'Planned'
            ]))
        
        # Experiment timeline
        fig = px.timeline(
            experiment_data['experiments'],
            x_start='start_date',
            x_end=experiment_data['experiments'].apply(
                lambda x: x['start_date'] + pd.Timedelta(days=x['duration_days']),
                axis=1
            ),
            y='experiment_id',
            color='status',
            title='Experiment Timeline'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Experiment details table
        st.write("### Experiment Details")
        st.dataframe(
            experiment_data['experiments'].style.highlight_max(axis=0)
        )
    
    with tab2:
        st.subheader("Design Matrix")
        
        # Experiment selector
        selected_experiment = st.selectbox(
            "Select Experiment",
            experiment_data['experiments']['experiment_id'].tolist(),
            format_func=lambda x: experiment_data['experiments'][
                experiment_data['experiments']['experiment_id'] == x
            ]['title'].iloc[0]
        )
        
        if selected_experiment:
            # Get experiment factors
            exp_factors = experiment_data['factors'][
                experiment_data['factors']['experiment_id'] == selected_experiment
            ]
            
            # Get experiment responses
            exp_responses = experiment_data['responses'][
                experiment_data['responses']['experiment_id'] == selected_experiment
            ]
            
            # Display factor settings
            st.write("### Factors")
            st.dataframe(exp_factors)
            
            # Display response variables
            st.write("### Response Variables")
            st.dataframe(exp_responses)
            
            # Generate design matrix
            st.write("### Design Matrix")
            
            # Create a full factorial design
            factors = exp_factors['factor_name'].tolist()
            levels = [-1, 1]  # Coded levels
            
            # Generate all combinations
            design_matrix = []
            for combination in np.array(np.meshgrid(
                *[levels for _ in factors]
            )).T.reshape(-1, len(factors)):
                run = {'Run': len(design_matrix) + 1}
                for factor, level in zip(factors, combination):
                    run[factor] = level
                design_matrix.append(run)
            
            design_df = pd.DataFrame(design_matrix)
            st.dataframe(design_df)
            
            # Power analysis
            st.write("### Power Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Statistical Power", "0.85")
            with col2:
                st.metric("Required Replicates", "3")
    
    with tab3:
        st.subheader("Analysis Tools")
        
        # Analysis type selector
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Main Effects", "Interactions", "Response Surface", "Optimization"]
        )
        
        if analysis_type == "Main Effects":
            # Sample main effects plot
            fig = go.Figure()
            
            # Add sample data
            factors = ['Temperature', 'Pressure', 'Flow Rate']
            effects = [2.5, -1.8, 3.2]
            
            fig.add_trace(go.Bar(
                x=factors,
                y=effects,
                name='Main Effects'
            ))
            
            fig.update_layout(
                title='Main Effects Plot',
                xaxis_title='Factors',
                yaxis_title='Effect Size'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Interactions":
            # Sample interaction plot
            x = np.linspace(-1, 1, 100)
            y1 = 2 * x + 1
            y2 = -2 * x + 1
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y1,
                name='Level 1'
            ))
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y2,
                name='Level 2'
            ))
            
            fig.update_layout(
                title='Interaction Plot',
                xaxis_title='Factor A',
                yaxis_title='Response'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Results Summary")
        
        # Filter completed experiments
        completed_experiments = experiment_data['experiments'][
            experiment_data['experiments']['status'] == 'Completed'
        ]
        
        if not completed_experiments.empty:
            # Results summary
            st.write("### Completed Experiments")
            st.dataframe(completed_experiments)
            
            # Generate sample results
            results = pd.DataFrame({
                'Factor': ['Temperature', 'Pressure', 'Flow Rate'],
                'Significance': [0.001, 0.05, 0.01],
                'Effect Size': [2.5, 1.8, 3.2],
                'Optimal Level': [75, 2.4, 45]
            })
            
            st.write("### Statistical Analysis")
            st.dataframe(results)
            
            # Optimization results
            st.write("### Optimization Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Optimal Response", "95.5%")
            with col2:
                st.metric("Confidence Interval", "±2.3%")
            
            # Validation suggestion
            st.info("""
            **Validation Recommendation:**
            Run confirmation trials at the optimal settings to verify the predicted response.
            Suggested number of replicates: 3
            """)
        
        else:
            st.warning("No completed experiments available for analysis.") 