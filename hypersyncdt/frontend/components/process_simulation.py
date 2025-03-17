import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import norm, uniform, triang
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

class ProcessSimulator:
    def __init__(self):
        self._initialize_session_state()
        self.setup_neural_engine()
        
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None
        if 'optimization_history' not in st.session_state:
            st.session_state.optimization_history = []
        if 'current_scenario' not in st.session_state:
            st.session_state.current_scenario = None
            
    def setup_neural_engine(self):
        """Initialize neural network for process simulation"""
        class ProcessNN(nn.Module):
            def __init__(self, input_size=6, hidden_size=64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size//2),
                    nn.ReLU(),
                    nn.Linear(hidden_size//2, 4)  # Output: Quality, Efficiency, Time, Cost
                )
                
            def forward(self, x):
                return self.network(x)
        
        self.model = ProcessNN()
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
    
    def run_monte_carlo_simulation(self, params, n_iterations=1000):
        """Run Monte Carlo simulation with given parameters"""
        results = []
        
        for _ in range(n_iterations):
            # Generate random variations for each parameter
            simulation_params = {
                key: np.random.normal(val['value'], val['std']) 
                for key, val in params.items()
            }
            
            # Convert parameters to tensor
            input_tensor = torch.tensor(
                [list(simulation_params.values())], 
                dtype=torch.float32
            ).to(self.device)
            
            # Get prediction from neural network
            with torch.no_grad():
                output = self.model(input_tensor)
                
            # Convert output to dictionary
            result = {
                'quality': output[0][0].item(),
                'efficiency': output[0][1].item(),
                'time': output[0][2].item(),
                'cost': output[0][3].item(),
                **simulation_params
            }
            results.append(result)
            
        return pd.DataFrame(results)
    
    def optimize_process(self, constraints, n_iterations=100):
        """Optimize process parameters given constraints"""
        best_params = None
        best_score = float('-inf')
        history = []
        
        for i in range(n_iterations):
            # Generate random parameters within constraints
            test_params = {
                param: np.random.uniform(
                    constraints[param]['min'],
                    constraints[param]['max']
                )
                for param in constraints
            }
            
            # Run simulation with these parameters
            input_tensor = torch.tensor(
                [list(test_params.values())],
                dtype=torch.float32
            ).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                
            # Calculate score based on objectives
            score = (
                output[0][0].item() * 0.4 +  # Quality
                output[0][1].item() * 0.3 +  # Efficiency
                (1 - output[0][2].item()) * 0.2 +  # Time (lower is better)
                (1 - output[0][3].item()) * 0.1    # Cost (lower is better)
            )
            
            history.append({
                'iteration': i,
                'score': score,
                **test_params,
                'quality': output[0][0].item(),
                'efficiency': output[0][1].item(),
                'time': output[0][2].item(),
                'cost': output[0][3].item()
            })
            
            if score > best_score:
                best_score = score
                best_params = test_params
        
        return best_params, pd.DataFrame(history)

def render_process_simulation():
    """Render the Process Simulation page"""
    st.title("Process Simulation")
    
    # Initialize simulator if not in session state
    if 'process_simulator' not in st.session_state:
        st.session_state.process_simulator = ProcessSimulator()
    
    # Create tabs for different simulation modes
    tab1, tab2, tab3 = st.tabs([
        "Monte Carlo Simulation",
        "Process Optimization",
        "Real-time Simulation"
    ])
    
    with tab1:
        st.subheader("Monte Carlo Process Simulation")
        
        # Parameter configuration
        st.markdown("### Process Parameters")
        
        params = {}
        col1, col2, col3 = st.columns(3)
        
        with col1:
            params['temperature'] = {
                'value': st.number_input("Temperature (°C)", 100.0, 300.0, 200.0),
                'std': st.number_input("Temperature Std Dev", 0.1, 10.0, 5.0)
            }
            params['pressure'] = {
                'value': st.number_input("Pressure (bar)", 1.0, 50.0, 25.0),
                'std': st.number_input("Pressure Std Dev", 0.1, 5.0, 2.0)
            }
            
        with col2:
            params['flow_rate'] = {
                'value': st.number_input("Flow Rate (L/min)", 10.0, 200.0, 100.0),
                'std': st.number_input("Flow Rate Std Dev", 0.1, 10.0, 3.0)
            }
            params['catalyst_conc'] = {
                'value': st.number_input("Catalyst Concentration (%)", 0.1, 10.0, 5.0),
                'std': st.number_input("Catalyst Std Dev", 0.01, 1.0, 0.2)
            }
            
        with col3:
            params['reaction_time'] = {
                'value': st.number_input("Reaction Time (min)", 1.0, 120.0, 60.0),
                'std': st.number_input("Time Std Dev", 0.1, 10.0, 2.0)
            }
            params['mixing_speed'] = {
                'value': st.number_input("Mixing Speed (rpm)", 100.0, 1000.0, 500.0),
                'std': st.number_input("Speed Std Dev", 1.0, 50.0, 20.0)
            }
        
        n_iterations = st.slider("Number of Iterations", 100, 10000, 1000)
        
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulation..."):
                results = st.session_state.process_simulator.run_monte_carlo_simulation(
                    params, n_iterations
                )
                st.session_state.simulation_results = results
                
                # Display summary statistics
                st.markdown("### Simulation Results")
                summary = results.describe()
                st.dataframe(summary)
                
                # Create visualization grid
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Quality Distribution", "Efficiency Distribution",
                                  "Time Distribution", "Cost Distribution")
                )
                
                # Add histograms
                fig.add_trace(
                    go.Histogram(x=results['quality'], name="Quality"),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Histogram(x=results['efficiency'], name="Efficiency"),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Histogram(x=results['time'], name="Time"),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Histogram(x=results['cost'], name="Cost"),
                    row=2, col=2
                )
                
                fig.update_layout(
                    height=800,
                    showlegend=False,
                    template='plotly_dark',
                    title_text="Simulation Outcome Distributions"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Process Optimization")
        
        # Define optimization constraints
        st.markdown("### Optimization Constraints")
        
        constraints = {}
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Temperature Constraints")
            constraints['temperature'] = {
                'min': st.number_input("Min Temperature", 100.0, 200.0, 150.0),
                'max': st.number_input("Max Temperature", 200.0, 300.0, 250.0)
            }
            
            st.markdown("#### Pressure Constraints")
            constraints['pressure'] = {
                'min': st.number_input("Min Pressure", 1.0, 25.0, 10.0),
                'max': st.number_input("Max Pressure", 25.0, 50.0, 40.0)
            }
            
            st.markdown("#### Flow Rate Constraints")
            constraints['flow_rate'] = {
                'min': st.number_input("Min Flow Rate", 10.0, 100.0, 50.0),
                'max': st.number_input("Max Flow Rate", 100.0, 200.0, 150.0)
            }
            
        with col2:
            st.markdown("#### Catalyst Constraints")
            constraints['catalyst_conc'] = {
                'min': st.number_input("Min Catalyst Conc", 0.1, 5.0, 2.0),
                'max': st.number_input("Max Catalyst Conc", 5.0, 10.0, 8.0)
            }
            
            st.markdown("#### Time Constraints")
            constraints['reaction_time'] = {
                'min': st.number_input("Min Time", 1.0, 60.0, 30.0),
                'max': st.number_input("Max Time", 60.0, 120.0, 90.0)
            }
            
            st.markdown("#### Speed Constraints")
            constraints['mixing_speed'] = {
                'min': st.number_input("Min Speed", 100.0, 500.0, 300.0),
                'max': st.number_input("Max Speed", 500.0, 1000.0, 700.0)
            }
        
        n_iterations = st.slider("Optimization Iterations", 10, 1000, 100)
        
        if st.button("Run Optimization"):
            with st.spinner("Optimizing process parameters..."):
                best_params, history = st.session_state.process_simulator.optimize_process(
                    constraints, n_iterations
                )
                
                # Display optimal parameters
                st.markdown("### Optimal Parameters")
                st.json(best_params)
                
                # Plot optimization history
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=history['iteration'],
                    y=history['score'],
                    mode='lines+markers',
                    name='Optimization Score'
                ))
                
                fig.update_layout(
                    title="Optimization Progress",
                    xaxis_title="Iteration",
                    yaxis_title="Score",
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show parallel coordinates plot for parameter relationships
                fig = px.parallel_coordinates(
                    history,
                    dimensions=['temperature', 'pressure', 'flow_rate',
                              'catalyst_conc', 'reaction_time', 'mixing_speed',
                              'quality', 'efficiency', 'time', 'cost', 'score'],
                    color='score'
                )
                
                fig.update_layout(
                    title="Parameter Relationships",
                    template='plotly_dark',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Real-time Process Simulation")
        
        # Real-time parameter controls
        st.markdown("### Process Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp = st.slider("Temperature Control", 100.0, 300.0, 200.0)
            pressure = st.slider("Pressure Control", 1.0, 50.0, 25.0)
            
        with col2:
            flow = st.slider("Flow Rate Control", 10.0, 200.0, 100.0)
            catalyst = st.slider("Catalyst Control", 0.1, 10.0, 5.0)
            
        with col3:
            time = st.slider("Reaction Time", 1.0, 120.0, 60.0)
            speed = st.slider("Mixing Speed", 100.0, 1000.0, 500.0)
        
        # Real-time simulation
        input_tensor = torch.tensor(
            [[temp, pressure, flow, catalyst, time, speed]],
            dtype=torch.float32
        ).to(st.session_state.process_simulator.device)
        
        with torch.no_grad():
            output = st.session_state.process_simulator.model(input_tensor)
        
        # Display real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Quality Score", f"{output[0][0].item():.2f}")
        with col2:
            st.metric("Process Efficiency", f"{output[0][1].item():.2f}")
        with col3:
            st.metric("Processing Time", f"{output[0][2].item():.2f}")
        with col4:
            st.metric("Operating Cost", f"{output[0][3].item():.2f}")
        
        # Real-time process visualization
        st.markdown("### Process Visualization")
        
        # Create animated flow diagram
        fig = go.Figure()
        
        # Add reactor vessel
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[0, 0, 2, 2, 0],
            mode='lines',
            line=dict(color='rgb(100, 255, 200)', width=2),
            fill='toself',
            name='Reactor'
        ))
        
        # Add dynamic elements based on current parameters
        fig.add_trace(go.Scatter(
            x=[0.2, 0.8],
            y=[0.2 + output[0][1].item(), 0.2 + output[0][1].item()],
            mode='lines',
            line=dict(
                color='rgb(255, 100, 100)',
                width=4
            ),
            name='Temperature'
        ))
        
        fig.update_layout(
            title="Real-time Process Visualization",
            template='plotly_dark',
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_process_simulation() 