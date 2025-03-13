import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class VirtualTestConfig:
    test_id: str
    test_type: str
    parameters: Dict[str, float]
    constraints: Dict[str, Dict[str, float]]
    success_criteria: Dict[str, float]
    duration: int
    sampling_rate: int

class VirtualTestEnvironment:
    def __init__(self):
        self.config_path = Path("configs/virtual_tests")
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.setup_environment()
        
    def setup_environment(self):
        """Initialize the virtual testing environment."""
        # Initialize simulation parameters
        if 'simulation_state' not in st.session_state:
            st.session_state.simulation_state = {
                'running': False,
                'current_step': 0,
                'results': None,
                'history': []
            }
        
        # Initialize test configurations
        if 'test_configs' not in st.session_state:
            st.session_state.test_configs = {}
            
        # Setup ML model for predictions
        self.model = self._initialize_model()
    
    def _initialize_model(self) -> nn.Module:
        """Initialize the ML model for predictions."""
        model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        
        # Use MPS if available (for M1 Macs)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            model = model.to(self.device)
        else:
            self.device = torch.device("cpu")
            
        return model
    
    def run_simulation(self, config: VirtualTestConfig) -> Dict[str, any]:
        """Run a virtual test simulation."""
        # Generate time series data
        n_steps = config.duration * config.sampling_rate
        time = np.linspace(0, config.duration, n_steps)
        
        # Simulate different parameters
        results = {
            'time': time,
            'temperature': self._simulate_temperature(time, config),
            'stress': self._simulate_stress(time, config),
            'strain': self._simulate_strain(time, config),
            'wear': self._simulate_wear(time, config)
        }
        
        # Check success criteria
        success_metrics = self._evaluate_success_criteria(results, config)
        
        return {
            'results': results,
            'metrics': success_metrics
        }
    
    def _simulate_temperature(self, time: np.ndarray, config: VirtualTestConfig) -> np.ndarray:
        """Simulate temperature evolution."""
        base_temp = config.parameters.get('initial_temperature', 20)
        max_temp = config.parameters.get('max_temperature', 150)
        
        # Temperature rises with some oscillation and noise
        temp = base_temp + (max_temp - base_temp) * (1 - np.exp(-time/100))
        temp += 5 * np.sin(time/10)  # Add oscillation
        temp += np.random.normal(0, 1, len(time))  # Add noise
        
        return temp
    
    def _simulate_stress(self, time: np.ndarray, config: VirtualTestConfig) -> np.ndarray:
        """Simulate stress evolution."""
        max_stress = config.parameters.get('max_stress', 1000)
        frequency = config.parameters.get('stress_frequency', 0.1)
        
        # Cyclic stress with increasing amplitude
        stress = max_stress * np.sin(2 * np.pi * frequency * time)
        stress *= (1 + time/max(time) * 0.2)  # Increasing amplitude
        stress += np.random.normal(0, max_stress * 0.05, len(time))  # Add noise
        
        return stress
    
    def _simulate_strain(self, time: np.ndarray, config: VirtualTestConfig) -> np.ndarray:
        """Simulate strain evolution."""
        max_strain = config.parameters.get('max_strain', 0.1)
        
        # Strain follows stress with some hysteresis
        strain = max_strain * (1 - np.exp(-time/200))
        strain += 0.02 * np.sin(2 * np.pi * 0.05 * time)  # Add oscillation
        strain += np.random.normal(0, max_strain * 0.02, len(time))  # Add noise
        
        return strain
    
    def _simulate_wear(self, time: np.ndarray, config: VirtualTestConfig) -> np.ndarray:
        """Simulate wear evolution."""
        wear_rate = config.parameters.get('wear_rate', 0.001)
        
        # Progressive wear with some randomness
        wear = wear_rate * time
        wear += 0.1 * wear_rate * np.random.random(len(time)).cumsum()  # Cumulative randomness
        
        return wear
    
    def _evaluate_success_criteria(self, results: Dict[str, np.ndarray], 
                                 config: VirtualTestConfig) -> Dict[str, any]:
        """Evaluate test results against success criteria."""
        metrics = {}
        
        # Temperature criteria
        max_temp = np.max(results['temperature'])
        metrics['max_temperature'] = {
            'value': max_temp,
            'limit': config.success_criteria.get('max_temperature', 150),
            'passed': max_temp <= config.success_criteria.get('max_temperature', 150)
        }
        
        # Stress criteria
        max_stress = np.max(np.abs(results['stress']))
        metrics['max_stress'] = {
            'value': max_stress,
            'limit': config.success_criteria.get('max_stress', 1000),
            'passed': max_stress <= config.success_criteria.get('max_stress', 1000)
        }
        
        # Strain criteria
        max_strain = np.max(results['strain'])
        metrics['max_strain'] = {
            'value': max_strain,
            'limit': config.success_criteria.get('max_strain', 0.1),
            'passed': max_strain <= config.success_criteria.get('max_strain', 0.1)
        }
        
        # Wear criteria
        final_wear = results['wear'][-1]
        metrics['final_wear'] = {
            'value': final_wear,
            'limit': config.success_criteria.get('max_wear', 0.5),
            'passed': final_wear <= config.success_criteria.get('max_wear', 0.5)
        }
        
        # Overall success
        metrics['overall_success'] = all(m['passed'] for m in metrics.values())
        
        return metrics
    
    def save_config(self, config: VirtualTestConfig):
        """Save test configuration."""
        config_dict = {
            'test_id': config.test_id,
            'test_type': config.test_type,
            'parameters': config.parameters,
            'constraints': config.constraints,
            'success_criteria': config.success_criteria,
            'duration': config.duration,
            'sampling_rate': config.sampling_rate
        }
        
        with open(self.config_path / f"{config.test_id}.json", "w") as f:
            json.dump(config_dict, f, indent=4)
    
    def load_config(self, test_id: str) -> Optional[VirtualTestConfig]:
        """Load test configuration."""
        config_file = self.config_path / f"{test_id}.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config_dict = json.load(f)
                return VirtualTestConfig(**config_dict)
        return None

def render_virtual_testing():
    """Render the Virtual Testing interface."""
    st.title("🔬 Virtual Testing Environment")
    
    # Initialize environment if not in session state
    if 'virtual_env' not in st.session_state:
        st.session_state.virtual_env = VirtualTestEnvironment()
    
    # Initialize simulation state if not in session state
    if 'simulation_state' not in st.session_state:
        st.session_state.simulation_state = {
            'results': None,
            'history': []
        }
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "Test Configuration",
        "Simulation",
        "Results Analysis",
        "Test History"
    ])
    
    with tab1:
        st.subheader("Test Configuration")
        
        # Basic test information
        col1, col2 = st.columns(2)
        with col1:
            test_id = st.text_input("Test ID", "TEST_001")
            test_type = st.selectbox(
                "Test Type",
                ["Fatigue Test", "Wear Test", "Thermal Test", "Load Test"]
            )
        
        with col2:
            duration = st.number_input("Test Duration (s)", 10, 1000, 100)
            sampling_rate = st.number_input("Sampling Rate (Hz)", 1, 100, 10)
        
        # Test parameters
        st.subheader("Test Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_temp = st.number_input("Initial Temperature (°C)", 20.0, 100.0, 25.0)
            max_temp = st.number_input("Maximum Temperature (°C)", 50.0, 200.0, 150.0)
        
        with col2:
            max_stress = st.number_input("Maximum Stress (MPa)", 100.0, 2000.0, 1000.0)
            stress_freq = st.number_input("Stress Frequency (Hz)", 0.1, 10.0, 1.0)
        
        with col3:
            max_strain = st.number_input("Maximum Strain", 0.01, 0.5, 0.1)
            wear_rate = st.number_input("Wear Rate", 0.0001, 0.01, 0.001, format="%.4f")
        
        # Success criteria
        st.subheader("Success Criteria")
        col1, col2 = st.columns(2)
        
        with col1:
            temp_limit = st.number_input("Temperature Limit (°C)", 50.0, 200.0, 150.0)
            stress_limit = st.number_input("Stress Limit (MPa)", 100.0, 2000.0, 1000.0)
        
        with col2:
            strain_limit = st.number_input("Strain Limit", 0.01, 0.5, 0.1)
            wear_limit = st.number_input("Wear Limit", 0.1, 1.0, 0.5)
        
        # Save configuration
        if st.button("Save Configuration", type="primary"):
            config = VirtualTestConfig(
                test_id=test_id,
                test_type=test_type,
                parameters={
                    'initial_temperature': initial_temp,
                    'max_temperature': max_temp,
                    'max_stress': max_stress,
                    'stress_frequency': stress_freq,
                    'max_strain': max_strain,
                    'wear_rate': wear_rate
                },
                constraints={
                    'temperature': {'min': 20, 'max': max_temp},
                    'stress': {'min': -max_stress, 'max': max_stress},
                    'strain': {'min': 0, 'max': max_strain}
                },
                success_criteria={
                    'max_temperature': temp_limit,
                    'max_stress': stress_limit,
                    'max_strain': strain_limit,
                    'max_wear': wear_limit
                },
                duration=duration,
                sampling_rate=sampling_rate
            )
            
            st.session_state.virtual_env.save_config(config)
            st.success("Configuration saved successfully!")
    
    with tab2:
        st.subheader("Test Simulation")
        
        # Load saved configurations
        configs = list(st.session_state.virtual_env.config_path.glob("*.json"))
        if not configs:
            st.warning("No test configurations found. Please create a configuration first.")
        else:
            # Select configuration
            selected_config = st.selectbox(
                "Select Test Configuration",
                [config.stem for config in configs]
            )
            
            # Load configuration
            config = st.session_state.virtual_env.load_config(selected_config)
            if config:
                # Display configuration summary
                st.json(config.__dict__)
                
                # Run simulation
                if st.button("Run Simulation", type="primary"):
                    with st.spinner("Running simulation..."):
                        results = st.session_state.virtual_env.run_simulation(config)
                        st.session_state.simulation_state['results'] = results
                        st.session_state.simulation_state['history'].append({
                            'timestamp': datetime.now().isoformat(),
                            'config': config,
                            'results': results
                        })
                    
                    # Display results summary
                    st.success("Simulation completed!")
                    
                    # Display metrics
                    metrics = results['metrics']
                    cols = st.columns(len(metrics) - 1)  # -1 for overall_success
                    
                    for col, (metric, data) in zip(cols, 
                        [(k,v) for k,v in metrics.items() if k != 'overall_success']):
                        with col:
                            st.metric(
                                metric.replace('_', ' ').title(),
                                f"{data['value']:.3f}",
                                f"Limit: {data['limit']:.3f}",
                                delta_color="normal" if data['passed'] else "inverse"
                            )
                    
                    # Overall success indicator
                    st.markdown(
                        f"### Overall Test Status: "
                        f"{'✅ PASSED' if metrics['overall_success'] else '❌ FAILED'}"
                    )
    
    with tab3:
        st.subheader("Results Analysis")
        
        if st.session_state.simulation_state['results'] is None:
            st.warning("No simulation results available. Please run a simulation first.")
        else:
            results = st.session_state.simulation_state['results']['results']
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Temperature", "Stress", "Strain", "Wear")
            )
            
            # Temperature plot
            fig.add_trace(
                go.Scatter(x=results['time'], y=results['temperature'],
                          name="Temperature"),
                row=1, col=1
            )
            
            # Stress plot
            fig.add_trace(
                go.Scatter(x=results['time'], y=results['stress'],
                          name="Stress"),
                row=1, col=2
            )
            
            # Strain plot
            fig.add_trace(
                go.Scatter(x=results['time'], y=results['strain'],
                          name="Strain"),
                row=2, col=1
            )
            
            # Wear plot
            fig.add_trace(
                go.Scatter(x=results['time'], y=results['wear'],
                          name="Wear"),
                row=2, col=2
            )
            
            fig.update_layout(height=800, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional analysis options
            analysis_type = st.selectbox(
                "Additional Analysis",
                ["Phase Plot", "FFT Analysis", "Statistical Analysis"]
            )
            
            if analysis_type == "Phase Plot":
                # Create phase plot (stress vs. strain)
                fig = px.scatter(
                    x=results['stress'],
                    y=results['strain'],
                    labels={'x': 'Stress', 'y': 'Strain'},
                    title="Stress-Strain Phase Plot"
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_type == "FFT Analysis":
                # Perform FFT analysis
                signal = results['stress']  # Analyze stress signal
                fft = np.fft.fft(signal)
                freq = np.fft.fftfreq(len(signal), d=1/config.sampling_rate)
                
                fig = px.line(
                    x=freq[:len(freq)//2],
                    y=np.abs(fft)[:len(freq)//2],
                    labels={'x': 'Frequency (Hz)', 'y': 'Magnitude'},
                    title="FFT Analysis of Stress Signal"
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_type == "Statistical Analysis":
                # Calculate statistics for each parameter
                stats = pd.DataFrame({
                    'Parameter': ['Temperature', 'Stress', 'Strain', 'Wear'],
                    'Mean': [np.mean(results[p]) for p in ['temperature', 'stress', 'strain', 'wear']],
                    'Std': [np.std(results[p]) for p in ['temperature', 'stress', 'strain', 'wear']],
                    'Min': [np.min(results[p]) for p in ['temperature', 'stress', 'strain', 'wear']],
                    'Max': [np.max(results[p]) for p in ['temperature', 'stress', 'strain', 'wear']]
                })
                
                st.dataframe(stats)
    
    with tab4:
        st.subheader("Test History")
        
        if not st.session_state.simulation_state['history']:
            st.warning("No test history available.")
        else:
            # Display test history
            for i, entry in enumerate(reversed(st.session_state.simulation_state['history'])):
                with st.expander(f"Test {entry['config'].test_id} - "
                               f"{entry['timestamp']}"):
                    st.json(entry['config'].__dict__)
                    
                    # Display results summary
                    metrics = entry['results']['metrics']
                    cols = st.columns(len(metrics) - 1)
                    
                    for col, (metric, data) in zip(cols, 
                        [(k,v) for k,v in metrics.items() if k != 'overall_success']):
                        with col:
                            st.metric(
                                metric.replace('_', ' ').title(),
                                f"{data['value']:.3f}",
                                f"Limit: {data['limit']:.3f}",
                                delta_color="normal" if data['passed'] else "inverse"
                            )

if __name__ == "__main__":
    render_virtual_testing() 