import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import json

class WhatIfAnalyzer:
    def __init__(self):
        self.setup_simulation_engine()
        self.baseline_scenario = self._generate_baseline_scenario()
        
    def _generate_baseline_scenario(self) -> Dict[str, float]:
        """Generate default baseline scenario parameters"""
        return {
            'cutting_speed': 200,      # m/min
            'feed_rate': 0.2,          # mm/rev
            'depth_of_cut': 2.0,       # mm
            'material_hardness': 40,    # HRC
            'coolant_flow': 20,        # L/min
            'tool_coating': 1.0,       # coating factor
            'machining_strategy': 1.0   # strategy factor
        }
    
    def setup_simulation_engine(self):
        """Initialize the simulation engine"""
        class ProcessSimulator:
            def __init__(self):
                self.quality_model = self._create_quality_model()
                self.cycle_time_model = self._create_cycle_time_model()
                self.cost_model = self._create_cost_model()
                self.tool_life_model = self._create_tool_life_model()
            
            def _create_quality_model(self):
                """Neural network for quality prediction"""
                import torch.nn as nn
                
                class QualityNet(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.network = nn.Sequential(
                            nn.Linear(7, 32),
                            nn.ReLU(),
                            nn.Linear(32, 16),
                            nn.ReLU(),
                            nn.Linear(16, 1),
                            nn.Sigmoid()
                        )
                    
                    def forward(self, x):
                        return self.network(x)
                
                return QualityNet()
            
            def _create_cycle_time_model(self):
                """Function for cycle time calculation"""
                def cycle_time_func(params):
                    cutting_speed = params['cutting_speed']
                    feed_rate = params['feed_rate']
                    depth_of_cut = params['depth_of_cut']
                    
                    # Basic cycle time calculation
                    machining_time = (100 / cutting_speed) * (1 / feed_rate)
                    setup_time = 5 * params['machining_strategy']
                    
                    return machining_time + setup_time
                
                return cycle_time_func
            
            def _create_cost_model(self):
                """Function for cost calculation"""
                def cost_func(params):
                    # Cost factors
                    machine_cost = 100  # $/hour
                    tool_cost = 50      # $/tool
                    coolant_cost = 2    # $/liter
                    
                    cycle_time = self.cycle_time_model(params)
                    machine_cost_per_part = (machine_cost / 3600) * cycle_time
                    tool_cost_per_part = tool_cost / self.tool_life_model(params)
                    coolant_cost_per_part = coolant_cost * params['coolant_flow'] * (cycle_time / 3600)
                    
                    return machine_cost_per_part + tool_cost_per_part + coolant_cost_per_part
                
                return cost_func
            
            def _create_tool_life_model(self):
                """Function for tool life prediction"""
                def tool_life_func(params):
                    # Taylor's tool life equation with modifications
                    v = params['cutting_speed']
                    f = params['feed_rate']
                    d = params['depth_of_cut']
                    h = params['material_hardness']
                    c = params['tool_coating']
                    
                    base_life = 1000 * (v ** -0.3) * (f ** -0.15) * (d ** -0.1)
                    hardness_factor = np.exp(-0.02 * h)
                    coating_factor = c
                    
                    return base_life * hardness_factor * coating_factor
                
                return tool_life_func
            
            def simulate(self, params: Dict[str, float]) -> Dict[str, float]:
                """Run simulation with given parameters"""
                quality_score = np.random.normal(0.9, 0.05)  # Placeholder for neural network
                cycle_time = self.cycle_time_model(params)
                production_cost = self.cost_model(params)
                tool_life = self.tool_life_model(params)
                
                return {
                    'quality_score': quality_score,
                    'cycle_time': cycle_time,
                    'production_cost': production_cost,
                    'tool_life': tool_life
                }
        
        self.simulator = ProcessSimulator()
    
    def simulate_scenario(self, params: Dict[str, float]) -> Dict[str, float]:
        """Simulate a scenario with given parameters"""
        return self.simulator.simulate(params)
    
    def run_sensitivity_analysis(self, base_params: Dict[str, float], 
                               param_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """Run sensitivity analysis on parameters"""
        results = []
        
        for param, (min_val, max_val) in param_ranges.items():
            values = np.linspace(min_val, max_val, 10)
            for value in values:
                test_params = base_params.copy()
                test_params[param] = value
                
                simulation_result = self.simulate_scenario(test_params)
                results.append({
                    'parameter': param,
                    'value': value,
                    **simulation_result
                })
        
        return pd.DataFrame(results)
    
    def calculate_impact_scores(self, scenario_results: Dict[str, float],
                              baseline_results: Dict[str, float]) -> Dict[str, float]:
        """Calculate impact scores compared to baseline"""
        impact_scores = {}
        
        for metric in ['quality_score', 'cycle_time', 'production_cost', 'tool_life']:
            baseline = baseline_results[metric]
            scenario = scenario_results[metric]
            
            # Calculate relative change
            if baseline != 0:
                relative_change = (scenario - baseline) / baseline
            else:
                relative_change = scenario
            
            # Convert to impact score (-1 to 1 scale)
            impact_scores[metric] = np.tanh(relative_change)
        
        return impact_scores

def render_what_if_analysis():
    """Render the What-If Analysis interface"""
    st.title("🔮 What-If Analysis")
    
    # Initialize analyzer if not in session state
    if 'what_if_analyzer' not in st.session_state:
        st.session_state.what_if_analyzer = WhatIfAnalyzer()
    
    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs([
        "Scenario Analysis",
        "Sensitivity Analysis",
        "Impact Assessment"
    ])
    
    with tab1:
        st.subheader("Scenario Analysis")
        
        # Parameter input
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_params = {
                'cutting_speed': st.slider(
                    "Cutting Speed (m/min)",
                    100, 300,
                    st.session_state.what_if_analyzer.baseline_scenario['cutting_speed']
                ),
                'feed_rate': st.slider(
                    "Feed Rate (mm/rev)",
                    0.1, 0.5,
                    st.session_state.what_if_analyzer.baseline_scenario['feed_rate']
                ),
                'depth_of_cut': st.slider(
                    "Depth of Cut (mm)",
                    0.5, 5.0,
                    st.session_state.what_if_analyzer.baseline_scenario['depth_of_cut']
                ),
                'material_hardness': st.slider(
                    "Material Hardness (HRC)",
                    20, 60,
                    st.session_state.what_if_analyzer.baseline_scenario['material_hardness']
                )
            }
        
        with col2:
            scenario_params.update({
                'coolant_flow': st.slider(
                    "Coolant Flow (L/min)",
                    10, 30,
                    st.session_state.what_if_analyzer.baseline_scenario['coolant_flow']
                ),
                'tool_coating': st.slider(
                    "Tool Coating Factor",
                    0.5, 1.5,
                    st.session_state.what_if_analyzer.baseline_scenario['tool_coating']
                ),
                'machining_strategy': st.slider(
                    "Machining Strategy Factor",
                    0.5, 1.5,
                    st.session_state.what_if_analyzer.baseline_scenario['machining_strategy']
                )
            })
        
        # Run simulation
        if st.button("Run Scenario"):
            # Simulate scenario
            scenario_results = st.session_state.what_if_analyzer.simulate_scenario(
                scenario_params
            )
            
            # Simulate baseline
            baseline_results = st.session_state.what_if_analyzer.simulate_scenario(
                st.session_state.what_if_analyzer.baseline_scenario
            )
            
            # Calculate impact scores
            impact_scores = st.session_state.what_if_analyzer.calculate_impact_scores(
                scenario_results, baseline_results
            )
            
            # Display results
            st.markdown("### Simulation Results")
            
            # Create comparison table
            results_df = pd.DataFrame({
                'Metric': ['Quality Score', 'Cycle Time', 'Production Cost', 'Tool Life'],
                'Baseline': [
                    f"{baseline_results['quality_score']:.2%}",
                    f"{baseline_results['cycle_time']:.1f} min",
                    f"${baseline_results['production_cost']:.2f}",
                    f"{baseline_results['tool_life']:.1f} hours"
                ],
                'Scenario': [
                    f"{scenario_results['quality_score']:.2%}",
                    f"{scenario_results['cycle_time']:.1f} min",
                    f"${scenario_results['production_cost']:.2f}",
                    f"{scenario_results['tool_life']:.1f} hours"
                ],
                'Impact': [
                    f"{impact_scores['quality_score']:+.1%}",
                    f"{impact_scores['cycle_time']:+.1%}",
                    f"{impact_scores['production_cost']:+.1%}",
                    f"{impact_scores['tool_life']:+.1%}"
                ]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=[100, 100, 100, 100],
                theta=['Quality', 'Cycle Time', 'Cost', 'Tool Life'],
                fill='toself',
                name='Baseline',
                line_color='rgba(100, 255, 200, 0.8)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=[
                    scenario_results['quality_score'] / baseline_results['quality_score'] * 100,
                    scenario_results['cycle_time'] / baseline_results['cycle_time'] * 100,
                    scenario_results['production_cost'] / baseline_results['production_cost'] * 100,
                    scenario_results['tool_life'] / baseline_results['tool_life'] * 100
                ],
                theta=['Quality', 'Cycle Time', 'Cost', 'Tool Life'],
                fill='toself',
                name='Scenario',
                line_color='rgba(255, 100, 100, 0.8)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 150]
                    )
                ),
                showlegend=True,
                title="Scenario Comparison",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Sensitivity Analysis")
        
        # Parameter selection
        selected_params = st.multiselect(
            "Select Parameters for Analysis",
            list(st.session_state.what_if_analyzer.baseline_scenario.keys()),
            default=['cutting_speed', 'feed_rate']
        )
        
        if selected_params:
            # Define parameter ranges
            param_ranges = {
                param: (
                    st.session_state.what_if_analyzer.baseline_scenario[param] * 0.5,
                    st.session_state.what_if_analyzer.baseline_scenario[param] * 1.5
                )
                for param in selected_params
            }
            
            # Run sensitivity analysis
            sensitivity_results = st.session_state.what_if_analyzer.run_sensitivity_analysis(
                st.session_state.what_if_analyzer.baseline_scenario,
                param_ranges
            )
            
            # Plot sensitivity results
            st.markdown("### Parameter Sensitivity")
            
            # Create multi-line plot
            fig = go.Figure()
            
            for param in selected_params:
                param_data = sensitivity_results[sensitivity_results['parameter'] == param]
                
                fig.add_trace(go.Scatter(
                    x=param_data['value'],
                    y=param_data['quality_score'],
                    name=f"{param} (Quality)",
                    line=dict(dash='solid')
                ))
                
                fig.add_trace(go.Scatter(
                    x=param_data['value'],
                    y=param_data['tool_life'],
                    name=f"{param} (Tool Life)",
                    line=dict(dash='dot')
                ))
            
            fig.update_layout(
                title="Parameter Sensitivity Analysis",
                xaxis_title="Parameter Value (Normalized)",
                yaxis_title="Output Value",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display sensitivity metrics
            st.markdown("### Sensitivity Metrics")
            
            sensitivity_metrics = {}
            for param in selected_params:
                param_data = sensitivity_results[sensitivity_results['parameter'] == param]
                
                sensitivity_metrics[param] = {
                    'quality_range': param_data['quality_score'].max() - param_data['quality_score'].min(),
                    'tool_life_range': param_data['tool_life'].max() - param_data['tool_life'].min(),
                    'cost_range': param_data['production_cost'].max() - param_data['production_cost'].min()
                }
            
            metrics_df = pd.DataFrame(sensitivity_metrics).T
            metrics_df.columns = ['Quality Impact', 'Tool Life Impact', 'Cost Impact']
            
            st.dataframe(metrics_df, use_container_width=True)
    
    with tab3:
        st.subheader("Impact Assessment")
        
        # Generate multiple scenarios
        n_scenarios = 100
        scenario_results = []
        
        for _ in range(n_scenarios):
            params = {
                key: np.random.normal(
                    st.session_state.what_if_analyzer.baseline_scenario[key],
                    st.session_state.what_if_analyzer.baseline_scenario[key] * 0.1
                )
                for key in st.session_state.what_if_analyzer.baseline_scenario.keys()
            }
            
            results = st.session_state.what_if_analyzer.simulate_scenario(params)
            scenario_results.append({**params, **results})
        
        scenarios_df = pd.DataFrame(scenario_results)
        
        # Create correlation heatmap
        corr_matrix = scenarios_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Parameter-Output Correlations",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key findings
        st.markdown("### Key Findings")
        
        # Calculate impact rankings
        impact_rankings = {}
        output_vars = ['quality_score', 'cycle_time', 'production_cost', 'tool_life']
        input_vars = list(st.session_state.what_if_analyzer.baseline_scenario.keys())
        
        for output in output_vars:
            correlations = abs(corr_matrix[output][input_vars])
            impact_rankings[output] = correlations.sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Quality Factors")
            for param, corr in impact_rankings['quality_score'].head(3).items():
                st.metric(param, f"{corr:.3f}")
        
        with col2:
            st.markdown("#### Top Cost Factors")
            for param, corr in impact_rankings['production_cost'].head(3).items():
                st.metric(param, f"{corr:.3f}")

if __name__ == "__main__":
    render_what_if_analysis() 