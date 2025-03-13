import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import random
import pandas as pd  # Add pandas import for report generation

class AdvancedInteractiveHeader:
    """Interactive header component with real-time data and actions."""
    
    def __init__(self):
        """Initialize the interactive header component."""
        self.animation_frame = 0
        self.last_update = datetime.now()
        self.data_buffer = []
        self.max_buffer_size = 100
        self.system_status = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'network_load': 0,
            'active_sensors': 0
        }
        
        # Advanced metrics
        self.advanced_metrics = {
            'efficiency': 87.5,
            'quality_score': 94.2,
            'throughput': 782,
            'utilization': 78.9
        }
        
        # Initialize status
        self.update_system_status()
        self.update_advanced_metrics()
        
        # Add custom CSS
        self._add_custom_css()
    
    def _add_custom_css(self):
        """Add custom CSS for metrics and alerts."""
        st.markdown("""
        <style>
        /* Metric styles */
        .metric-container {
            padding: 10px;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.1);
            margin-bottom: 10px;
        }
        .metric-label {
            font-size: 0.8em;
            color: #888;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-text {
            font-size: 0.8em;
            color: #888;
        }
        .metric-chart {
            width: 100%;
            height: 6px;
            background: rgba(0, 0, 0, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }
        .metric-bar {
            height: 100%;
            transition: width 0.5s ease;
        }
        
        /* Alert styles */
        .alert-item {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            margin-bottom: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def update_system_status(self):
        """Update the system status with real-time data"""
        if 'system_status' not in st.session_state:
            # Initialize with starting values if not present
            st.session_state.system_status = {
                'cpu_usage': random.uniform(25, 45),  # Start with moderate CPU usage
                'memory_usage': random.uniform(35, 55),  # Start with moderate memory usage
                'network_load': random.uniform(15, 35),  # Start with moderate network load
                'active_sensors': random.randint(8, 12)  # Start with reasonable number of sensors
            }
            
            # Initialize historical data if needed
            if 'historical_data' not in st.session_state:
                st.session_state.historical_data = {
                    'cpu': [],
                    'memory': [],
                    'network': [],
                    'sensors': [],
                    'temperature': [],  # New temperature metric
                    'response_time': [], # New response time metric
                    'error_rate': [],   # New error rate metric
                    'throughput': [],   # New throughput metric
                    'power': []         # New power consumption metric
                }
        
        # Retrieve current status
        self.system_status = st.session_state.system_status
        
        # Introduce some randomness for realism (values drift over time)
        self.system_status['cpu_usage'] = max(0, min(100, self.system_status['cpu_usage'] + random.uniform(-3, 5)))
        self.system_status['memory_usage'] = max(0, min(100, self.system_status['memory_usage'] + random.uniform(-2, 3)))  
        self.system_status['network_load'] = max(0, min(100, self.system_status['network_load'] + random.uniform(-4, 6)))
        
        # Occasionally change the number of active sensors
        if random.random() < 0.2:  # 20% chance to change sensor count
            change = random.choice([-1, 0, 1])
            self.system_status['active_sensors'] = max(0, self.system_status['active_sensors'] + change)
        
        # Update stored status
        st.session_state.system_status = self.system_status
    
    def generate_live_data(self):
        """Generate new live data points for charts"""
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = {
                'cpu': [],
                'memory': [],
                'network': [],
                'sensors': [],
                'temperature': [],  # New temperature metric
                'response_time': [], # New response time metric
                'error_rate': [],   # New error rate metric
                'throughput': [],   # New throughput metric
                'power': []         # New power consumption metric
            }
        
        # Get current time
        now = datetime.now()
        
        # Add current metrics to history
        st.session_state.historical_data['cpu'].append((now, self.system_status['cpu_usage']))
        st.session_state.historical_data['memory'].append((now, self.system_status['memory_usage']))
        st.session_state.historical_data['network'].append((now, self.system_status['network_load']))
        st.session_state.historical_data['sensors'].append((now, self.system_status['active_sensors']))
        
        # Add new metrics with simulated values
        # Temperature between 40-60°C with some random variation
        temperature = 50 + random.uniform(-10, 10)
        st.session_state.historical_data['temperature'].append((now, temperature))
        
        # Response time between 50-150ms with some random variation
        response_time = 100 + random.uniform(-50, 50)
        st.session_state.historical_data['response_time'].append((now, response_time))
        
        # Error rate between 0-5% with some random variation
        error_rate = random.uniform(0, 5)
        st.session_state.historical_data['error_rate'].append((now, error_rate))
        
        # Throughput between 700-900 units with some random variation
        throughput = 800 + random.uniform(-100, 100)
        st.session_state.historical_data['throughput'].append((now, throughput))
        
        # Power consumption between 400-600W with some random variation
        power = 500 + random.uniform(-100, 100)
        st.session_state.historical_data['power'].append((now, power))
        
        # Limit history length to keep memory usage reasonable
        max_history = 100
        for key in st.session_state.historical_data:
            if len(st.session_state.historical_data[key]) > max_history:
                st.session_state.historical_data[key] = st.session_state.historical_data[key][-max_history:]
        
        self.last_update = now
        self.update_system_status()
        return self.system_status

    def render_system_metrics(self):
        """Render system metrics with proper styling and actual values"""
        # Get metric values
        cpu = self.system_status['cpu_usage']
        memory = self.system_status['memory_usage']
        network = self.system_status['network_load']
        sensors = self.system_status['active_sensors']
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Determine status colors based on thresholds
        def get_status_color(value, warning=70, critical=90):
            if value >= critical:
                return "red"
            elif value >= warning:
                return "orange"
            else:
                return "green"
        
        # Format and display CPU usage
        cpu_color = get_status_color(cpu)
        col1.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">CPU Usage</div>
            <div class="metric-value" style="color: {cpu_color};">{cpu:.1f}%</div>
            <div class="metric-chart">
                <div class="metric-bar" style="width: {cpu}%; background-color: {cpu_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Format and display Memory usage
        memory_color = get_status_color(memory)
        col2.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Memory</div>
            <div class="metric-value" style="color: {memory_color};">{memory:.1f}%</div>
            <div class="metric-chart">
                <div class="metric-bar" style="width: {memory}%; background-color: {memory_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Format and display Network load
        network_color = get_status_color(network)
        col3.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Network</div>
            <div class="metric-value" style="color: {network_color};">{network:.1f}%</div>
            <div class="metric-chart">
                <div class="metric-bar" style="width: {network}%; background-color: {network_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Format and display Active Sensors
        sensor_status = "green" if sensors > 0 else "red"
        col4.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Active Sensors</div>
            <div class="metric-value" style="color: {sensor_status};">{sensors}</div>
            <div class="metric-text">of 12 sensors online</div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_system_alerts(self):
        """Render system alerts section with actual alerts"""
        # Check for alert conditions
        alerts = []
        
        if self.system_status['cpu_usage'] > 80:
            alerts.append({
                "severity": "critical",
                "message": "CPU usage critically high",
                "time": datetime.now().strftime("%H:%M:%S")
            })
        elif self.system_status['cpu_usage'] > 70:
            alerts.append({
                "severity": "warning",
                "message": "CPU usage elevated",
                "time": datetime.now().strftime("%H:%M:%S")
            })
            
        if self.system_status['memory_usage'] > 80:
            alerts.append({
                "severity": "critical",
                "message": "Memory usage critically high",
                "time": datetime.now().strftime("%H:%M:%S")
            })
        elif self.system_status['memory_usage'] > 70:
            alerts.append({
                "severity": "warning",
                "message": "Memory usage elevated",
                "time": datetime.now().strftime("%H:%M:%S")
            })
            
        if self.system_status['network_load'] > 80:
            alerts.append({
                "severity": "critical",
                "message": "Network load critically high",
                "time": datetime.now().strftime("%H:%M:%S")
            })
        
        if self.system_status['active_sensors'] < 5:
            alerts.append({
                "severity": "warning", 
                "message": f"Only {self.system_status['active_sensors']} sensors active",
                "time": datetime.now().strftime("%H:%M:%S")
            })
            
        # Store alerts in session state
        if 'system_alerts' not in st.session_state:
            st.session_state.system_alerts = []
            
        # Add new alerts to the list
        for alert in alerts:
            if alert not in st.session_state.system_alerts:
                st.session_state.system_alerts.insert(0, alert)
                
        # Limit the number of stored alerts
        if len(st.session_state.system_alerts) > 10:
            st.session_state.system_alerts = st.session_state.system_alerts[:10]
            
        # Display alerts
        with st.expander("System Alerts", expanded=len(alerts) > 0):
            if not st.session_state.system_alerts:
                st.info("No system alerts")
            else:
                for alert in st.session_state.system_alerts:
                    severity_color = "red" if alert["severity"] == "critical" else "orange"
                    severity_icon = "🔴" if alert["severity"] == "critical" else "🟠"
                    
                    st.markdown(f"""
                    <div class="alert-item" style="margin-bottom: 8px; padding: 8px; border-left: 3px solid {severity_color};">
                        <div style="display: flex; justify-content: space-between;">
                            <span>{severity_icon} {alert["message"]}</span>
                            <span style="color: gray; font-size: 0.8em;">{alert["time"]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("Clear Alerts", key="clear_alerts"):
                    st.session_state.system_alerts = []
                    st.rerun()

    def render_3d_visualization(self):
        """Enhanced 3D visualization with dynamic surface"""
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        
        # Create dynamic surface with time-varying components
        time_factor = self.animation_frame / 20
        Z = (np.sin(np.sqrt(X**2 + Y**2) + time_factor) / 
             (np.sqrt(X**2 + Y**2) + 1) + 
             0.3 * np.sin(X + time_factor) * np.cos(Y + time_factor))
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            showscale=False,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                fresnel=0.2,
                specular=1.0,
                roughness=0.5
            )
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, showline=False),
                yaxis=dict(showgrid=False, showticklabels=False, showline=False),
                zaxis=dict(showgrid=False, showticklabels=False, showline=False),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5 * np.cos(time_factor/5), 
                            y=1.5 * np.sin(time_factor/5), 
                            z=1.5)
                ),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
        )
        
        return fig

    def render_status_indicators(self):
        """Render status indicators with animations"""
        latest_data = self.data_buffer[-1] if self.data_buffer else {'status': 'optimal'}
        status_color = {
            'optimal': 'rgba(100,255,100,0.8)',
            'warning': 'rgba(255,200,100,0.8)',
            'critical': 'rgba(255,100,100,0.8)'
        }[latest_data['status']]
        
        st.markdown(f"""
        <div class="status-indicator" style="background: {status_color}">
            <span class="status-text">{latest_data['status'].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    def render(self):
        """Render the interactive header with all components"""
        # Container for the header
        with st.container():
            # Debug info to verify methods exist
            print("Debug: Rendering header")
            
            # List available methods for debugging
            methods = [method for method in dir(self) if callable(getattr(self, method)) and not method.startswith("_")]
            print(f"Available methods: {methods}")
            
            # Auto-refresh data every 3 seconds
            self._schedule_refresh()
            
            # Layout with 3 columns for top row
            col1, col2, col3 = st.columns([1, 2, 1])
            
            # Left column - Quick actions
            with col1:
                print("Calling render_quick_actions...")
                try:
                    self.render_quick_actions()
                    print("render_quick_actions completed")
                except Exception as e:
                    st.error(f"Error in render_quick_actions: {str(e)}")
                    print(f"Error in render_quick_actions: {str(e)}")
            
            # Middle column - System metrics
            with col2:
                try:
                    self.render_system_metrics()
                except Exception as e:
                    st.error(f"Error in render_system_metrics: {str(e)}")
                    print(f"Error in render_system_metrics: {str(e)}")
            
            # Right column - System alerts
            with col3:
                try:
                    self.render_system_alerts()
                except Exception as e:
                    st.error(f"Error in render_system_alerts: {str(e)}")
                    print(f"Error in render_system_alerts: {str(e)}")
            
            # Add a small spacing
            st.markdown("---")
            
            # Second row - Advanced metrics with real-time monitoring
            try:
                self.render_advanced_metrics()
            except Exception as e:
                st.error(f"Error in render_advanced_metrics: {str(e)}")
                print(f"Error in render_advanced_metrics: {str(e)}")
            
            # Update data on each render
            try:
                self.update_advanced_metrics()
            except Exception as e:
                st.error(f"Error in update_advanced_metrics: {str(e)}")
                print(f"Error in update_advanced_metrics: {str(e)}")
    
    def _schedule_refresh(self):
        """Schedule automatic data refresh"""
        # Initialize last refresh time if not exists
        if 'last_header_refresh' not in st.session_state:
            st.session_state.last_header_refresh = datetime.now() - timedelta(seconds=10)
            
        # Check if it's time to refresh (every 3 seconds)
        now = datetime.now()
        if (now - st.session_state.last_header_refresh).total_seconds() >= 3:
            # Update data
            self.generate_live_data()
            
            # Store last refresh time
            st.session_state.last_header_refresh = now

    def render_right_screen(self):
        """Render real-time signal visualization"""
        if not self.data_buffer:
            return go.Figure()

        # Extract data for plotting
        timestamps = [d['timestamp'] for d in self.data_buffer]
        signal1 = [d['signal1'] for d in self.data_buffer]
        signal2 = [d['signal2'] for d in self.data_buffer]
        signal3 = [d['signal3'] for d in self.data_buffer]

        fig = go.Figure()
        
        # Add signal traces
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=signal1,
            name='Signal 1',
            line=dict(color='rgba(0,255,255,0.8)', width=2),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=signal2,
            name='Signal 2',
            line=dict(color='rgba(255,165,0,0.8)', width=2),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=signal3,
            name='Signal 3',
            line=dict(color='rgba(255,105,180,0.8)', width=2),
            fill='tonexty'
        ))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                showline=False,
                showticklabels=True,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                showline=False,
                showticklabels=True,
            )
        )
        
        return fig

    def render_performance_indicators(self):
        """Render key performance indicators"""
        kpis = self._calculate_kpis()
        
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-grid">
        """, unsafe_allow_html=True)

        for kpi in kpis:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{kpi['icon']}</div>
                <div class="kpi-details">
                    <div class="kpi-value">{kpi['value']}</div>
                    <div class="kpi-label">{kpi['label']}</div>
                </div>
                <div class="kpi-trend {kpi['trend_class']}">
                    {kpi['trend_icon']} {kpi['trend_value']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    def _calculate_kpis(self) -> List[Dict]:
        """Calculate key performance indicators"""
        return [
            {
                'icon': '🎯',
                'value': '98.5%',
                'label': 'Accuracy',
                'trend_icon': '↗️',
                'trend_value': '+2.3%',
                'trend_class': 'trend-up'
            },
            {
                'icon': '⚡',
                'value': '123ms',
                'label': 'Response Time',
                'trend_icon': '↘️',
                'trend_value': '-5ms',
                'trend_class': 'trend-down'
            },
            {
                'icon': '🔄',
                'value': '99.9%',
                'label': 'Uptime',
                'trend_icon': '→',
                'trend_value': '0%',
                'trend_class': 'trend-stable'
            },
            {
                'icon': '📈',
                'value': '456',
                'label': 'Active Users',
                'trend_icon': '↗️',
                'trend_value': '+12%',
                'trend_class': 'trend-up'
            }
        ]

    def render_system_health(self):
        """Render system health status with radar chart"""
        health_metrics = {
            'Performance': 85,
            'Reliability': 92,
            'Security': 88,
            'Efficiency': 78,
            'Scalability': 82
        }

        fig = go.Figure()

        # Add radar chart
        fig.add_trace(go.Scatterpolar(
            r=list(health_metrics.values()),
            theta=list(health_metrics.keys()),
            fill='toself',
            line=dict(color='rgba(0,255,255,0.8)', width=2),
            fillcolor='rgba(0,255,255,0.2)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=False,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )

        return fig

    def update_advanced_metrics(self):
        """Update advanced metrics with realistic data"""
        if 'advanced_metrics' not in st.session_state:
            st.session_state.advanced_metrics = self.advanced_metrics.copy()
        
        # Retrieve current metrics
        self.advanced_metrics = st.session_state.advanced_metrics
        
        # Add some natural variation with slight upward bias (reflecting improvements)
        self.advanced_metrics['efficiency'] = max(50, min(100, self.advanced_metrics['efficiency'] + random.uniform(-1, 1.2)))
        self.advanced_metrics['quality_score'] = max(70, min(100, self.advanced_metrics['quality_score'] + random.uniform(-0.8, 1.0)))
        self.advanced_metrics['throughput'] = max(500, min(1000, self.advanced_metrics['throughput'] + random.uniform(-15, 20)))
        self.advanced_metrics['utilization'] = max(50, min(100, self.advanced_metrics['utilization'] + random.uniform(-1.5, 1.8)))
        
        # Update stored metrics
        st.session_state.advanced_metrics = self.advanced_metrics

    def render_advanced_metrics(self):
        """Render advanced metrics with interactive elements"""
        st.markdown("<h3>Advanced Metrics</h3>", unsafe_allow_html=True)
        
        # Create a grid of metrics
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        # Efficiency
        efficiency = self.advanced_metrics['efficiency']
        efficiency_color = "green" if efficiency > 85 else "orange" if efficiency > 70 else "red"
        col1.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Efficiency</div>
            <div class="metric-value" style="color: {efficiency_color};">{efficiency:.1f}%</div>
            <div class="metric-chart">
                <div class="metric-bar" style="width: {efficiency}%; background-color: {efficiency_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quality Score
        quality = self.advanced_metrics['quality_score']
        quality_color = "green" if quality > 90 else "orange" if quality > 80 else "red"
        col2.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Quality Score</div>
            <div class="metric-value" style="color: {quality_color};">{quality:.1f}</div>
            <div class="metric-chart">
                <div class="metric-bar" style="width: {quality}%; background-color: {quality_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Throughput
        throughput = self.advanced_metrics['throughput']
        throughput_percent = (throughput - 500) / (1000 - 500) * 100  # Scale to percentage
        throughput_color = "green" if throughput > 750 else "orange" if throughput > 650 else "red"
        col3.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Throughput</div>
            <div class="metric-value" style="color: {throughput_color};">{throughput:.0f} units</div>
            <div class="metric-chart">
                <div class="metric-bar" style="width: {throughput_percent}%; background-color: {throughput_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Utilization
        utilization = self.advanced_metrics['utilization']
        utilization_color = "green" if 70 <= utilization <= 90 else "orange" if utilization > 90 else "red"
        col4.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Utilization</div>
            <div class="metric-value" style="color: {utilization_color};">{utilization:.1f}%</div>
            <div class="metric-chart">
                <div class="metric-bar" style="width: {utilization}%; background-color: {utilization_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add new section for real-time monitoring visualization
        st.markdown("## Real-Time Monitoring", unsafe_allow_html=True)
        
        # Create tabs for different monitoring categories
        tabs = st.tabs(["System Performance", "Resource Usage", "Temperature", "Response Time", "Errors"])
        
        with tabs[0]:
            self._render_system_performance_chart()
            
        with tabs[1]:
            self._render_resource_usage_chart()
            
        with tabs[2]:
            self._render_temperature_chart()
            
        with tabs[3]:
            self._render_response_time_chart()
            
        with tabs[4]:
            self._render_error_rate_chart()
            
        # Add more indicators as KPI cards
        st.markdown("### Key Performance Indicators", unsafe_allow_html=True)
        kpi_cols = st.columns(4)
        
        # Add KPI for Uptime
        uptime_hours = random.randint(240, 720)  # Random uptime between 10-30 days
        uptime_days = uptime_hours // 24
        remaining_hours = uptime_hours % 24
        kpi_cols[0].markdown(f"""
        <div style="background-color: rgba(0,100,0,0.1); padding: 10px; border-radius: 5px; text-align: center;">
            <div style="font-size: 1.5rem; font-weight: bold; color: green;">⏱️ {uptime_days}d {remaining_hours}h</div>
            <div style="font-size: 0.8rem; color: gray;">System Uptime</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add KPI for Response Time
        avg_response = random.randint(80, 120)
        response_color = "green" if avg_response < 100 else "orange" if avg_response < 150 else "red"
        kpi_cols[1].markdown(f"""
        <div style="background-color: rgba(0,100,0,0.1); padding: 10px; border-radius: 5px; text-align: center;">
            <div style="font-size: 1.5rem; font-weight: bold; color: {response_color};">⚡ {avg_response} ms</div>
            <div style="font-size: 0.8rem; color: gray;">Avg Response Time</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add KPI for Throughput
        daily_throughput = random.randint(12000, 20000)
        kpi_cols[2].markdown(f"""
        <div style="background-color: rgba(0,100,0,0.1); padding: 10px; border-radius: 5px; text-align: center;">
            <div style="font-size: 1.5rem; font-weight: bold; color: green;">🔄 {daily_throughput:,}</div>
            <div style="font-size: 0.8rem; color: gray;">Daily Throughput</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add KPI for System Health
        health_score = random.randint(85, 99)
        health_color = "green" if health_score > 90 else "orange" if health_score > 80 else "red"
        kpi_cols[3].markdown(f"""
        <div style="background-color: rgba(0,100,0,0.1); padding: 10px; border-radius: 5px; text-align: center;">
            <div style="font-size: 1.5rem; font-weight: bold; color: {health_color};">❤️ {health_score}%</div>
            <div style="font-size: 0.8rem; color: gray;">System Health</div>
        </div>
        """, unsafe_allow_html=True)

    def _render_system_performance_chart(self):
        """Render system performance chart using historical data"""
        if 'historical_data' not in st.session_state or not st.session_state.historical_data['cpu']:
            st.info("Collecting data... Please wait.")
            return
            
        # Create dataframe from historical data
        cpu_data = st.session_state.historical_data['cpu'][-30:]  # Last 30 data points
        
        # Format the data for plotting
        times = [t.strftime('%H:%M:%S') for t, _ in cpu_data]
        values = [v for _, v in cpu_data]
        
        # Create a chart using plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode='lines+markers',
            name='CPU Usage',
            line=dict(color='rgba(0, 100, 255, 0.8)', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.1)'
        ))
        
        # Add a target line
        fig.add_trace(go.Scatter(
            x=times,
            y=[70] * len(times),  # Target CPU line at 70%
            mode='lines',
            name='Target',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='CPU Usage Over Time',
            xaxis_title='Time',
            yaxis_title='Usage (%)',
            yaxis=dict(range=[0, 100]),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add statistics below the chart
        cols = st.columns(4)
        
        # Calculate statistics
        avg_value = sum(values) / len(values) if values else 0
        min_value = min(values) if values else 0
        max_value = max(values) if values else 0
        current = values[-1] if values else 0
        
        # Display statistics with color coding
        avg_color = "green" if avg_value < 70 else "orange" if avg_value < 90 else "red"
        min_color = "green"  # Min is always good
        max_color = "green" if max_value < 70 else "orange" if max_value < 90 else "red"
        current_color = "green" if current < 70 else "orange" if current < 90 else "red"
        
        cols[0].metric("Current", f"{current:.1f}%", delta=f"{current - avg_value:.1f}%")
        cols[1].metric("Average", f"{avg_value:.1f}%")
        cols[2].metric("Minimum", f"{min_value:.1f}%")
        cols[3].metric("Maximum", f"{max_value:.1f}%")

    def _render_resource_usage_chart(self):
        """Render resource usage chart using historical data"""
        if 'historical_data' not in st.session_state or not st.session_state.historical_data['memory']:
            st.info("Collecting data... Please wait.")
            return
            
        # Create dataframe from historical data
        memory_data = st.session_state.historical_data['memory'][-30:]  # Last 30 data points
        network_data = st.session_state.historical_data['network'][-30:]  # Last 30 data points
        
        # Format the data for plotting
        times = [t.strftime('%H:%M:%S') for t, _ in memory_data]
        memory_values = [v for _, v in memory_data]
        network_values = [v for _, v in network_data]
        
        # Create a chart using plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=memory_values,
            mode='lines',
            name='Memory Usage',
            line=dict(color='rgba(255, 165, 0, 0.8)', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=network_values,
            mode='lines',
            name='Network Load',
            line=dict(color='rgba(0, 128, 0, 0.8)', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Memory & Network Usage',
            xaxis_title='Time',
            yaxis_title='Usage (%)',
            yaxis=dict(range=[0, 100]),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add resource usage gauge meters
        cols = st.columns(2)
        
        # Memory gauge
        current_memory = memory_values[-1] if memory_values else 0
        cols[0].markdown(f"""
        <div style="text-align: center;">
            <p style="margin-bottom: 5px;"><strong>Memory Usage</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        memory_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_memory,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': ""},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(255, 165, 0, 0.8)"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(0, 255, 0, 0.1)"},
                    {'range': [50, 80], 'color': "rgba(255, 255, 0, 0.1)"},
                    {'range': [80, 100], 'color': "rgba(255, 0, 0, 0.1)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        memory_fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
        cols[0].plotly_chart(memory_fig, use_container_width=True)
        
        # Network gauge
        current_network = network_values[-1] if network_values else 0
        cols[1].markdown(f"""
        <div style="text-align: center;">
            <p style="margin-bottom: 5px;"><strong>Network Load</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        network_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_network,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': ""},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(0, 128, 0, 0.8)"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(0, 255, 0, 0.1)"},
                    {'range': [50, 80], 'color': "rgba(255, 255, 0, 0.1)"},
                    {'range': [80, 100], 'color': "rgba(255, 0, 0, 0.1)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        network_fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
        cols[1].plotly_chart(network_fig, use_container_width=True)
        
    def _render_temperature_chart(self):
        """Render temperature chart using historical data"""
        if 'historical_data' not in st.session_state or 'temperature' not in st.session_state.historical_data or not st.session_state.historical_data['temperature']:
            st.info("Collecting temperature data... Please wait.")
            return
            
        # Create dataframe from historical data
        temp_data = st.session_state.historical_data['temperature'][-30:]  # Last 30 data points
        
        # Format the data for plotting
        times = [t.strftime('%H:%M:%S') for t, _ in temp_data]
        values = [v for _, v in temp_data]
        
        # Create a chart using plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode='lines+markers',
            name='Temperature',
            line=dict(color='rgba(255, 50, 50, 0.8)', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 50, 50, 0.1)'
        ))
        
        # Add a warning threshold line
        fig.add_trace(go.Scatter(
            x=times,
            y=[55] * len(times),  # Warning temperature at 55°C
            mode='lines',
            name='Warning',
            line=dict(color='rgba(255, 165, 0, 0.8)', width=1, dash='dash')
        ))
        
        # Add a critical threshold line
        fig.add_trace(go.Scatter(
            x=times,
            y=[65] * len(times),  # Critical temperature at 65°C
            mode='lines',
            name='Critical',
            line=dict(color='rgba(255, 0, 0, 0.8)', width=1, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='System Temperature',
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
            yaxis=dict(range=[30, 70]),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add temperature indicator
        current_temp = values[-1] if values else 0
        temp_color = "green" if current_temp < 55 else "orange" if current_temp < 65 else "red"
        
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <div style="display: flex; align-items: center; background-color: rgba(0,0,0,0.05); padding: 15px; border-radius: 5px; min-width: 300px; justify-content: center;">
                <div style="font-size: 2.5rem; color: {temp_color}; margin-right: 10px;">🌡️</div>
                <div>
                    <div style="font-size: 2rem; font-weight: bold; color: {temp_color};">{current_temp:.1f}°C</div>
                    <div style="font-size: 0.9rem; color: gray;">Current Temperature</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    def _render_response_time_chart(self):
        """Render response time chart using historical data"""
        if 'historical_data' not in st.session_state or 'response_time' not in st.session_state.historical_data or not st.session_state.historical_data['response_time']:
            st.info("Collecting response time data... Please wait.")
            return
            
        # Create dataframe from historical data
        resp_data = st.session_state.historical_data['response_time'][-30:]  # Last 30 data points
        
        # Format the data for plotting
        times = [t.strftime('%H:%M:%S') for t, _ in resp_data]
        values = [v for _, v in resp_data]
        
        # Create a chart using plotly
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=times,
            y=values,
            name='Response Time',
            marker_color='rgba(50, 100, 255, 0.7)'
        ))
        
        # Add a threshold line for good response time
        fig.add_trace(go.Scatter(
            x=times,
            y=[100] * len(times),  # Good response time threshold
            mode='lines',
            name='Target',
            line=dict(color='rgba(0, 200, 0, 0.8)', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='System Response Time',
            xaxis_title='Time',
            yaxis_title='Response Time (ms)',
            yaxis=dict(range=[0, 200]),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add response time statistics
        cols = st.columns(4)
        
        # Calculate statistics
        avg_value = sum(values) / len(values) if values else 0
        min_value = min(values) if values else 0
        max_value = max(values) if values else 0
        current = values[-1] if values else 0
        
        cols[0].metric("Current", f"{current:.1f} ms", delta=f"{avg_value - current:.1f} ms", delta_color="inverse")
        cols[1].metric("Average", f"{avg_value:.1f} ms")
        cols[2].metric("Minimum", f"{min_value:.1f} ms")
        cols[3].metric("Maximum", f"{max_value:.1f} ms")
        
    def _render_error_rate_chart(self):
        """Render error rate chart using historical data"""
        if 'historical_data' not in st.session_state or 'error_rate' not in st.session_state.historical_data or not st.session_state.historical_data['error_rate']:
            st.info("Collecting error rate data... Please wait.")
            return
            
        # Create dataframe from historical data
        error_data = st.session_state.historical_data['error_rate'][-30:]  # Last 30 data points
        
        # Format the data for plotting
        times = [t.strftime('%H:%M:%S') for t, _ in error_data]
        values = [v for _, v in error_data]
        
        # Create a chart using plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode='lines+markers',
            name='Error Rate',
            line=dict(color='rgba(255, 0, 0, 0.7)', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
        
        # Add a threshold line
        fig.add_trace(go.Scatter(
            x=times,
            y=[2.0] * len(times),  # Target threshold at 2%
            mode='lines',
            name='Threshold',
            line=dict(color='rgba(255, 165, 0, 0.8)', width=1, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='System Error Rate',
            xaxis_title='Time',
            yaxis_title='Error Rate (%)',
            yaxis=dict(range=[0, 6]),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add error summary table if there are errors
        if any(v > 0 for v in values):
            st.subheader("Recent Errors")
            
            # Generate some simulated error data
            error_types = [
                "Database Connection Timeout", 
                "API Rate Limit Exceeded", 
                "Memory Allocation Failed",
                "Network Socket Error",
                "File I/O Error"
            ]
            
            # Create a random number of errors (1-5)
            num_errors = random.randint(1, 5)
            error_log = []
            
            for _ in range(num_errors):
                error_type = random.choice(error_types)
                timestamp = (datetime.now() - timedelta(minutes=random.randint(1, 30))).strftime('%H:%M:%S')
                count = random.randint(1, 10)
                severity = random.choice(["Low", "Medium", "High"])
                error_log.append({
                    "Error Type": error_type,
                    "Time": timestamp,
                    "Count": count,
                    "Severity": severity
                })
            
            # Convert to DataFrame and display
            error_df = pd.DataFrame(error_log)
            st.dataframe(error_df, use_container_width=True)
        else:
            st.success("No errors detected in the current time period.")

    def render_quick_actions(self):
        """Render quick action buttons and active screens"""
        st.markdown("<h3>Quick Actions & Active Screens</h3>", unsafe_allow_html=True)
        
        # Add screen selection to session state if not exists
        if 'active_screen' not in st.session_state:
            st.session_state.active_screen = None
        
        # Create a row of buttons using columns
        cols = st.columns(4)
        
        # Optimize button - also acts as a screen selector
        optimize_active = st.session_state.active_screen == "optimize"
        if cols[0].button("⚡ Optimize", key="header_optimize", 
                         help="View and run system optimization", 
                         use_container_width=True):
            # Toggle screen
            st.session_state.active_screen = "optimize" if not optimize_active else None
            st.rerun()
            
        # Refresh button - refreshes data without screen
        if cols[1].button("🔄 Refresh", key="header_refresh", 
                         help="Refresh all system data",
                         use_container_width=True):
            # Actually refresh data
            self._refresh_data()
            st.success("Data refreshed!")
            
        # Report button - also acts as a screen selector
        report_active = st.session_state.active_screen == "report"
        if cols[2].button("📊 Report", key="header_report", 
                         help="View and generate system reports",
                         use_container_width=True):
            # Toggle screen
            st.session_state.active_screen = "report" if not report_active else None
            st.rerun()
            
        # Settings button - also acts as a screen selector
        settings_active = st.session_state.active_screen == "settings"
        if cols[3].button("⚙️ Settings", key="header_settings",
                         help="Configure system settings",
                         use_container_width=True):
            # Toggle screen
            st.session_state.active_screen = "settings" if not settings_active else None
            st.rerun()
        
        # Display the selected active screen
        if st.session_state.active_screen:
            st.markdown("---")
            
            # Optimize screen
            if st.session_state.active_screen == "optimize":
                self._render_optimize_screen()
                
            # Report screen
            elif st.session_state.active_screen == "report":
                self._render_report_screen()
                
            # Settings screen
            elif st.session_state.active_screen == "settings":
                self._render_settings_screen()
    
    def _render_optimize_screen(self):
        """Render the optimization screen"""
        st.subheader("System Optimization")
        
        # Create optimization options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Optimization Targets")
            cpu_target = st.slider("CPU Usage Target (%)", 
                                 min_value=20, max_value=80, 
                                 value=40, step=5,
                                 help="Lower values save power but may reduce performance")
            
            memory_target = st.slider("Memory Usage Target (%)", 
                                    min_value=30, max_value=90, 
                                    value=60, step=5,
                                    help="Lower values free up memory for other applications")
            
            network_target = st.slider("Network Load Target (%)", 
                                     min_value=20, max_value=80, 
                                     value=50, step=5,
                                     help="Lower values reduce bandwidth consumption")
        
        with col2:
            st.write("Optimization Strategy")
            strategy = st.radio("Strategy", 
                              ["Balanced", "Performance", "Power Saving", "Custom"],
                              help="Determines how resources are allocated")
            
            priority = st.radio("Priority", 
                              ["Normal", "High", "Real-time"],
                              help="Sets the priority level for system processes")
            
            background_tasks = st.checkbox("Pause background tasks", 
                                         value=True,
                                         help="Suspends non-essential background operations")
        
        # Run optimization button
        if st.button("Run Optimization", key="run_optimization", type="primary"):
            return self._run_optimization_with_targets(
                cpu_target, memory_target, network_target, strategy, priority, background_tasks
            )
    
    def _render_report_screen(self):
        """Render the report generation screen"""
        st.subheader("System Reports")
        
        # Report options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Report Configuration")
            report_type = st.selectbox("Report Type", 
                                     ["System Performance", "Resource Usage", "Comprehensive", "Custom"],
                                     help="Type of data to include in the report")
            
            time_period = st.selectbox("Time Period", 
                                     ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"],
                                     help="Time range for the report data")
            
            if time_period == "Custom":
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
                end_date = st.date_input("End Date", value=datetime.now())
        
        with col2:
            st.write("Output Options")
            format_option = st.selectbox("Format", 
                                       ["CSV", "Excel", "PDF", "JSON"],
                                       help="File format for the generated report")
            
            include_charts = st.checkbox("Include Charts", 
                                       value=True,
                                       help="Add graphical visualization to reports")
            
            include_alerts = st.checkbox("Include Alerts", 
                                       value=True,
                                       help="Include system alert history in reports")
        
        # Generate report button
        if st.button("Generate Report", key="generate_report", type="primary"):
            self._generate_custom_report(report_type, time_period, format_option, include_charts, include_alerts)
            
            # Offer download of generated report
            if format_option == "CSV":
                st.download_button(
                    "📥 Download Report",
                    self._get_report_data(),
                    "system_report.csv",
                    key="download_report_file"
                )
            elif format_option == "JSON":
                st.download_button(
                    "📥 Download Report",
                    self._get_report_data(format="json"),
                    "system_report.json",
                    key="download_report_file"
                )
    
    def _render_settings_screen(self):
        """Render the settings configuration screen"""
        st.subheader("System Settings")
        
        # Create tabs for different settings categories
        tabs = st.tabs(["General", "Performance", "Notifications", "Advanced"])
        
        # General settings tab
        with tabs[0]:
            st.write("General Settings")
            
            update_interval = st.slider("Data Update Interval (seconds)", 
                                      min_value=1, max_value=60, 
                                      value=3, step=1,
                                      help="How frequently to refresh system data")
            
            dark_mode = st.checkbox("Dark Mode", 
                                  value=True,
                                  help="Use dark theme for interface")
            
            compact_view = st.checkbox("Compact View", 
                                     value=False,
                                     help="Use condensed layout to show more information")
        
        # Performance settings tab
        with tabs[1]:
            st.write("Performance Settings")
            
            cpu_limit = st.slider("CPU Usage Limit (%)", 
                                min_value=50, max_value=100, 
                                value=90, step=5,
                                help="Maximum allowed CPU usage")
            
            memory_limit = st.slider("Memory Usage Limit (%)", 
                                   min_value=50, max_value=100, 
                                   value=85, step=5,
                                   help="Maximum allowed memory usage")
            
            throttling = st.checkbox("Enable Throttling", 
                                   value=True,
                                   help="Automatically reduce resource usage when limits are reached")
        
        # Notification settings tab
        with tabs[2]:
            st.write("Notification Settings")
            
            enable_alerts = st.checkbox("Enable Alert Notifications", 
                                      value=True,
                                      help="Show notifications for system alerts")
            
            alert_level = st.selectbox("Minimum Alert Level", 
                                     ["Low", "Medium", "High", "Critical"],
                                     index=1,
                                     help="Only show alerts at or above this severity level")
            
            desktop_notifications = st.checkbox("Desktop Notifications", 
                                             value=True,
                                             help="Show system notifications on desktop")
        
        # Advanced settings tab
        with tabs[3]:
            st.write("Advanced Settings")
            
            st.warning("Changing these settings may affect system stability")
            
            debug_mode = st.checkbox("Debug Mode", 
                                   value=False,
                                   help="Enable detailed logging and diagnostics")
            
            api_access = st.checkbox("Enable API Access", 
                                   value=True,
                                   help="Allow external systems to access data via API")
            
            retention_period = st.slider("Data Retention Period (days)", 
                                       min_value=7, max_value=365, 
                                       value=30, step=1,
                                       help="How long to keep historical data")
        
        # Save settings button
        if st.button("Save Settings", key="save_settings", type="primary"):
            # Save all settings to session state
            settings = {
                "general": {
                    "update_interval": update_interval,
                    "dark_mode": dark_mode,
                    "compact_view": compact_view
                },
                "performance": {
                    "cpu_limit": cpu_limit,
                    "memory_limit": memory_limit,
                    "throttling": throttling
                },
                "notifications": {
                    "enable_alerts": enable_alerts,
                    "alert_level": alert_level,
                    "desktop_notifications": desktop_notifications
                },
                "advanced": {
                    "debug_mode": debug_mode,
                    "api_access": api_access,
                    "retention_period": retention_period
                }
            }
            
            st.session_state.system_settings = settings
            st.success("Settings saved successfully!")
    
    def _run_optimization_with_targets(self, cpu_target, memory_target, network_target, 
                                     strategy, priority, background_tasks):
        """Run system optimization with custom targets"""
        import time
        
        # Create a progress bar for the optimization
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_area = st.empty()
        
        # Simulate optimization steps
        steps = [
            f"Analyzing current resource usage...",
            f"Setting CPU target to {cpu_target}%...",
            f"Setting memory target to {memory_target}%...",
            f"Setting network target to {network_target}%...",
            f"Applying {strategy} optimization strategy...",
            f"Setting process priority to {priority}...",
            "Optimizing system resources..."
        ]
        
        if background_tasks:
            steps.append("Suspending background tasks...")
            
        steps.append("Finalizing optimizations...")
        
        # Execute optimization steps
        for i, step in enumerate(steps):
            # Update status and progress
            status_text.text(step)
            progress_bar.progress((i+1)/len(steps))
            time.sleep(0.5)  # Simulate work
        
        # Calculate new resource values based on targets
        new_cpu = max(5, min(cpu_target + random.uniform(-5, 5), 
                            self.system_status['cpu_usage'] * 0.7))
        new_memory = max(10, min(memory_target + random.uniform(-5, 5), 
                               self.system_status['memory_usage'] * 0.8))
        new_network = max(5, min(network_target + random.uniform(-5, 5), 
                               self.system_status['network_load'] * 0.9))
        
        # Clear the progress indicators when done
        status_text.empty()
        progress_bar.empty()
        
        # Show optimization results
        results_area.success("System optimized successfully!")
        
        # Create a results table
        results = pd.DataFrame({
            "Metric": ["CPU Usage", "Memory Usage", "Network Load"],
            "Before": [f"{self.system_status['cpu_usage']:.1f}%", 
                     f"{self.system_status['memory_usage']:.1f}%", 
                     f"{self.system_status['network_load']:.1f}%"],
            "After": [f"{new_cpu:.1f}%", f"{new_memory:.1f}%", f"{new_network:.1f}%"],
            "Change": [f"-{self.system_status['cpu_usage'] - new_cpu:.1f}%", 
                     f"-{self.system_status['memory_usage'] - new_memory:.1f}%", 
                     f"-{self.system_status['network_load'] - new_network:.1f}%"]
        })
        
        results_area.table(results)
        
        # Update the system status with optimized values
        optimized_status = {
            'cpu_usage': new_cpu,
            'memory_usage': new_memory,
            'network_load': new_network,
            'active_sensors': self.system_status['active_sensors']
        }
        
        # Return optimized values
        st.session_state.system_status = optimized_status
        return optimized_status
        
    def _generate_custom_report(self, report_type, time_period, format_option, include_charts, include_alerts):
        """Generate a comprehensive system report"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Determine the time range
        now = datetime.now()
        if time_period == "Last Hour":
            start_time = now - timedelta(hours=1)
        elif time_period == "Last 24 Hours":
            start_time = now - timedelta(days=1)
        elif time_period == "Last 7 Days":
            start_time = now - timedelta(days=7)
        elif time_period == "Last 30 Days":
            start_time = now - timedelta(days=30)
        else:  # Custom or default
            start_time = now - timedelta(days=1)
        
        # Generate intervals within the time range
        interval_count = 12
        delta = (now - start_time) / interval_count
        intervals = [start_time + i * delta for i in range(interval_count + 1)]
        
        # Create simulated metrics for the report based on report type
        base_data = {
            'timestamp': intervals,
            'cpu_usage': [max(0, min(100, self.system_status['cpu_usage'] + np.random.normal(0, 5))) for _ in intervals],
            'memory_usage': [max(0, min(100, self.system_status['memory_usage'] + np.random.normal(0, 3))) for _ in intervals],
            'network_load': [max(0, min(100, self.system_status['network_load'] + np.random.normal(0, 4))) for _ in intervals],
            'active_sensors': [max(0, self.system_status['active_sensors'] + int(np.random.normal(0, 1))) for _ in intervals]
        }
        
        # Add additional metrics based on report type
        if report_type == "Comprehensive" or report_type == "Custom":
            base_data.update({
                'disk_usage': [max(0, min(100, 60 + np.random.normal(0, 3))) for _ in intervals],
                'temperature': [max(0, min(100, 45 + np.random.normal(0, 2))) for _ in intervals],
                'power_usage': [max(0, min(200, 120 + np.random.normal(0, 10))) for _ in intervals],
                'uptime_hours': [(now - start_time).total_seconds() / 3600 + i for i in range(len(intervals))],
                'error_count': [max(0, int(np.random.normal(1, 1))) for _ in intervals]
            })
        
        # Format timestamp for display
        report_data = base_data.copy()
        report_data['timestamp'] = [t.strftime('%Y-%m-%d %H:%M:%S') for t in intervals]
        
        # Create and store the report
        st.session_state.system_report = pd.DataFrame(report_data)
        st.session_state.report_format = format_option
        
        # If charts are included, generate and display them
        if include_charts:
            self._display_report_charts(report_data, report_type)
        
        # If alerts are included, display recent alerts
        if include_alerts and hasattr(st.session_state, 'system_alerts'):
            st.subheader("Recent Alerts")
            if not st.session_state.system_alerts:
                st.info("No alerts recorded during this period")
            else:
                for alert in st.session_state.system_alerts[:5]:  # Show top 5 alerts
                    severity_color = "red" if alert["severity"] == "critical" else "orange"
                    severity_icon = "🔴" if alert["severity"] == "critical" else "🟠"
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 8px; padding: 8px; border-left: 3px solid {severity_color};">
                        <div style="display: flex; justify-content: space-between;">
                            <span>{severity_icon} {alert["message"]}</span>
                            <span style="color: gray; font-size: 0.8em;">{alert["time"]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    def _display_report_charts(self, report_data, report_type):
        """Display charts for the generated report"""
        st.subheader("Report Visualizations")
        
        # Create a line chart for key metrics
        metrics_chart = pd.DataFrame({
            'Time': report_data['timestamp'],
            'CPU Usage (%)': report_data['cpu_usage'],
            'Memory Usage (%)': report_data['memory_usage'],
            'Network Load (%)': report_data['network_load']
        })
        
        st.line_chart(metrics_chart.set_index('Time'))
        
        # For comprehensive reports, show additional charts
        if report_type == "Comprehensive" or report_type == "Custom":
            # Create a second chart for additional metrics
            resource_chart = pd.DataFrame({
                'Time': report_data['timestamp'],
                'Disk Usage (%)': report_data['disk_usage'],
                'Temperature (°C)': report_data['temperature'],
                'Power Usage (W)': [p/2 for p in report_data['power_usage']]  # Scale for better visualization
            })
            
            st.line_chart(resource_chart.set_index('Time'))
            
            # Create a bar chart for error counts
            if 'error_count' in report_data:
                # Group errors by hour for visualization
                error_data = pd.DataFrame({
                    'Timestamp': report_data['timestamp'],
                    'Errors': report_data['error_count']
                })
                st.bar_chart(error_data.set_index('Timestamp'))
    
    def _get_report_data(self, format="csv"):
        """Return the report data in the specified format"""
        if 'system_report' in st.session_state:
            if format == "csv":
                return st.session_state.system_report.to_csv(index=False)
            elif format == "json":
                return st.session_state.system_report.to_json(orient="records")
            else:
                return st.session_state.system_report.to_csv(index=False)
        else:
            # Create a default report if none exists
            self._generate_report()
            return self._get_report_data(format)
    
    def _refresh_data(self):
        """Actually refresh system data"""
        # Generate new data
        import time
        
        # Simulate data refresh
        with st.spinner("Refreshing system data..."):
            time.sleep(1)  # Simulate network delay
        
        # Update animation frame to force redraw
        self.animation_frame += 10
        
        # Update all metrics with fresh data
        self.update_system_status()
        
        # Generate new data points
        for _ in range(5):  # Add several new data points
            self.generate_live_data()
            
    def _optimize_system(self):
        """Actually optimize system resources"""
        # Simulate resource optimization
        import time
        
        # Create a progress bar for the optimization
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate optimization steps
        steps = ["Analyzing CPU usage...", "Optimizing memory allocation...", 
                 "Cleaning cache...", "Adjusting network priorities...", 
                 "Finalizing optimizations..."]
        
        for i, step in enumerate(steps):
            # Update status and progress
            status_text.text(step)
            progress_bar.progress((i+1)/len(steps))
            time.sleep(0.5)  # Simulate work
        
        # Clear the progress indicators when done
        status_text.empty()
        progress_bar.empty()
        
        # Return optimized status with improved metrics
        return {
            'cpu_usage': max(5, self.system_status['cpu_usage'] * 0.7),  # Reduce by 30%
            'memory_usage': max(10, self.system_status['memory_usage'] * 0.8),  # Reduce by 20%
            'network_load': max(5, self.system_status['network_load'] * 0.9),  # Reduce by 10%
            'active_sensors': self.system_status['active_sensors']
        }
    
    def _generate_report(self):
        """Generate a basic system report (for compatibility with older code)"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create simple report data
        now = datetime.now()
        intervals = [now - timedelta(minutes=i*10) for i in range(12)]
        
        # Create simulated metrics for the report
        report_data = {
            'timestamp': intervals,
            'cpu_usage': [max(0, min(100, self.system_status['cpu_usage'] + np.random.normal(0, 5))) for _ in intervals],
            'memory_usage': [max(0, min(100, self.system_status['memory_usage'] + np.random.normal(0, 3))) for _ in intervals],
            'network_load': [max(0, min(100, self.system_status['network_load'] + np.random.normal(0, 4))) for _ in intervals],
            'active_sensors': [max(0, self.system_status['active_sensors'] + int(np.random.normal(0, 1))) for _ in intervals]
        }
        
        # Store report in session state
        st.session_state.system_report = pd.DataFrame(report_data)

# The following lines cause Streamlit commands to run when this module is imported
# Remove or comment them out to prevent conflicts with st.set_page_config()
# header = AdvancedInteractiveHeader()
# header.render()
