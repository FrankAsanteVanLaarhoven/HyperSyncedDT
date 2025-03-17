import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

class ModernDashboardUI:
    def __init__(self):
        # Initialize data storage
        self.metrics = {
            "cpu_usage": 49.7,
            "memory": 41.0,
            "network": 32.7,
            "active_sensors": 10,
            "total_sensors": 12,
            "efficiency": 88.5,
            "quality_score": 95.1,
            "throughput": 781,
            "utilization": 79.8
        }
        self.system_status = "Connected"
        self.last_update = datetime.now()
        
    def render_header(self):
        header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
        
        with header_col1:
            st.markdown("# HyperSyncDT")
            st.markdown("### Select Role")
            role = st.selectbox("Role", ["Operator", "Administrator", "Maintenance", "Supervisor"], label_visibility="collapsed")
            
        with header_col2:
            # Empty column for spacing
            pass
            
        with header_col3:
            st.markdown("<div style='text-align: right;'>⚙️ CONNECTING</div>", unsafe_allow_html=True)
    
    def render_sidebar(self):
        st.markdown("## NAVIGATE TO")
        st.markdown("### Category")
        category = st.selectbox("Category", ["Factory Operations", "Digital Twin", "Maintenance", "Analytics"], key="category_select", label_visibility="collapsed")
        
        st.markdown("### Page")
        page = st.selectbox("Page", ["Factory Connect", "Factory Build", "Factory Analyze", "Factory Operate"], key="page_select", label_visibility="collapsed")
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("📊 Generate Report", use_container_width=True)
        with col2:
            st.button("🔄 Sync Data", use_container_width=True)
            
        st.markdown("---")
        st.markdown("## SYSTEM STATUS")
        
        # Show a green dot for connected status
        st.markdown("🟢 Connected as Operator")
        
        st.markdown("## BACKUP INFORMATION")
        st.markdown("📁 Backup Location:")
        st.markdown("~/Desktop/hyper-synced-dt-mvp-test")
        
        st.progress(0.7)
        
        st.markdown(f"Last backup: 2025-03-13 09:08")
    
    def render_quick_actions(self):
        st.markdown("## Quick Actions & Active Screens")
        
        # Create a row of action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 24px;'>⚡</div>
                <div style='font-size: 14px;'>Optimize</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 24px;'>📝</div>
                <div style='font-size: 14px;'>Refresh</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 24px;'>📊</div>
                <div style='font-size: 14px;'>Report</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 24px;'>⚙️</div>
                <div style='font-size: 14px;'>Settings</div>
            </div>
            """, unsafe_allow_html=True)
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.render_metric_card("CPU Usage", f"{self.metrics['cpu_usage']}%", color="green")
            
        with col2:
            self.render_metric_card("Memory", f"{self.metrics['memory']}%", color="green")
            
        with col3:
            self.render_metric_card("Network", f"{self.metrics['network']}%", color="green")
            
        with col4:
            self.render_metric_card("Active Sensors", f"{self.metrics['active_sensors']}", 
                              subtitle=f"of {self.metrics['total_sensors']} sensors online", color="green")
    
    def render_metric_card(self, title, value, subtitle=None, color="blue"):
        color_map = {
            "blue": "#3498db",
            "green": "#2ecc71",
            "red": "#e74c3c",
            "yellow": "#f39c12"
        }
        
        hex_color = color_map.get(color, color_map["blue"])
        
        # Create a progress bar value (for demonstration)
        progress_value = 0.5
        if isinstance(value, str) and "%" in value:
            try:
                progress_value = float(value.replace("%", "")) / 100
            except:
                pass
        
        card_html = f"""
        <div style='background-color: #1E1E1E; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
            <div style='color: #AAAAAA; font-size: 14px;'>{title}</div>
            <div style='color: #FFFFFF; font-size: 22px; font-weight: bold;'>{value}</div>
            <div style='width: 100%; background-color: #333333; height: 5px; border-radius: 5px; margin-top: 5px;'>
                <div style='width: {progress_value * 100}%; background-color: {hex_color}; height: 5px; border-radius: 5px;'></div>
            </div>
        """
        
        if subtitle:
            card_html += f"<div style='color: #AAAAAA; font-size: 12px; margin-top: 5px;'>{subtitle}</div>"
            
        card_html += "</div>"
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    def render_advanced_metrics(self):
        st.markdown("## Advanced Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_advanced_metric_card("Efficiency", f"{self.metrics['efficiency']}%", color="green")
            
        with col2:
            self.render_advanced_metric_card("Quality Score", f"{self.metrics['quality_score']}", color="green")
            
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_advanced_metric_card("Throughput", f"{self.metrics['throughput']} units", color="green")
            
        with col2:
            self.render_advanced_metric_card("Utilization", f"{self.metrics['utilization']}%", color="green")
    
    def render_advanced_metric_card(self, title, value, color="green"):
        color_map = {
            "blue": "#3498db",
            "green": "#2ecc71",
            "red": "#e74c3c",
            "yellow": "#f39c12"
        }
        
        hex_color = color_map.get(color, color_map["blue"])
        
        # Create a progress bar value (for demonstration)
        progress_value = 0.5
        if isinstance(value, str):
            if "%" in value:
                try:
                    # Ensure value is between 0 and 1
                    progress_value = min(1.0, max(0.0, float(value.replace("%", "")) / 100))
                except:
                    pass
            elif "units" in value:
                try:
                    # Cap at 1.0 to ensure valid progress bar
                    raw_value = float(value.replace(" units", ""))
                    progress_value = min(1.0, max(0.0, raw_value / 1000))
                except:
                    pass
            else:
                try:
                    # Assume value is already a percentage but without % sign
                    # Cap at 1.0 to ensure valid progress bar
                    raw_value = float(value)
                    progress_value = min(1.0, max(0.0, raw_value / 100))
                except:
                    pass
        
        st.markdown(f"##### {title}")
        st.markdown(f"<div style='color: {hex_color}; font-size: 28px; font-weight: bold;'>{value}</div>", unsafe_allow_html=True)
        
        # Add safety check to ensure progress value is always between 0.0 and 1.0
        progress_value = min(1.0, max(0.0, progress_value))
        st.progress(progress_value, f"Progress for {title}")
    
    def render_real_time_monitoring(self):
        st.markdown("## Real-Time Monitoring")
        
        tabs = st.tabs(["System Performance", "Resource Usage", "Temperature", "Response Time", "Errors"])
        
        with tabs[0]:
            # Create sample CPU usage data
            dates = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='1h')
            cpu_values = np.random.uniform(30, 70, size=len(dates))
            target_values = np.ones(len(dates)) * 60
            
            fig = go.Figure()
            
            # Add CPU usage line
            fig.add_trace(go.Scatter(
                x=dates,
                y=cpu_values,
                mode='lines',
                name='CPU Usage',
                line=dict(color='#3498db', width=2)
            ))
            
            # Add target line
            fig.add_trace(go.Scatter(
                x=dates,
                y=target_values,
                mode='lines',
                name='Target',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="CPU Usage Over Time",
                xaxis_title="Time",
                yaxis_title="Percentage",
                yaxis=dict(range=[0, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='#121212',
                paper_bgcolor='#121212',
                font=dict(color='#CCCCCC')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_system_alerts(self):
        st.markdown("## System Alerts")
        
        alerts = [
            {"priority": "High", "component": "Machine #3", "message": "Temperature exceeding threshold", "time": "10:32 AM"},
            {"priority": "Medium", "component": "Production Line A", "message": "Efficiency decreased by 5%", "time": "09:47 AM"},
            {"priority": "Low", "component": "Network Switch", "message": "Packet loss detected", "time": "08:15 AM"},
        ]
        
        # Create a dataframe
        alerts_df = pd.DataFrame(alerts)
        
        # Apply color styling based on priority
        def highlight_priority(val):
            color_map = {
                "High": 'background-color: rgba(231, 76, 60, 0.2)',
                "Medium": 'background-color: rgba(243, 156, 18, 0.2)',
                "Low": 'background-color: rgba(46, 204, 113, 0.2)'
            }
            return color_map.get(val, '')
        
        # Use the newer .map() method instead of .applymap()
        styled_df = alerts_df.style.map(highlight_priority, subset=['priority'])
        
        # Display the styled dataframe
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=140
        )
    
    def render_full_ui(self):
        """Render the complete UI dashboard"""
        # Configure the page
        st.set_page_config(
            page_title="HyperSyncDT Dashboard",
            page_icon="🏭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add custom CSS to more closely match the design
        st.markdown("""
        <style>
        .css-18e3th9 {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        .css-1d391kg {
            padding-top: 1rem;
            padding-right: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #1E1E1E;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #333333;
        }
        .css-10trblm {
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
        }
        [data-testid="stSidebar"] {
            background-color: #121212;
        }
        [data-testid="stSidebarContent"] {
            background-color: #121212;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create the sidebar
        with st.sidebar:
            self.render_sidebar()
            
        # Main content
        # self.render_header()
        self.render_quick_actions()
        self.render_advanced_metrics()
        self.render_real_time_monitoring()
        
        # Create the alerts in an expandable element
        with st.expander("System Alerts", expanded=True):
            self.render_system_alerts()

def render_modern_dashboard():
    """Helper function to initialize and render the dashboard"""
    dashboard = ModernDashboardUI()
    dashboard.render_full_ui() 