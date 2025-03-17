import streamlit as st
from .process_monitoring import ProcessMonitor
from .advanced_ml_models import SynchronizedDigitalTwin
import time

class IntegratedDashboard:
    def __init__(self):
        self.process_monitor = ProcessMonitor()
        self.digital_twin = SynchronizedDigitalTwin()
        
    def render(self):
        # Main title and description
        st.title("🏭 Manufacturing Process Digital Twin")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs([
            "Integrated View",
            "Process Monitoring",
            "Digital Twin"
        ])
        
        with tab1:
            self._render_integrated_view()
        
        with tab2:
            self.process_monitor.render_monitoring_dashboard()
        
        with tab3:
            self.digital_twin.render_synchronized_view()
    
    def _render_integrated_view(self):
        """Render the integrated view combining process monitoring and digital twin"""
        # Create a 2-column layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Digital Twin Visualization")
            twin_placeholder = st.empty()
            
            # Initialize digital twin if not running
            if not hasattr(self.digital_twin, 'is_running') or not self.digital_twin.is_running:
                self.digital_twin.start_video_capture()
            
            # Create 3D visualization
            fig = self.digital_twin.create_3d_twin_visualization()
            twin_placeholder.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Process Metrics")
            metrics_placeholder = st.empty()
            
            # Initialize process monitoring if not running
            if not self.process_monitor.is_running:
                self.process_monitor.start_monitoring()
            
            # Create metrics visualization
            fig = self.process_monitor._create_metrics_visualization()
            metrics_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Create status indicators below
        st.subheader("Real-time Status")
        status_cols = st.columns(4)
        
        # Update status indicators
        if self.process_monitor.metrics['production_rate'] and len(self.process_monitor.metrics['production_rate']) > 0:
            with status_cols[0]:
                current_rate = self.process_monitor.metrics['production_rate'][-1]
                delta = None
                if len(self.process_monitor.metrics['production_rate']) > 1:
                    delta = f"{current_rate - self.process_monitor.metrics['production_rate'][-2]:.1f}%"
                st.metric(
                    "Production Rate",
                    f"{current_rate:.1f}%",
                    delta
                )
        
        if self.process_monitor.metrics['quality_score'] and len(self.process_monitor.metrics['quality_score']) > 0:
            with status_cols[1]:
                current_score = self.process_monitor.metrics['quality_score'][-1]
                delta = None
                if len(self.process_monitor.metrics['quality_score']) > 1:
                    delta = f"{current_score - self.process_monitor.metrics['quality_score'][-2]:.1f}%"
                st.metric(
                    "Quality Score",
                    f"{current_score:.1f}%",
                    delta
                )
        
        with status_cols[2]:
            if (self.process_monitor.metrics['production_rate'] and 
                self.process_monitor.metrics['quality_score'] and
                len(self.process_monitor.metrics['production_rate']) > 0 and
                len(self.process_monitor.metrics['quality_score']) > 0):
                efficiency = (self.process_monitor.metrics['production_rate'][-1] * 
                            self.process_monitor.metrics['quality_score'][-1] / 100)
                st.metric("Overall Efficiency", f"{efficiency:.1f}%")
            else:
                st.metric("Overall Efficiency", "N/A")
        
        with status_cols[3]:
            if len(self.process_monitor.metrics['timestamps']) > 0:
                uptime = 100 - (len([a for a in self.process_monitor.alerts if a['level'] == 'CRITICAL']) / 
                            max(1, len(self.process_monitor.metrics['timestamps'])) * 100)
                st.metric("System Uptime", f"{uptime:.1f}%")
            else:
                st.metric("System Uptime", "N/A")
        
        # Recent alerts
        st.subheader("Recent Alerts")
        alerts_container = st.container()
        with alerts_container:
            if self.process_monitor.alerts:
                for alert in reversed(self.process_monitor.alerts[-5:]):
                    color = "🔴" if alert['level'] == 'CRITICAL' else "🟡"
                    st.markdown(f"{color} **{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}")
            else:
                st.info("No alerts to display")
        
        # Controls
        st.subheader("System Controls")
        control_cols = st.columns(4)
        
        with control_cols[0]:
            if st.button("Emergency Stop", type="primary", key="id_emergency_stop"):
                st.error("Emergency stop activated!")
                self.process_monitor.stop()
                self.digital_twin.stop()
        
        with control_cols[1]:
            if st.button("Reset Alerts", key="id_reset_alerts"):
                self.process_monitor.alerts.clear()
        
        with control_cols[2]:
            if st.button("Restart Monitoring", key="id_restart_monitoring"):
                self.process_monitor.stop()
                self.digital_twin.stop()
                time.sleep(1)
                self.process_monitor.start_monitoring()
                self.digital_twin.start_video_capture()
        
        with control_cols[3]:
            st.button("Export Data", key="id_export_data")
    
    def stop(self):
        """Stop all monitoring and visualization processes"""
        self.process_monitor.stop()
        self.digital_twin.stop()

def render_integrated_dashboard():
    """Render the integrated dashboard view."""
    dashboard = IntegratedDashboard()
    dashboard.render()

def main():
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = IntegratedDashboard()
    
    st.session_state.dashboard.render()

if __name__ == "__main__":
    main() 