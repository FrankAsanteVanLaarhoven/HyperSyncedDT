import streamlit as st
import sys
import importlib

# Force reload the module
if 'components.interactive_header' in sys.modules:
    importlib.reload(sys.modules['components.interactive_header'])

from components.interactive_header import AdvancedInteractiveHeader

# Set page config
st.set_page_config(page_title="Simple Header Test")

st.title("Simple Header Test")

# Create a fresh instance
header = AdvancedInteractiveHeader()

# Test methods individually
st.subheader("Testing Individual Methods")

# Test system metrics
st.write("### Testing render_system_metrics")
try:
    header.render_system_metrics()
    st.success("✅ render_system_metrics successful")
except Exception as e:
    st.error(f"❌ Error in render_system_metrics: {str(e)}")

# Test system alerts
st.write("### Testing render_system_alerts")
try:
    header.render_system_alerts()
    st.success("✅ render_system_alerts successful")
except Exception as e:
    st.error(f"❌ Error in render_system_alerts: {str(e)}")

# Test quick actions
st.write("### Testing render_quick_actions")
try:
    header.render_quick_actions()
    st.success("✅ render_quick_actions successful")
except Exception as e:
    st.error(f"❌ Error in render_quick_actions: {str(e)}")
    
    # Check if specific methods exist
    missing_methods = []
    for method in ["_refresh_data", "_optimize_system", "_generate_report"]:
        if not hasattr(header, method):
            missing_methods.append(method)
    
    if missing_methods:
        st.error(f"Missing methods: {', '.join(missing_methods)}")

# Test advanced metrics
st.write("### Testing render_advanced_metrics")
try:
    header.render_advanced_metrics()
    st.success("✅ render_advanced_metrics successful")
except Exception as e:
    st.error(f"❌ Error in render_advanced_metrics: {str(e)}")
