import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="System Operations Optimization",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("System Operations")
st.sidebar.image("https://img.icons8.com/color/96/000000/settings-3--v1.png", width=100)

optimization_option = st.sidebar.selectbox(
    "Select Optimization Area",
    ["Performance Monitoring", "Resource Allocation", "Error Handling", "Security Measures"]
)

# Main content
st.title("System Operations Optimization Dashboard")
st.write("This dashboard provides tools for optimizing system operations across all environments.")

# Create tabs
tabs = st.tabs(["Overview", "Monitoring", "Optimization", "Documentation"])

with tabs[0]:
    st.header("System Overview")
    st.write("Current system status and key metrics")
    
    # Create some sample metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPU Usage", "42%", "-3%")
    col2.metric("Memory Usage", "3.2 GB", "+0.5 GB")
    col3.metric("Response Time", "120ms", "-15ms")
    col4.metric("Error Rate", "0.05%", "-0.02%")
    
    # Sample chart
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['System A', 'System B', 'System C']
    )
    st.line_chart(chart_data)

with tabs[1]:
    st.header("Performance Monitoring")
    st.write("Track and analyze system performance metrics")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - pd.Timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Generate sample data
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    performance_data = pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': np.random.uniform(20, 80, size=len(dates)),
        'memory_usage': np.random.uniform(2000, 8000, size=len(dates)),
        'response_time': np.random.uniform(50, 200, size=len(dates)),
        'error_rate': np.random.uniform(0, 0.2, size=len(dates))
    })
    
    # Plot the data
    metric_to_plot = st.selectbox("Select Metric to Plot", 
                                 ['cpu_usage', 'memory_usage', 'response_time', 'error_rate'])
    
    fig = px.line(performance_data, x='timestamp', y=metric_to_plot, 
                 title=f"{metric_to_plot.replace('_', ' ').title()} Over Time")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("System Optimization")
    st.write("Tools and recommendations for optimizing system performance")
    
    # Optimization areas
    optimization_areas = {
        "Code Efficiency": {
            "status": "Good",
            "recommendations": [
                "Use caching for frequently accessed data",
                "Implement asynchronous processing for I/O operations",
                "Optimize database queries with proper indexing"
            ]
        },
        "Resource Allocation": {
            "status": "Needs Improvement",
            "recommendations": [
                "Scale up database resources during peak hours",
                "Implement auto-scaling for web servers",
                "Optimize memory usage in data processing pipelines"
            ]
        },
        "Error Handling": {
            "status": "Good",
            "recommendations": [
                "Implement comprehensive logging for all critical operations",
                "Set up automated alerts for critical errors",
                "Create fallback mechanisms for critical services"
            ]
        },
        "Security": {
            "status": "Excellent",
            "recommendations": [
                "Regular security audits and penetration testing",
                "Keep all dependencies updated to latest secure versions",
                "Implement proper authentication and authorization"
            ]
        }
    }
    
    # Display optimization areas
    for area, details in optimization_areas.items():
        with st.expander(f"{area} - Status: {details['status']}"):
            st.write("Recommendations:")
            for rec in details['recommendations']:
                st.write(f"- {rec}")
            
            if st.button(f"Apply Optimizations for {area}"):
                st.success(f"Optimization plan for {area} has been scheduled!")

with tabs[3]:
    st.header("System Documentation")
    st.write("Documentation and resources for system operations")
    
    # Sample documentation
    with st.expander("Environment Configuration"):
        st.write("""
        ## Environment Configuration
        
        The system is configured to work across multiple environments:
        
        - **Development**: For active development and testing
        - **Staging**: For pre-production validation
        - **Production**: Live environment
        
        Each environment uses separate configuration files stored in `.env` files.
        """)
    
    with st.expander("Deployment Process"):
        st.write("""
        ## Deployment Process
        
        1. Code changes are committed to the development branch
        2. CI/CD pipeline runs tests and builds artifacts
        3. Changes are deployed to staging for validation
        4. After approval, changes are deployed to production
        
        Deployments are scheduled during low-traffic periods to minimize impact.
        """)
    
    with st.expander("Troubleshooting Guide"):
        st.write("""
        ## Troubleshooting Guide
        
        ### Common Issues:
        
        1. **High CPU Usage**
           - Check for runaway processes
           - Review recent code changes
           - Examine database query performance
        
        2. **Memory Leaks**
           - Monitor memory usage patterns
           - Check for unclosed resources
           - Review object lifecycle management
        
        3. **Slow Response Times**
           - Check network latency
           - Review database query execution plans
           - Examine caching effectiveness
        """)

# Footer
st.markdown("---")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 