import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

def generate_sample_resource_data() -> Dict[str, pd.DataFrame]:
    """Generate sample resource allocation data."""
    np.random.seed(42)
    
    # Generate machine utilization data
    machines = pd.DataFrame({
        'machine_id': [f'Machine-{i}' for i in range(1, 6)],
        'utilization': np.random.uniform(60, 95, 5),
        'efficiency': np.random.uniform(75, 98, 5),
        'maintenance_hours': np.random.uniform(2, 8, 5),
        'queue_size': np.random.randint(0, 10, 5)
    })
    
    # Generate operator assignment data
    operators = pd.DataFrame({
        'operator_id': [f'Operator-{i}' for i in range(1, 4)],
        'shift': ['Morning', 'Afternoon', 'Night'],
        'machines_assigned': [
            'Machine-1, Machine-2',
            'Machine-3, Machine-4',
            'Machine-5'
        ],
        'workload': np.random.uniform(70, 90, 3)
    })
    
    # Generate material inventory data
    materials = pd.DataFrame({
        'material_id': [f'Material-{i}' for i in range(1, 7)],
        'current_stock': np.random.uniform(100, 1000, 6),
        'reorder_point': np.random.uniform(200, 400, 6),
        'lead_time_days': np.random.uniform(3, 10, 6),
        'daily_consumption': np.random.uniform(10, 50, 6)
    })
    
    return {
        'machines': machines,
        'operators': operators,
        'materials': materials
    }

def render_resource_allocation():
    """Render the resource allocation dashboard."""
    st.header("Resource Allocation Dashboard")
    
    # Initialize components
    digital_twin = SynchronizedDigitalTwin()
    visualizer = MultiModalVisualizer()
    
    # Sidebar controls
    st.sidebar.subheader("Resource Parameters")
    resource_type = st.sidebar.selectbox(
        "Resource Type",
        ["Machines", "Operators", "Materials"]
    )
    
    time_period = st.sidebar.selectbox(
        "Time Period",
        ["Current Shift", "Today", "This Week", "This Month"]
    )
    
    # Generate sample data
    resource_data = generate_sample_resource_data()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Resource Details",
        "Optimization",
        "Reports"
    ])
    
    with tab1:
        st.subheader("Resource Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_machine_util = resource_data['machines']['utilization'].mean()
            st.metric("Avg Machine Utilization",
                     f"{avg_machine_util:.1f}%",
                     f"{np.random.uniform(-2, 2):.1f}%")
        
        with col2:
            avg_operator_workload = resource_data['operators']['workload'].mean()
            st.metric("Avg Operator Workload",
                     f"{avg_operator_workload:.1f}%",
                     f"{np.random.uniform(-2, 2):.1f}%")
        
        with col3:
            materials_below_reorder = len(
                resource_data['materials'][
                    resource_data['materials']['current_stock'] <
                    resource_data['materials']['reorder_point']
                ]
            )
            st.metric("Materials Below Reorder",
                     str(materials_below_reorder),
                     "-1" if materials_below_reorder > 0 else "0")
        
        with col4:
            total_queue = resource_data['machines']['queue_size'].sum()
            st.metric("Total Queue Size",
                     str(total_queue),
                     f"{-1 if total_queue > 20 else 1}")
        
        # Resource utilization chart
        if resource_type == "Machines":
            fig = px.bar(
                resource_data['machines'],
                x='machine_id',
                y=['utilization', 'efficiency'],
                title='Machine Utilization and Efficiency',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif resource_type == "Operators":
            fig = px.bar(
                resource_data['operators'],
                x='operator_id',
                y='workload',
                color='shift',
                title='Operator Workload by Shift'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Materials
            fig = go.Figure()
            materials_df = resource_data['materials']
            
            fig.add_trace(go.Bar(
                x=materials_df['material_id'],
                y=materials_df['current_stock'],
                name='Current Stock'
            ))
            
            fig.add_trace(go.Scatter(
                x=materials_df['material_id'],
                y=materials_df['reorder_point'],
                name='Reorder Point',
                mode='lines+markers',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='Material Inventory Levels',
                xaxis_title='Material',
                yaxis_title='Quantity'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Resource Details")
        
        if resource_type == "Machines":
            st.dataframe(resource_data['machines'])
            
            # Machine status cards
            for _, machine in resource_data['machines'].iterrows():
                status_color = (
                    'red' if machine['utilization'] > 90
                    else 'orange' if machine['utilization'] > 80
                    else 'green'
                )
                
                st.markdown(f"""
                <div style='padding: 10px; border-left: 5px solid {status_color}; margin: 10px 0;'>
                    <strong>{machine['machine_id']}</strong><br>
                    Utilization: {machine['utilization']:.1f}%<br>
                    Efficiency: {machine['efficiency']:.1f}%<br>
                    Queue Size: {machine['queue_size']}<br>
                    Maintenance Hours: {machine['maintenance_hours']:.1f}
                </div>
                """, unsafe_allow_html=True)
        
        elif resource_type == "Operators":
            st.dataframe(resource_data['operators'])
            
            # Operator assignment visualization
            fig = px.timeline(
                resource_data['operators'],
                x_start=0,
                x_end=100,  # Using workload as width
                y='operator_id',
                color='shift',
                title='Operator Assignments'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Materials
            st.dataframe(resource_data['materials'])
            
            # Material status
            for _, material in resource_data['materials'].iterrows():
                days_until_reorder = (
                    material['current_stock'] - material['reorder_point']
                ) / material['daily_consumption']
                
                status_color = (
                    'red' if days_until_reorder < 0
                    else 'orange' if days_until_reorder < material['lead_time_days']
                    else 'green'
                )
                
                st.markdown(f"""
                <div style='padding: 10px; border-left: 5px solid {status_color}; margin: 10px 0;'>
                    <strong>{material['material_id']}</strong><br>
                    Current Stock: {material['current_stock']:.0f}<br>
                    Days until Reorder: {max(0, days_until_reorder):.1f}<br>
                    Lead Time: {material['lead_time_days']:.1f} days<br>
                    Daily Consumption: {material['daily_consumption']:.1f}
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Resource Optimization")
        
        # Optimization suggestions
        if resource_type == "Machines":
            st.write("### Machine Allocation Suggestions")
            
            # Find overloaded and underloaded machines
            overloaded = resource_data['machines'][
                resource_data['machines']['utilization'] > 85
            ]
            underloaded = resource_data['machines'][
                resource_data['machines']['utilization'] < 70
            ]
            
            if not overloaded.empty:
                st.warning("#### Overloaded Machines")
                for _, machine in overloaded.iterrows():
                    st.markdown(f"""
                    - **{machine['machine_id']}** (Utilization: {machine['utilization']:.1f}%)
                      - Suggestion: Redistribute workload or schedule maintenance
                    """)
            
            if not underloaded.empty:
                st.info("#### Underloaded Machines")
                for _, machine in underloaded.iterrows():
                    st.markdown(f"""
                    - **{machine['machine_id']}** (Utilization: {machine['utilization']:.1f}%)
                      - Suggestion: Increase workload or schedule maintenance
                    """)
        
        elif resource_type == "Operators":
            st.write("### Operator Assignment Suggestions")
            
            # Analyze operator workload
            high_workload = resource_data['operators'][
                resource_data['operators']['workload'] > 85
            ]
            
            if not high_workload.empty:
                st.warning("#### High Workload Operators")
                for _, operator in high_workload.iterrows():
                    st.markdown(f"""
                    - **{operator['operator_id']}** ({operator['shift']} shift)
                      - Current workload: {operator['workload']:.1f}%
                      - Suggestion: Redistribute machines or add support
                    """)
        
        else:  # Materials
            st.write("### Inventory Optimization Suggestions")
            
            # Analyze inventory levels
            low_stock = resource_data['materials'][
                resource_data['materials']['current_stock'] <
                resource_data['materials']['reorder_point']
            ]
            
            if not low_stock.empty:
                st.error("#### Materials Requiring Attention")
                for _, material in low_stock.iterrows():
                    days_until_stockout = material['current_stock'] / material['daily_consumption']
                    st.markdown(f"""
                    - **{material['material_id']}**
                      - Current stock: {material['current_stock']:.0f}
                      - Days until stockout: {days_until_stockout:.1f}
                      - Suggestion: Place order immediately
                    """)
    
    with tab4:
        st.subheader("Resource Reports")
        
        report_type = st.selectbox(
            "Report Type",
            ["Utilization Summary", "Efficiency Analysis", "Resource Planning"]
        )
        
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 Hours", "Last Week", "Last Month", "Custom"]
        )
        
        if time_range == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
        
        # Report generation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Report"):
                st.info("Generating resource allocation report...")
        with col2:
            if st.button("Export Data"):
                st.success("Data exported successfully") 