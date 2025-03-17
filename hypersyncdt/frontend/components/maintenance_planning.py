import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def generate_sample_maintenance_data() -> Dict[str, pd.DataFrame]:
    """Generate sample maintenance data for demonstration."""
    # Generate equipment data
    equipment_data = pd.DataFrame({
        'equipment_id': [f'EQ-{i:03d}' for i in range(1, 11)],
        'type': np.random.choice(['CNC Machine', 'Robot', 'Conveyor', 'Press', 'Assembly'], 10),
        'last_maintenance': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'next_maintenance': pd.date_range(start='2024-03-01', periods=10, freq='D'),
        'health_score': np.random.uniform(60, 100, 10),
        'priority': np.random.choice(['High', 'Medium', 'Low'], 10)
    })
    
    # Generate maintenance tasks
    tasks_data = pd.DataFrame({
        'task_id': [f'MT-{i:04d}' for i in range(1, 21)],
        'equipment_id': np.random.choice(equipment_data['equipment_id'], 20),
        'task_type': np.random.choice(['Preventive', 'Corrective', 'Inspection'], 20),
        'scheduled_date': pd.date_range(start='2024-03-01', periods=20, freq='D'),
        'estimated_duration': np.random.uniform(1, 8, 20),
        'assigned_technician': np.random.choice(['Tech A', 'Tech B', 'Tech C', 'Tech D'], 20),
        'status': np.random.choice(['Scheduled', 'In Progress', 'Completed', 'Delayed'], 20)
    })
    
    # Generate resource availability
    technicians_data = pd.DataFrame({
        'technician': ['Tech A', 'Tech B', 'Tech C', 'Tech D'],
        'availability': np.random.uniform(70, 100, 4),
        'tasks_assigned': np.random.randint(2, 8, 4),
        'specialization': ['Mechanical', 'Electrical', 'Software', 'General']
    })
    
    return {
        'equipment': equipment_data,
        'tasks': tasks_data,
        'technicians': technicians_data
    }

def render_maintenance_planning():
    """Render the maintenance planning dashboard."""
    st.header("Maintenance Planning Dashboard", divider="rainbow")
    
    # Get sample data
    data = generate_sample_maintenance_data()
    
    # Sidebar filters
    with st.sidebar:
        st.subheader("Filters")
        
        equipment_type = st.multiselect(
            "Equipment Type",
            options=data['equipment']['type'].unique(),
            default=data['equipment']['type'].unique()
        )
        
        priority = st.multiselect(
            "Priority",
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium', 'Low']
        )
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now(), datetime.now() + timedelta(days=30))
        )
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs([
        "Overview",
        "Schedule",
        "Resources"
    ])
    
    with tab1:
        st.subheader("Maintenance Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tasks = len(data['tasks'])
            st.metric("Total Tasks", total_tasks)
        
        with col2:
            pending_tasks = len(data['tasks'][data['tasks']['status'].isin(['Scheduled', 'Delayed'])])
            st.metric("Pending Tasks", pending_tasks)
        
        with col3:
            avg_duration = data['tasks']['estimated_duration'].mean()
            st.metric("Avg Duration (hrs)", f"{avg_duration:.1f}")
        
        with col4:
            completion_rate = len(data['tasks'][data['tasks']['status'] == 'Completed']) / total_tasks * 100
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Equipment health distribution
        st.subheader("Equipment Health Distribution")
        fig = px.histogram(
            data['equipment'],
            x='health_score',
            color='type',
            title='Equipment Health Scores by Type',
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Task status breakdown
        st.subheader("Task Status Breakdown")
        status_counts = data['tasks']['status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Maintenance Tasks by Status'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Maintenance Schedule")
        
        # Gantt chart
        fig = px.timeline(
            data['tasks'],
            x_start='scheduled_date',
            x_end=data['tasks']['scheduled_date'] + pd.to_timedelta(data['tasks']['estimated_duration'], unit='h'),
            y='equipment_id',
            color='task_type',
            title='Maintenance Schedule Timeline'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tasks table
        st.subheader("Upcoming Tasks")
        st.dataframe(
            data['tasks'].sort_values('scheduled_date'),
            hide_index=True,
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Resource Management")
        
        # Technician workload
        st.subheader("Technician Workload")
        fig = go.Figure(data=[
            go.Bar(
                name='Tasks Assigned',
                x=data['technicians']['technician'],
                y=data['technicians']['tasks_assigned']
            ),
            go.Bar(
                name='Availability (%)',
                x=data['technicians']['technician'],
                y=data['technicians']['availability']
            )
        ])
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Technician details
        st.subheader("Technician Details")
        st.dataframe(
            data['technicians'],
            hide_index=True,
            use_container_width=True
        )

if __name__ == "__main__":
    render_maintenance_planning() 