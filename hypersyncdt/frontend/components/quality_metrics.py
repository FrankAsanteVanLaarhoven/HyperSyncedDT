import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

def generate_sample_quality_metrics() -> Dict[str, pd.DataFrame]:
    """Generate sample quality metrics data."""
    np.random.seed(42)
    
    # Generate current metrics
    metrics = pd.DataFrame({
        'metric_id': [
            'First Pass Yield',
            'Defect Rate',
            'Customer Returns',
            'Process Capability',
            'OEE',
            'Scrap Rate'
        ],
        'current_value': [
            95.5,
            0.8,
            0.2,
            1.33,
            88.5,
            1.2
        ],
        'unit': [
            '%',
            '%',
            '%',
            'Cpk',
            '%',
            '%'
        ],
        'target': [
            98.0,
            0.5,
            0.1,
            1.50,
            90.0,
            1.0
        ],
        'threshold': [
            95.0,
            1.0,
            0.3,
            1.20,
            85.0,
            1.5
        ]
    })
    
    # Generate historical data
    timestamps = pd.date_range(
        start='2024-03-10',
        end='2024-03-17',
        freq='H'
    )
    
    historical_data = []
    for metric in metrics.itertuples():
        base_value = metric.current_value
        target = metric.target
        variation = abs(target - base_value) * 0.2
        
        for ts in timestamps:
            value = base_value + np.random.normal(0, variation)
            historical_data.append({
                'timestamp': ts,
                'metric_id': metric.metric_id,
                'value': value,
                'unit': metric.unit,
                'target': target
            })
    
    historical_df = pd.DataFrame(historical_data)
    
    return {
        'metrics': metrics,
        'historical': historical_df
    }

def render_quality_metrics():
    """Render the quality metrics dashboard."""
    st.header("Quality Metrics Dashboard")
    
    # Initialize components
    digital_twin = SynchronizedDigitalTwin()
    visualizer = MultiModalVisualizer()
    
    # Sidebar controls
    st.sidebar.subheader("Metric Settings")
    metric_category = st.sidebar.selectbox(
        "Metric Category",
        ["All Metrics", "Product Quality", "Process Quality", "Customer Satisfaction"]
    )
    
    time_period = st.sidebar.selectbox(
        "Time Period",
        ["Last 24 Hours", "Last Week", "Last Month", "Year to Date"]
    )
    
    # Generate sample data
    quality_data = generate_sample_quality_metrics()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Metrics Overview",
        "Trend Analysis",
        "Comparative Analysis",
        "Reports"
    ])
    
    with tab1:
        st.subheader("Quality Metrics Overview")
        
        # Metrics cards
        for _, metric in quality_data['metrics'].iterrows():
            # Calculate status color based on target and threshold
            value = metric['current_value']
            if value >= metric['target']:
                status_color = 'green'
            elif value >= metric['threshold']:
                status_color = 'orange'
            else:
                status_color = 'red'
            
            # Calculate performance vs target
            performance = ((value - metric['threshold']) /
                         (metric['target'] - metric['threshold'])) * 100
            performance = min(max(performance, 0), 100)
            
            st.markdown(f"""
            <div style='padding: 10px; border-left: 5px solid {status_color}; margin: 10px 0;'>
                <strong>{metric['metric_id']}</strong><br>
                Current Value: {value:.2f} {metric['unit']}<br>
                Target: {metric['target']:.2f} {metric['unit']}<br>
                Performance: {performance:.1f}% of target<br>
                Threshold: {metric['threshold']:.2f} {metric['unit']}
            </div>
            """, unsafe_allow_html=True)
        
        # Overall quality score
        overall_score = np.mean([
            (row['current_value'] - row['threshold']) /
            (row['target'] - row['threshold']) * 100
            for _, row in quality_data['metrics'].iterrows()
        ])
        
        st.markdown(f"""
        ### Overall Quality Score
        <div style='text-align: center; font-size: 48px; margin: 20px;'>
            {overall_score:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Trend Analysis")
        
        # Metric selector
        selected_metrics = st.multiselect(
            "Select Metrics",
            quality_data['metrics']['metric_id'].tolist(),
            default=quality_data['metrics']['metric_id'].tolist()[:3]
        )
        
        if selected_metrics:
            # Filter historical data
            filtered_data = quality_data['historical'][
                quality_data['historical']['metric_id'].isin(selected_metrics)
            ]
            
            # Create trend chart
            fig = go.Figure()
            
            for metric_id in selected_metrics:
                metric_data = filtered_data[
                    filtered_data['metric_id'] == metric_id
                ]
                
                fig.add_trace(go.Scatter(
                    x=metric_data['timestamp'],
                    y=metric_data['value'],
                    name=metric_id,
                    mode='lines'
                ))
                
                # Add target line
                target_value = quality_data['metrics'][
                    quality_data['metrics']['metric_id'] == metric_id
                ]['target'].iloc[0]
                
                fig.add_trace(go.Scatter(
                    x=metric_data['timestamp'],
                    y=[target_value] * len(metric_data),
                    name=f"{metric_id} Target",
                    line=dict(dash='dash'),
                    opacity=0.5
                ))
            
            fig.update_layout(
                title="Quality Metrics Trends",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Comparative Analysis")
        
        # Time period comparison
        comparison_period = st.selectbox(
            "Comparison Period",
            ["Previous Week", "Previous Month", "Previous Quarter", "Previous Year"]
        )
        
        # Create comparison chart
        metrics_df = quality_data['metrics']
        
        fig = go.Figure()
        
        # Current values
        fig.add_trace(go.Bar(
            name="Current",
            x=metrics_df['metric_id'],
            y=metrics_df['current_value'],
            marker_color='blue'
        ))
        
        # Target values
        fig.add_trace(go.Bar(
            name="Target",
            x=metrics_df['metric_id'],
            y=metrics_df['target'],
            marker_color='green'
        ))
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            name="Threshold",
            x=metrics_df['metric_id'],
            y=metrics_df['threshold'],
            mode='lines',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Quality Metrics Comparison",
            xaxis_title="Metric",
            yaxis_title="Value",
            barmode='group',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.write("### Statistical Summary")
        
        summary_data = []
        for _, metric in metrics_df.iterrows():
            historical_values = quality_data['historical'][
                quality_data['historical']['metric_id'] == metric['metric_id']
            ]['value']
            
            summary_data.append({
                'Metric': metric['metric_id'],
                'Mean': historical_values.mean(),
                'Std Dev': historical_values.std(),
                'Min': historical_values.min(),
                'Max': historical_values.max(),
                'Target': metric['target'],
                'Current': metric['current_value']
            })
        
        st.dataframe(pd.DataFrame(summary_data))
    
    with tab4:
        st.subheader("Quality Reports")
        
        report_type = st.selectbox(
            "Report Type",
            [
                "Daily Quality Summary",
                "Weekly Performance Report",
                "Monthly Trend Analysis",
                "Quarterly Review"
            ]
        )
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
        
        # Report options
        st.write("### Report Options")
        include_charts = st.checkbox("Include Charts", value=True)
        include_statistics = st.checkbox("Include Statistics", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        # Generate report button
        if st.button("Generate Report"):
            st.info("Generating quality metrics report...")
            
            if include_charts:
                st.write("#### Key Metrics Visualization")
                # Add sample chart here
                
            if include_statistics:
                st.write("#### Statistical Analysis")
                # Add statistics here
                
            if include_recommendations:
                st.write("#### Recommendations")
                st.markdown("""
                1. Improve First Pass Yield by optimizing process parameters
                2. Investigate root causes of recent quality variations
                3. Update control limits based on recent performance
                """)
            
            # Export options
            st.download_button(
                "Export Report (PDF)",
                "Sample report content",
                file_name="quality_metrics_report.pdf"
            ) 