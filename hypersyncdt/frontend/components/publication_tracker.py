import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

def generate_sample_publication_data() -> Dict[str, pd.DataFrame]:
    """Generate sample publication tracking data."""
    np.random.seed(42)
    
    # Generate publications data
    publications = pd.DataFrame({
        'publication_id': [f'PUB-{i:03d}' for i in range(1, 31)],
        'title': [
            'Digital Twin Framework for Smart Manufacturing',
            'AI-Driven Process Optimization in Industry 4.0',
            'Real-time Monitoring with Digital Twins',
            'Machine Learning for Predictive Maintenance',
            'IoT Integration in Manufacturing Systems',
            'Advanced Process Control Using Digital Twins',
            'Quality Control in Smart Factories',
            'Data-Driven Decision Making in Manufacturing',
            'Sensor Fusion for Process Monitoring',
            'Optimization of Manufacturing Systems'
        ] * 3,
        'status': np.random.choice(
            ['Draft', 'In Review', 'Accepted', 'Published', 'Rejected'],
            30
        ),
        'target_journal': [
            'Journal of Manufacturing Systems',
            'IEEE Transactions on Industrial Informatics',
            'Computers in Industry',
            'Journal of Intelligent Manufacturing',
            'Manufacturing Letters'
        ] * 6,
        'submission_date': pd.date_range(
            start='2023-01-01',
            end='2024-03-17',
            periods=30
        ),
        'impact_factor': np.random.uniform(2.0, 5.0, 30)
    })
    
    # Set publication_id as index
    publications.set_index('publication_id', inplace=True)
    
    # Generate author data
    authors = []
    for pub_id in publications.index:
        num_authors = np.random.randint(2, 5)
        for i in range(num_authors):
            authors.append({
                'publication_id': pub_id,
                'author_name': np.random.choice([
                    'John Smith',
                    'Maria Garcia',
                    'David Chen',
                    'Sarah Johnson',
                    'Michael Brown',
                    'Emma Wilson',
                    'James Taylor',
                    'Lisa Anderson',
                    'Robert Miller',
                    'Jennifer Davis'
                ]),
                'affiliation': np.random.choice([
                    'University A',
                    'Research Institute B',
                    'Industry Partner C',
                    'Technical University D',
                    'Research Center E'
                ]),
                'role': np.random.choice([
                    'First Author',
                    'Co-author',
                    'Corresponding Author'
                ])
            })
    
    authors_df = pd.DataFrame(authors)
    
    # Generate review data
    reviews = []
    for pub_id, pub in publications.iterrows():
        if pub['status'] in ['In Review', 'Accepted', 'Published', 'Rejected']:
            num_reviews = np.random.randint(2, 4)
            for i in range(num_reviews):
                reviews.append({
                    'publication_id': pub_id,
                    'reviewer_id': f'REV-{np.random.randint(1, 100):03d}',
                    'review_date': pub['submission_date'] + pd.Timedelta(days=np.random.randint(30, 90)),
                    'decision': np.random.choice([
                        'Accept',
                        'Minor Revision',
                        'Major Revision',
                        'Reject'
                    ]),
                    'confidence': np.random.randint(1, 6)
                })
    
    reviews_df = pd.DataFrame(reviews)
    
    return {
        'publications': publications,
        'authors': authors_df,
        'reviews': reviews_df
    }

def render_publication_tracker():
    """Render the publication tracker dashboard."""
    st.header("Publication Tracker")
    
    # Initialize components
    digital_twin = SynchronizedDigitalTwin()
    visualizer = MultiModalVisualizer()
    
    # Sidebar controls
    st.sidebar.subheader("Publication Settings")
    publication_status = st.sidebar.multiselect(
        "Publication Status",
        ["Draft", "In Review", "Accepted", "Published", "Rejected"],
        default=["Draft", "In Review", "Accepted"]
    )
    
    time_period = st.sidebar.selectbox(
        "Time Period",
        ["Last 3 Months", "Last 6 Months", "Last Year", "All Time"]
    )
    
    # Generate sample data
    publication_data = generate_sample_publication_data()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Publication Overview",
        "Manuscript Tracking",
        "Review Management",
        "Analytics"
    ])
    
    with tab1:
        st.subheader("Publication Overview")
        
        # Publication status summary
        status_counts = publication_data['publications']['status'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Publications",
                len(publication_data['publications'])
            )
        with col2:
            st.metric(
                "In Review",
                len(publication_data['publications'][
                    publication_data['publications']['status'] == 'In Review'
                ])
            )
        with col3:
            st.metric(
                "Accepted/Published",
                len(publication_data['publications'][
                    publication_data['publications']['status'].isin(['Accepted', 'Published'])
                ])
            )
        with col4:
            acceptance_rate = len(publication_data['publications'][
                publication_data['publications']['status'].isin(['Accepted', 'Published'])
            ]) / len(publication_data['publications']) * 100
            st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
        
        # Publication timeline
        timeline_data = publication_data['publications'].reset_index()
        fig = px.timeline(
            timeline_data,
            x_start='submission_date',
            x_end=timeline_data['submission_date'] + pd.Timedelta(days=90),
            y='publication_id',
            color='status',
            title='Publication Timeline'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Journal distribution
        journal_counts = publication_data['publications']['target_journal'].value_counts()
        
        fig = px.pie(
            values=journal_counts.values,
            names=journal_counts.index,
            title='Distribution by Target Journal'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Manuscript Tracking")
        
        # Manuscript selector
        selected_manuscript = st.selectbox(
            "Select Manuscript",
            publication_data['publications'].index.tolist(),
            format_func=lambda x: publication_data['publications'].loc[x, 'title']
        )
        
        if selected_manuscript:
            # Get manuscript details
            manuscript = publication_data['publications'].loc[selected_manuscript]
            
            # Get authors
            manuscript_authors = publication_data['authors'][
                publication_data['authors']['publication_id'] == selected_manuscript
            ]
            
            # Display manuscript details
            st.write("### Manuscript Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Title:**", manuscript['title'])
                st.write("**Status:**", manuscript['status'])
                st.write("**Target Journal:**", manuscript['target_journal'])
            with col2:
                st.write("**Submission Date:**", manuscript['submission_date'].strftime('%Y-%m-%d'))
                st.write("**Impact Factor:**", f"{manuscript['impact_factor']:.2f}")
            
            # Display authors
            st.write("### Authors")
            st.dataframe(manuscript_authors)
            
            # Display reviews if available
            manuscript_reviews = publication_data['reviews'][
                publication_data['reviews']['publication_id'] == selected_manuscript
            ]
            
            if not manuscript_reviews.empty:
                st.write("### Reviews")
                st.dataframe(manuscript_reviews)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Update Status"):
                    st.info("Updating manuscript status...")
            with col2:
                if st.button("Add Review"):
                    st.info("Adding new review...")
            with col3:
                if st.button("Generate Report"):
                    st.info("Generating manuscript report...")
    
    with tab3:
        st.subheader("Review Management")
        
        # Review statistics
        reviews_df = publication_data['reviews']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Reviews",
                len(reviews_df)
            )
        with col2:
            avg_confidence = reviews_df['confidence'].mean()
            st.metric(
                "Average Confidence",
                f"{avg_confidence:.1f}/5"
            )
        with col3:
            # Calculate average review time using proper indexing
            review_times = reviews_df['review_date'] - publication_data['publications'].loc[reviews_df['publication_id']]['submission_date'].values
            avg_review_time = review_times.mean().days
            st.metric(
                "Average Review Time",
                f"{avg_review_time:.0f} days"
            )
        
        # Review decision distribution
        decision_counts = reviews_df['decision'].value_counts()
        
        fig = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title='Review Decisions Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Review timeline
        review_timeline_data = reviews_df.copy()
        fig = px.scatter(
            review_timeline_data,
            x='review_date',
            y='publication_id',
            color='decision',
            size='confidence',
            title='Review Timeline'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Publication Analytics")
        
        # Publication trends
        pub_data_with_index = publication_data['publications'].reset_index()
        monthly_pubs = pd.DataFrame({
            'date': pub_data_with_index['submission_date'],
            'count': 1
        }).resample('M', on='date').sum()
        
        fig = px.line(
            monthly_pubs,
            x=monthly_pubs.index,
            y='count',
            title='Publication Submissions Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact factor analysis
        st.write("### Impact Factor Analysis")
        
        impact_data = publication_data['publications'].reset_index()
        fig = px.box(
            impact_data,
            x='target_journal',
            y='impact_factor',
            title='Impact Factor Distribution by Journal'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Author collaboration network
        st.write("### Author Collaboration")
        
        author_counts = publication_data['authors']['author_name'].value_counts()
        
        fig = px.bar(
            x=author_counts.index,
            y=author_counts.values,
            title='Author Publication Counts'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.write("### Export Analytics")
        
        report_type = st.selectbox(
            "Report Type",
            [
                "Publication Summary",
                "Review Analysis",
                "Author Statistics",
                "Journal Performance"
            ]
        )
        
        if st.button("Generate Analytics Report"):
            st.info("Generating analytics report...")
            st.download_button(
                "Download Report",
                "Sample report content",
                file_name="publication_analytics.pdf"
            ) 