import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

class ScientificLiteratureManager:
    def __init__(self):
        self.sample_papers = self._generate_sample_papers()
        self.sample_citations = self._generate_sample_citations()
        
    def _generate_sample_papers(self) -> pd.DataFrame:
        """Generate sample scientific papers data."""
        titles = [
            "Advanced Digital Twin Applications in Manufacturing",
            "Machine Learning for Predictive Maintenance",
            "Real-time Process Optimization Using AI",
            "Smart Factory Integration Patterns",
            "Industry 4.0 Implementation Strategies"
        ]
        
        authors = [
            "Smith et al.",
            "Johnson et al.",
            "Zhang et al.",
            "Brown et al.",
            "Wilson et al."
        ]
        
        journals = [
            "Journal of Manufacturing Technology",
            "AI in Industry",
            "Smart Manufacturing Systems",
            "Digital Twin Research",
            "Industry 4.0 Journal"
        ]
        
        dates = pd.date_range(end=datetime.now(), periods=5, freq='M')
        
        return pd.DataFrame({
            'title': titles,
            'authors': authors,
            'journal': journals,
            'publication_date': dates,
            'citations': np.random.randint(10, 200, 5),
            'impact_factor': np.random.uniform(2.0, 8.0, 5),
            'relevance_score': np.random.uniform(0.7, 0.99, 5)
        })
        
    def _generate_sample_citations(self) -> pd.DataFrame:
        """Generate sample citation network data."""
        papers = range(5)
        citations = []
        
        for paper in papers:
            for _ in range(np.random.randint(2, 6)):
                citing_paper = np.random.choice(papers)
                if citing_paper != paper:
                    citations.append({
                        'citing_paper': citing_paper,
                        'cited_paper': paper,
                        'citation_year': np.random.randint(2020, 2024)
                    })
                    
        return pd.DataFrame(citations)

def render_scientific_literature():
    """Render the scientific literature analysis interface."""
    st.header("Scientific Literature Analysis", divider="rainbow")
    
    # Initialize manager
    manager = ScientificLiteratureManager()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Literature Controls")
        
        search_query = st.text_input("Search Papers", placeholder="Enter keywords...")
        
        st.divider()
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=365), datetime.now())
        )
        
        min_citations = st.slider("Minimum Citations", 0, 200, 10)
        min_impact = st.slider("Minimum Impact Factor", 0.0, 10.0, 2.0)
        
        st.divider()
        if st.button("Export Bibliography", use_container_width=True):
            st.info("Exporting bibliography...")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs([
        "Paper Overview",
        "Citation Network",
        "Journal Analysis"
    ])
    
    with tab1:
        st.subheader("Recent Publications")
        
        # Paper metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Papers", "157")
        with col2:
            st.metric("Average Citations", "45.3")
        with col3:
            st.metric("h-index", "23")
        
        # Papers table
        st.dataframe(
            manager.sample_papers.style.background_gradient(subset=['relevance_score']),
            hide_index=True,
            use_container_width=True
        )
        
        # Citation trends
        fig = px.line(
            manager.sample_papers,
            x='publication_date',
            y='citations',
            title='Citation Trends Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Citation Network Analysis")
        
        # Citation statistics
        col1, col2 = st.columns(2)
        with col1:
            # Citation distribution
            citation_dist = pd.DataFrame({
                'Citations': ['0-10', '11-50', '51-100', '100+'],
                'Count': np.random.randint(10, 50, 4)
            })
            
            fig = px.bar(
                citation_dist,
                x='Citations',
                y='Count',
                title='Citation Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top citing journals
            top_journals = pd.DataFrame({
                'Journal': manager.sample_papers['journal'].unique(),
                'Citations': np.random.randint(50, 200, 5)
            }).sort_values('Citations', ascending=True)
            
            fig = px.bar(
                top_journals,
                x='Citations',
                y='Journal',
                title='Top Citing Journals',
                orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Journal Impact Analysis")
        
        # Journal metrics
        journal_metrics = pd.DataFrame({
            'Journal': manager.sample_papers['journal'].unique(),
            'Impact Factor': np.random.uniform(2.0, 8.0, 5),
            'H5 Index': np.random.randint(20, 80, 5),
            'Papers Published': np.random.randint(100, 1000, 5)
        })
        
        # Impact factor visualization
        fig = px.scatter(
            journal_metrics,
            x='Impact Factor',
            y='H5 Index',
            size='Papers Published',
            hover_data=['Journal'],
            title='Journal Impact Metrics'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Journal rankings
        st.subheader("Journal Rankings")
        st.dataframe(
            journal_metrics.sort_values('Impact Factor', ascending=False),
            hide_index=True,
            use_container_width=True
        )

if __name__ == "__main__":
    render_scientific_literature() 