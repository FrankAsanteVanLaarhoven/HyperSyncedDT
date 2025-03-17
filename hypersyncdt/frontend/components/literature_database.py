import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .digital_twin_components import SynchronizedDigitalTwin
from .advanced_visualizations import MultiModalVisualizer

def generate_sample_literature_data() -> Dict[str, pd.DataFrame]:
    """Generate sample literature database data."""
    np.random.seed(42)
    
    # Generate papers data
    papers = pd.DataFrame({
        'paper_id': [f'PAPER-{i:03d}' for i in range(1, 51)],
        'title': [
            'Digital Twin Framework for Manufacturing',
            'AI-Driven Process Optimization',
            'Smart Factory Implementation',
            'Industry 4.0 Integration Strategies',
            'Machine Learning in Manufacturing',
            'Real-time Process Monitoring',
            'Predictive Maintenance Systems',
            'Quality Control Automation',
            'IoT in Manufacturing',
            'Advanced Process Control'
        ] * 5,
        'authors': [
            'Smith et al.',
            'Johnson et al.',
            'Williams et al.',
            'Brown et al.',
            'Davis et al.',
            'Miller et al.',
            'Wilson et al.',
            'Moore et al.',
            'Taylor et al.',
            'Anderson et al.'
        ] * 5,
        'year': np.random.randint(2018, 2025, 50),
        'citations': np.random.randint(0, 500, 50),
        'relevance_score': np.random.uniform(0.5, 1.0, 50)
    })
    
    # Generate keywords data
    keywords = []
    all_keywords = [
        'Digital Twin',
        'Machine Learning',
        'Process Optimization',
        'Industry 4.0',
        'IoT',
        'Predictive Maintenance',
        'Quality Control',
        'Smart Manufacturing',
        'Real-time Monitoring',
        'Data Analytics'
    ]
    
    for paper_id in papers['paper_id']:
        num_keywords = np.random.randint(2, 5)
        paper_keywords = np.random.choice(all_keywords, num_keywords, replace=False)
        for kw in paper_keywords:
            keywords.append({
                'paper_id': paper_id,
                'keyword': kw
            })
    
    keywords_df = pd.DataFrame(keywords)
    
    # Generate abstracts data (simplified)
    abstracts = pd.DataFrame({
        'paper_id': papers['paper_id'],
        'abstract': [
            f"This paper presents a novel approach to {kw.lower()} "
            f"in manufacturing environments..."
            for kw in np.random.choice(all_keywords, len(papers))
        ]
    })
    
    return {
        'papers': papers,
        'keywords': keywords_df,
        'abstracts': abstracts
    }

def render_literature_database():
    """Render the literature database dashboard."""
    st.header("Literature Database")
    
    # Initialize components
    digital_twin = SynchronizedDigitalTwin()
    visualizer = MultiModalVisualizer()
    
    # Sidebar controls
    st.sidebar.subheader("Search Parameters")
    search_type = st.sidebar.selectbox(
        "Search Type",
        ["Keyword", "Author", "Year", "Citation Count"]
    )
    
    sort_by = st.sidebar.selectbox(
        "Sort By",
        ["Relevance", "Year", "Citations", "Title"]
    )
    
    # Generate sample data
    literature_data = generate_sample_literature_data()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Search & Browse",
        "Analytics",
        "Paper Details",
        "Export"
    ])
    
    with tab1:
        st.subheader("Literature Search")
        
        # Search box
        search_query = st.text_input("Search Query")
        
        # Advanced filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year_range = st.slider(
                "Year Range",
                min_value=2018,
                max_value=2024,
                value=(2018, 2024)
            )
        
        with col2:
            min_citations = st.number_input(
                "Minimum Citations",
                min_value=0,
                value=0
            )
        
        with col3:
            selected_keywords = st.multiselect(
                "Keywords",
                literature_data['keywords']['keyword'].unique()
            )
        
        # Filter papers based on criteria
        filtered_papers = literature_data['papers']
        
        if year_range:
            filtered_papers = filtered_papers[
                (filtered_papers['year'] >= year_range[0]) &
                (filtered_papers['year'] <= year_range[1])
            ]
        
        if min_citations > 0:
            filtered_papers = filtered_papers[
                filtered_papers['citations'] >= min_citations
            ]
        
        if selected_keywords:
            relevant_papers = literature_data['keywords'][
                literature_data['keywords']['keyword'].isin(selected_keywords)
            ]['paper_id'].unique()
            filtered_papers = filtered_papers[
                filtered_papers['paper_id'].isin(relevant_papers)
            ]
        
        # Display results
        st.write(f"### Search Results ({len(filtered_papers)} papers)")
        
        for _, paper in filtered_papers.iterrows():
            # Get paper keywords
            paper_keywords = literature_data['keywords'][
                literature_data['keywords']['paper_id'] == paper['paper_id']
            ]['keyword'].tolist()
            
            # Get paper abstract
            abstract = literature_data['abstracts'][
                literature_data['abstracts']['paper_id'] == paper['paper_id']
            ]['abstract'].iloc[0]
            
            st.markdown(f"""
            <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0;'>
                <h4>{paper['title']}</h4>
                <p><strong>Authors:</strong> {paper['authors']} ({paper['year']})</p>
                <p><strong>Citations:</strong> {paper['citations']}</p>
                <p><strong>Keywords:</strong> {', '.join(paper_keywords)}</p>
                <p><em>Abstract:</em> {abstract[:200]}...</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Literature Analytics")
        
        # Publications per year
        yearly_pubs = filtered_papers.groupby('year').size().reset_index(
            name='count'
        )
        
        fig = px.bar(
            yearly_pubs,
            x='year',
            y='count',
            title='Publications per Year'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Keyword analysis
        keyword_counts = literature_data['keywords']['keyword'].value_counts()
        
        fig = px.pie(
            values=keyword_counts.values,
            names=keyword_counts.index,
            title='Keyword Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Citation analysis
        st.write("### Citation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Average Citations",
                f"{filtered_papers['citations'].mean():.1f}"
            )
        
        with col2:
            st.metric(
                "Total Citations",
                str(filtered_papers['citations'].sum())
            )
        
        # Citation distribution
        fig = px.histogram(
            filtered_papers,
            x='citations',
            title='Citation Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Paper Details")
        
        selected_paper = st.selectbox(
            "Select Paper",
            filtered_papers['paper_id'].tolist(),
            format_func=lambda x: filtered_papers[
                filtered_papers['paper_id'] == x
            ]['title'].iloc[0]
        )
        
        if selected_paper:
            paper = filtered_papers[
                filtered_papers['paper_id'] == selected_paper
            ].iloc[0]
            
            st.write("### " + paper['title'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Authors:**", paper['authors'])
                st.write("**Year:**", paper['year'])
                st.write("**Citations:**", paper['citations'])
            
            with col2:
                paper_keywords = literature_data['keywords'][
                    literature_data['keywords']['paper_id'] == paper['paper_id']
                ]['keyword'].tolist()
                st.write("**Keywords:**", ", ".join(paper_keywords))
                st.write("**Relevance Score:**", f"{paper['relevance_score']:.2f}")
            
            st.write("### Abstract")
            abstract = literature_data['abstracts'][
                literature_data['abstracts']['paper_id'] == paper['paper_id']
            ]['abstract'].iloc[0]
            st.write(abstract)
            
            # Related papers
            st.write("### Related Papers")
            
            # Find papers with similar keywords
            related_papers = []
            for kw in paper_keywords:
                related_paper_ids = literature_data['keywords'][
                    (literature_data['keywords']['keyword'] == kw) &
                    (literature_data['keywords']['paper_id'] != paper['paper_id'])
                ]['paper_id'].unique()
                
                for pid in related_paper_ids:
                    paper_data = filtered_papers[
                        filtered_papers['paper_id'] == pid
                    ].iloc[0]
                    related_papers.append({
                        'title': paper_data['title'],
                        'authors': paper_data['authors'],
                        'year': paper_data['year'],
                        'shared_keywords': len(set(paper_keywords) & set(
                            literature_data['keywords'][
                                literature_data['keywords']['paper_id'] == pid
                            ]['keyword'].tolist()
                        ))
                    })
            
            if related_papers:
                related_df = pd.DataFrame(related_papers)
                related_df = related_df.sort_values(
                    'shared_keywords',
                    ascending=False
                ).head(5)
                st.dataframe(related_df)
    
    with tab4:
        st.subheader("Export Options")
        
        # Export format selection
        export_format = st.selectbox(
            "Export Format",
            ["BibTeX", "CSV", "JSON", "RIS"]
        )
        
        # Export options
        st.write("### Export Options")
        include_abstracts = st.checkbox("Include Abstracts", value=True)
        include_keywords = st.checkbox("Include Keywords", value=True)
        include_citations = st.checkbox("Include Citation Count", value=True)
        
        # Export button
        if st.button("Export Selected Papers"):
            st.success(f"Exported {len(filtered_papers)} papers in {export_format} format")
            
            # Download button
            st.download_button(
                "Download Export File",
                "Sample export content",
                file_name=f"literature_export.{export_format.lower()}"
            ) 