import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import plotly.express as px

class RAGAssistant:
    def __init__(self):
        self.sample_queries = [
            "How to optimize tool wear parameters?",
            "What are the current machine learning models in use?",
            "Show me the latest maintenance reports",
            "Explain the quality control process",
            "What are the key performance indicators?"
        ]
        
    def generate_sample_responses(self) -> List[Dict]:
        """Generate sample responses for demonstration."""
        responses = []
        for query in self.sample_queries:
            responses.append({
                "query": query,
                "timestamp": datetime.now(),
                "response": f"Sample response for: {query}",
                "confidence": np.random.uniform(0.7, 0.99),
                "sources": [f"Document_{i}" for i in range(1, 4)],
                "processing_time": np.random.uniform(0.1, 2.0)
            })
        return responses

def render_rag_assistant():
    """Render the RAG Assistant interface."""
    st.header("RAG Assistant", divider="rainbow")
    
    # Initialize the assistant
    assistant = RAGAssistant()
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Assistant Settings")
        
        st.slider("Response Length", 50, 500, 200)
        st.slider("Temperature", 0.0, 1.0, 0.7)
        st.multiselect(
            "Knowledge Sources",
            ["Technical Docs", "Maintenance Logs", "Research Papers", "Process Data"],
            default=["Technical Docs", "Maintenance Logs"]
        )
        
        st.divider()
        st.checkbox("Enable Real-time Processing", value=True)
        st.checkbox("Include Source Citations", value=True)
    
    # Main chat interface
    st.subheader("Interactive Assistant")
    
    # Query input
    user_query = st.text_area("Enter your query:", placeholder="Ask me anything about the system...")
    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button("Submit Query", use_container_width=True):
            st.info("Query processing... (Demo Mode)")
            
    with col2:
        st.button("Clear", use_container_width=True)
    
    # Sample responses visualization
    responses = assistant.generate_sample_responses()
    
    # Display sample interaction
    st.divider()
    st.subheader("Recent Interactions")
    
    for resp in responses:
        with st.expander(f"Q: {resp['query']}", expanded=False):
            st.write(resp['response'])
            st.caption(f"Confidence: {resp['confidence']:.2%} | Processing Time: {resp['processing_time']:.2f}s")
            st.caption(f"Sources: {', '.join(resp['sources'])}")
    
    # Performance metrics
    st.divider()
    st.subheader("Assistant Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Response Time", "0.8s")
    with col2:
        st.metric("Average Confidence", "92%")
    with col3:
        st.metric("Queries Processed", "157")
    
    # Performance over time visualization
    performance_data = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=24, freq='H'),
        'response_time': np.random.normal(0.8, 0.2, 24),
        'confidence': np.random.normal(0.92, 0.05, 24)
    })
    
    fig = px.line(
        performance_data,
        x='timestamp',
        y=['response_time', 'confidence'],
        title='Assistant Performance Metrics Over Time'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Knowledge base stats
    st.subheader("Knowledge Base Statistics")
    kb_stats = pd.DataFrame({
        'Source': ['Technical Docs', 'Maintenance Logs', 'Research Papers', 'Process Data'],
        'Documents': np.random.randint(100, 1000, 4),
        'Last Updated': pd.date_range(end=datetime.now(), periods=4, freq='D')
    })
    st.dataframe(kb_stats, use_container_width=True)

if __name__ == "__main__":
    render_rag_assistant() 