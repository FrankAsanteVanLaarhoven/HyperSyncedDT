import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import json
import os

class RAGAssistant:
    def __init__(self):
        self._initialize_session_state()
        self.setup_models()
        
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'knowledge_base' not in st.session_state:
            st.session_state.knowledge_base = self._load_knowledge_base()
        if 'embeddings_index' not in st.session_state:
            st.session_state.embeddings_index = None
            
    def _load_knowledge_base(self):
        """Load the knowledge base from JSON files"""
        knowledge_base = {
            'documentation': [],
            'technical_specs': [],
            'troubleshooting': [],
            'best_practices': []
        }
        
        # Sample data - in production, this would load from actual files
        knowledge_base['documentation'] = [
            {
                'title': 'Tool Condition Monitoring',
                'content': 'Comprehensive guide to monitoring tool conditions including vibration analysis, temperature monitoring, and acoustic emission tracking.',
                'category': 'technical'
            },
            {
                'title': 'Predictive Maintenance',
                'content': 'Best practices for implementing predictive maintenance using machine learning and sensor data.',
                'category': 'maintenance'
            }
        ]
        
        return knowledge_base
        
    def setup_models(self):
        """Initialize the embedding model and RAG components"""
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create FAISS index for fast similarity search
        self.build_embeddings_index()
        
    def build_embeddings_index(self):
        """Build FAISS index from knowledge base"""
        # Combine all documents
        documents = []
        for category in st.session_state.knowledge_base.values():
            documents.extend(category)
        
        # Generate embeddings
        texts = [doc['content'] for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        st.session_state.embeddings_index = {
            'index': index,
            'documents': documents
        }
        
    def search_knowledge_base(self, query, k=3):
        """Search knowledge base using semantic similarity"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS index
        D, I = st.session_state.embeddings_index['index'].search(
            query_embedding.astype('float32'),
            k
        )
        
        # Get relevant documents
        results = []
        for idx in I[0]:
            doc = st.session_state.embeddings_index['documents'][idx]
            results.append({
                'title': doc['title'],
                'content': doc['content'],
                'category': doc['category'],
                'relevance': float(D[0][len(results)])
            })
            
        return results
        
    def generate_response(self, query, context):
        """Generate response using retrieved context"""
        # In production, this would use a language model
        # For now, return a template response
        response = f"Based on the available information about {context[0]['title']}, "
        response += f"I can tell you that {context[0]['content']}"
        return response
        
    def add_to_chat_history(self, role, content):
        """Add message to chat history"""
        st.session_state.chat_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })
        
    def clear_chat_history(self):
        """Clear the chat history"""
        st.session_state.chat_history = []

def render_rag_assistant():
    """Render the RAG Assistant interface"""
    st.title("AI Assistant")
    
    # Initialize RAG assistant if not in session state
    if 'rag_assistant' not in st.session_state:
        st.session_state.rag_assistant = RAGAssistant()
    
    # Create sidebar for settings
    with st.sidebar:
        st.markdown("### Assistant Settings")
        
        # Knowledge base settings
        st.markdown("#### Knowledge Base")
        kb_categories = st.multiselect(
            "Active Categories",
            ["Documentation", "Technical Specs", "Troubleshooting", "Best Practices"],
            default=["Documentation", "Technical Specs"]
        )
        
        # Search settings
        st.markdown("#### Search Settings")
        num_results = st.slider(
            "Number of results",
            min_value=1,
            max_value=5,
            value=3
        )
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.rag_assistant.clear_chat_history()
            st.rerun()
    
    # Main chat interface
    st.markdown("### Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: flex-end;
                    margin: 10px 0;
                ">
                    <div style="
                        background: rgba(100, 255, 200, 0.1);
                        padding: 10px;
                        border-radius: 10px;
                        max-width: 80%;
                    ">
                        <p style="margin: 0;">{message['content']}</p>
                        <small style="color: rgba(255,255,255,0.5);">
                            {message['timestamp'].strftime('%H:%M')}
                        </small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: flex-start;
                    margin: 10px 0;
                ">
                    <div style="
                        background: rgba(255, 255, 255, 0.1);
                        padding: 10px;
                        border-radius: 10px;
                        max-width: 80%;
                    ">
                        <p style="margin: 0;">{message['content']}</p>
                        <small style="color: rgba(255,255,255,0.5);">
                            {message['timestamp'].strftime('%H:%M')}
                        </small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("### Ask a Question")
    with st.form(key='query_form'):
        query = st.text_area("Type your question here")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and query:
            # Add user message to chat
            st.session_state.rag_assistant.add_to_chat_history('user', query)
            
            # Search knowledge base
            results = st.session_state.rag_assistant.search_knowledge_base(
                query,
                k=num_results
            )
            
            # Generate response
            response = st.session_state.rag_assistant.generate_response(query, results)
            
            # Add assistant response to chat
            st.session_state.rag_assistant.add_to_chat_history('assistant', response)
            
            # Show relevant sources
            st.markdown("### Relevant Sources")
            for result in results:
                st.markdown(f"""
                <div style="
                    background: rgba(255, 255, 255, 0.05);
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                ">
                    <strong>{result['title']}</strong><br>
                    <small>Category: {result['category']}</small><br>
                    <p>{result['content']}</p>
                    <small>Relevance: {result['relevance']:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.rerun()

if __name__ == "__main__":
    render_rag_assistant() 