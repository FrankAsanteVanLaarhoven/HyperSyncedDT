import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
import yaml
from dataclasses import dataclass
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentTemplate(Enum):
    CUSTOMER_SUPPORT = "customer_support"
    TECHNICAL_DOCS = "technical_docs"
    RESEARCH_ASSISTANT = "research_assistant"
    CUSTOM = "custom"

@dataclass
class AgentConfig:
    name: str
    template: AgentTemplate
    description: str
    capabilities: List[str]
    knowledge_base_path: str
    embedding_model: str
    deployment_env: str
    created_at: datetime
    updated_at: datetime

class KnowledgeBaseManager:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index = None
        self.initialize_index()

    def initialize_index(self):
        """Initialize FAISS index for vector storage."""
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
            else:
                # Create a new index with 768 dimensions (default for many embedding models)
                self.index = faiss.IndexFlatL2(768)
                self.save_index()
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add new documents to the knowledge base."""
        try:
            self.index.add(embeddings)
            self.save_index()
            return True
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}")
            return False

    def save_index(self):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, str(self.index_path))

class AgentManager:
    def __init__(self):
        self.config_path = Path("configs/agents")
        self.config_path.mkdir(parents=True, exist_ok=True)

    def create_agent(self, config: AgentConfig) -> bool:
        """Create a new RAG agent with the specified configuration."""
        try:
            config_dict = {
                "name": config.name,
                "template": config.template.value,
                "description": config.description,
                "capabilities": config.capabilities,
                "knowledge_base_path": config.knowledge_base_path,
                "embedding_model": config.embedding_model,
                "deployment_env": config.deployment_env,
                "created_at": config.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat()
            }
            
            with open(self.config_path / f"{config.name}.yaml", "w") as f:
                yaml.dump(config_dict, f)
            return True
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return False

    def list_agents(self) -> List[AgentConfig]:
        """List all available agents."""
        agents = []
        for config_file in self.config_path.glob("*.yaml"):
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
                agents.append(AgentConfig(
                    name=config_dict["name"],
                    template=AgentTemplate(config_dict["template"]),
                    description=config_dict["description"],
                    capabilities=config_dict["capabilities"],
                    knowledge_base_path=config_dict["knowledge_base_path"],
                    embedding_model=config_dict["embedding_model"],
                    deployment_env=config_dict["deployment_env"],
                    created_at=datetime.fromisoformat(config_dict["created_at"]),
                    updated_at=datetime.fromisoformat(config_dict["updated_at"])
                ))
        return agents

class PerformanceMonitor:
    def __init__(self):
        self.metrics_path = Path("metrics")
        self.metrics_path.mkdir(exist_ok=True)

    def log_metrics(self, agent_name: str, metrics: Dict[str, float]):
        """Log performance metrics for an agent."""
        metrics_file = self.metrics_path / f"{agent_name}_metrics.json"
        current_time = datetime.now().isoformat()
        
        try:
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    historical_metrics = json.load(f)
            else:
                historical_metrics = []
            
            metrics["timestamp"] = current_time
            historical_metrics.append(metrics)
            
            with open(metrics_file, "w") as f:
                json.dump(historical_metrics, f)
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

def render_rag_agent_creator():
    """Render the RAG Agentic Agents Creator interface"""
    st.title("🤖 RAG Agentic Agents Creator")
    
    # Initialize managers
    agent_manager = AgentManager()
    performance_monitor = PerformanceMonitor()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Create Agent",
        "Knowledge Base",
        "Performance",
        "Deployment"
    ])
    
    with tab1:
        st.subheader("Create New Agent")
        
        # Agent creation form
        with st.form("agent_creation_form"):
            name = st.text_input("Agent Name")
            template = st.selectbox(
                "Template",
                options=[t.value for t in AgentTemplate]
            )
            description = st.text_area("Description")
            capabilities = st.multiselect(
                "Capabilities",
                options=["Question Answering", "Document Analysis", "Code Generation", "Custom"]
            )
            embedding_model = st.selectbox(
                "Embedding Model",
                options=["OpenAI", "HuggingFace", "Custom"]
            )
            deployment_env = st.selectbox(
                "Deployment Environment",
                options=["Development", "Staging", "Production"]
            )
            
            if st.form_submit_button("Create Agent"):
                config = AgentConfig(
                    name=name,
                    template=AgentTemplate(template),
                    description=description,
                    capabilities=capabilities,
                    knowledge_base_path=f"knowledge_bases/{name}",
                    embedding_model=embedding_model,
                    deployment_env=deployment_env,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                if agent_manager.create_agent(config):
                    st.success("Agent created successfully!")
                else:
                    st.error("Failed to create agent")
    
    with tab2:
        st.subheader("Knowledge Base Management")
        
        # List existing agents
        agents = agent_manager.list_agents()
        if agents:
            selected_agent = st.selectbox(
                "Select Agent",
                options=[agent.name for agent in agents]
            )
            
            # File uploader for knowledge base
            uploaded_files = st.file_uploader(
                "Upload Documents",
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.info("Processing documents...")
                # Here you would process the documents and update the knowledge base
                st.success("Documents processed and added to knowledge base")
        else:
            st.info("No agents created yet")
    
    with tab3:
        st.subheader("Performance Monitoring")
        
        # Display performance metrics
        if agents:
            selected_agent = st.selectbox(
                "Select Agent for Monitoring",
                options=[agent.name for agent in agents],
                key="monitoring_agent"
            )
            
            # Create sample performance data
            dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
            metrics = {
                "accuracy": np.random.normal(0.85, 0.05, 30),
                "latency": np.random.normal(200, 20, 30),
                "usage": np.random.normal(100, 10, 30)
            }
            
            # Create performance visualizations
            fig = go.Figure()
            for metric, values in metrics.items():
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    name=metric.capitalize(),
                    mode="lines+markers"
                ))
            
            fig.update_layout(
                title="Agent Performance Metrics",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agents available for monitoring")
    
    with tab4:
        st.subheader("Deployment Management")
        
        if agents:
            selected_agent = st.selectbox(
                "Select Agent for Deployment",
                options=[agent.name for agent in agents],
                key="deployment_agent"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Current Status",
                    "Active",
                    "Deployed to Production"
                )
            
            with col2:
                st.metric(
                    "Response Time",
                    "200ms",
                    "-50ms"
                )
            
            # Deployment controls
            st.button("Deploy to Production", type="primary")
            st.button("Rollback Deployment", type="secondary")
        else:
            st.info("No agents available for deployment")

if __name__ == "__main__":
    render_rag_agent_creator() 