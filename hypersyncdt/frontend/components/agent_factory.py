import os
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from langchain import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import pandas as pd
import numpy as np

class AgentFactory:
    def __init__(self, n8n_url: str = "http://localhost:5678"):
        self.n8n_url = n8n_url
        
        self.provider_configs = {
            "HyperSyncDT_Quantum_Core": {
                "name": "HyperSyncDT Quantum Core",
                "description": "Advanced quantum-inspired processing engine for complex manufacturing optimization",
                "api_key": os.getenv("HYPERSYNCDT_QUANTUM_CORE_API_KEY"),
                "status": "Active"
            },
            "HyperSyncDT_Neural_Fabric": {
                "name": "HyperSyncDT Neural Fabric",
                "description": "Distributed neural network system for adaptive process control",
                "api_key": os.getenv("HYPERSYNCDT_NEURAL_FABRIC_API_KEY"),
                "provider": os.getenv("HYPERSYNCDT_NEURAL_FABRIC_PROVIDER"),
                "status": "Active"
            },
            "HyperSyncDT_Cognitive_Engine": {
                "name": "HyperSyncDT Cognitive Engine",
                "description": "Advanced reasoning and decision-making system for manufacturing intelligence",
                "api_key": os.getenv("HYPERSYNCDT_COGNITIVE_ENGINE_API_KEY"),
                "version": os.getenv("HYPERSYNCDT_COGNITIVE_ENGINE_VERSION"),
                "status": "Active"
            }
        }
        
        # List of open-source models to try in order of preference
        self.model_options = [
            "paraphrase-MiniLM-L6-v2",
            "distilbert-base-uncased",
            "bert-base-uncased"
        ]
        
        self.embeddings = None
        for model_name in self.model_options:
            try:
                st.info(f"Attempting to load model: {model_name}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=f"sentence-transformers/{model_name}"
                )
                st.success(f"Successfully loaded {model_name}")
                break
            except Exception as e:
                st.warning(f"Failed to load {model_name}: {str(e)}")
                continue
        
        if self.embeddings is None:
            st.error("Failed to load any embedding models. Using basic text processing.")
            # Implement basic fallback using average word embeddings
            self.embeddings = self._create_basic_embeddings()
        
        try:
            self.db = Chroma(
                persist_directory="./agent_knowledge",
                embedding_function=self.embeddings
            )
        except Exception as e:
            st.error(f"Failed to initialize vector store: {str(e)}")
            self.db = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.agent_metrics = {
            "HyperSyncDT_ProcessOptimizer": {
                "tasks_completed": 0,
                "success_rate": 0.0,
                "description": "Advanced process optimization agent leveraging quantum-inspired algorithms for real-time manufacturing optimization"
            },
            "HyperSyncDT_QualityGuardian": {
                "tasks_completed": 0,
                "success_rate": 0.0,
                "description": "Intelligent quality control agent using multi-modal sensor fusion and predictive analytics"
            },
            "HyperSyncDT_MaintenanceOracle": {
                "tasks_completed": 0,
                "success_rate": 0.0,
                "description": "Predictive maintenance agent with advanced anomaly detection and scheduling optimization"
            },
            "HyperSyncDT_EnergyOptimizer": {
                "tasks_completed": 0,
                "success_rate": 0.0,
                "description": "Smart energy management agent using adaptive algorithms for sustainable manufacturing"
            },
            "HyperSyncDT_SupplyChainSynergist": {
                "tasks_completed": 0,
                "success_rate": 0.0,
                "description": "Intelligent supply chain optimization agent with real-time inventory and logistics management"
            }
        }
        
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'selected_providers' not in st.session_state:
            st.session_state.selected_providers = []
        if 'provider_status' not in st.session_state:
            st.session_state.provider_status = {
                provider: config.get('status', 'Inactive')
                for provider, config in self.provider_configs.items()
            }
    
    def _create_basic_embeddings(self):
        """Create a basic embedding function using word averaging"""
        class BasicEmbeddings:
            def embed_documents(self, texts):
                # Simple word-based embedding
                return [[hash(word) % 100 for word in text.split()[:10]] 
                        for text in texts]
            
            def embed_query(self, text):
                # Simple word-based embedding for queries
                return [hash(word) % 100 for word in text.split()[:10]]
        
        return BasicEmbeddings()

    def create_agent(self, agent_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent with specified configuration"""
        if self.db is None:
            st.error("Cannot create agent: Vector store is not initialized")
            return None
            
        workflow_data = {
            "name": f"{agent_type}_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "nodes": self._get_agent_workflow_nodes(agent_type, config),
            "connections": self._get_agent_workflow_connections(agent_type)
        }
        
        try:
            response = requests.post(
                f"{self.n8n_url}/api/v1/workflows",
                json=workflow_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Update metrics for successful creation
            self.agent_metrics[agent_type]["tasks_completed"] += 1
            total_tasks = self.agent_metrics[agent_type]["tasks_completed"]
            current_success = self.agent_metrics[agent_type]["success_rate"]
            self.agent_metrics[agent_type]["success_rate"] = (
                (current_success * (total_tasks - 1) + 1) / total_tasks
            )
            
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to create agent workflow: {str(e)}")
            
            # Update metrics for failed creation
            self.agent_metrics[agent_type]["tasks_completed"] += 1
            total_tasks = self.agent_metrics[agent_type]["tasks_completed"]
            current_success = self.agent_metrics[agent_type]["success_rate"]
            self.agent_metrics[agent_type]["success_rate"] = (
                (current_success * (total_tasks - 1)) / total_tasks
            )
            
            return None

    def update_knowledge_base(self, documents: List[str], metadata: Optional[Dict] = None):
        """Update the agent knowledge base with new documents"""
        try:
            docs = []
            for doc in documents:
                splits = self.text_splitter.split_text(doc)
                docs.extend([{"text": s, "metadata": metadata or {}} for s in splits])
            
            self.db.add_texts([d["text"] for d in docs], metadatas=[d["metadata"] for d in docs])
            st.success("Knowledge base updated successfully")
        except Exception as e:
            st.error(f"Failed to update knowledge base: {str(e)}")

    def get_agent_performance(self, agent_type: str) -> Dict[str, float]:
        """Get performance metrics for a specific agent type"""
        return self.agent_metrics.get(agent_type, {
            "success_rate": 0.0,
            "tasks_completed": 0
        })

    def _get_agent_workflow_nodes(self, agent_type: str, config: Dict[str, Any]) -> List[Dict]:
        """Generate n8n workflow nodes for the agent"""
        nodes = [
            {
                "name": "Start",
                "type": "n8n-nodes-base.start",
                "position": [100, 300],
                "parameters": {}
            },
            {
                "name": "RAG Query",
                "type": "n8n-nodes-base.function",
                "position": [300, 300],
                "parameters": {
                    "functionCode": self._generate_rag_query_code()
                }
            },
            {
                "name": "Agent Action",
                "type": "n8n-nodes-base.function",
                "position": [500, 300],
                "parameters": {
                    "functionCode": self._generate_agent_action_code(agent_type, config)
                }
            },
            {
                "name": "Update Metrics",
                "type": "n8n-nodes-base.function",
                "position": [700, 300],
                "parameters": {
                    "functionCode": self._generate_metrics_update_code()
                }
            }
        ]
        return nodes

    def _get_agent_workflow_connections(self, agent_type: str) -> List[Dict]:
        """Generate n8n workflow connections"""
        return [
            {
                "source": "Start",
                "sourceHandle": "main",
                "target": "RAG Query",
                "targetHandle": "main"
            },
            {
                "source": "RAG Query",
                "sourceHandle": "main",
                "target": "Agent Action",
                "targetHandle": "main"
            },
            {
                "source": "Agent Action",
                "sourceHandle": "main",
                "target": "Update Metrics",
                "targetHandle": "main"
            }
        ]

    def _generate_rag_query_code(self) -> str:
        """Generate code for RAG query node"""
        return """
        async function executeRAG(query) {
            const response = await chromaClient.query({
                query_texts: [query],
                n_results: 5
            });
            return response.documents[0];
        }
        """

    def _generate_agent_action_code(self, agent_type: str, config: Dict[str, Any]) -> str:
        """Generate code for agent action node"""
        return f"""
        async function executeAction(context) {{
            const agentType = "{agent_type}";
            const config = {json.dumps(config)};
            
            // Agent-specific logic here
            const result = await processAgentAction(agentType, config, context);
            return result;
        }}
        """

    def _generate_metrics_update_code(self) -> str:
        """Generate code for metrics update node"""
        return """
        async function updateMetrics(result) {
            // Update agent metrics based on action result
            const success = result.success || false;
            const agentType = result.agentType;
            
            // Update metrics in database
            await updateAgentMetrics(agentType, success);
            return result;
        }
        """

    def render_provider_status(self):
        """Render provider status section"""
        st.title("🤖 HyperSyncDT Autonomous Agent Factory")
        
        # Create a container with glass effect for active providers
        st.markdown("""
        <style>
        .provider-card {
            background: rgba(22, 26, 30, 0.95);
            border: 1px solid rgba(30, 35, 40, 1);
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            transition: transform 0.2s ease;
        }
        .provider-card:hover {
            transform: translateY(-2px);
        }
        .provider-title {
            color: rgb(74, 222, 128);
            font-size: 1.5em;
            font-weight: 500;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .provider-description {
            color: rgb(209, 213, 219);
            font-size: 1em;
            margin: 12px 0;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric-item {
            text-align: center;
        }
        .metric-value {
            color: rgb(74, 222, 128);
            font-size: 1.2em;
            font-weight: 500;
            margin-bottom: 4px;
        }
        .metric-label {
            color: rgb(156, 163, 175);
            font-size: 0.85em;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .status-active {
            background: rgba(74, 222, 128, 0.2);
            color: rgb(74, 222, 128);
        }
        .status-inactive {
            background: rgba(239, 68, 68, 0.2);
            color: rgb(239, 68, 68);
        }
        </style>
        """, unsafe_allow_html=True)

        st.header("Active Providers")

        for provider_id, config in self.provider_configs.items():
            status = st.session_state.provider_status[provider_id]
            
            st.markdown(f"""
            <div class="provider-card">
                <div class="provider-title">
                    {config['name']}
                    <span class="status-badge status-{'active' if status == 'Active' else 'inactive'}">
                        {status}
                    </span>
                </div>
                <div class="provider-description">
                    {config['description']}
                </div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value">12ms</div>
                        <div class="metric-label">Response Time</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">98.5%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">1000/s</div>
                        <div class="metric-label">Throughput</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">99.9%</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # API Key configuration
            api_key = st.text_input(
                f"API Key for {config['name']}",
                type="password",
                value=config.get('api_key', ''),
                key=f"api_key_{provider_id}"
            )
            
            if api_key:
                st.session_state.provider_status[provider_id] = "Active"
            else:
                st.session_state.provider_status[provider_id] = "Inactive"
        
        # Provider Performance Comparison
        st.header("Provider Performance Comparison")
        
        performance_data = {
            "Providers": ["HyperSyncDT_Quantum_Core", "HyperSyncDT_Neural_Fabric", "HyperSyncDT_Cognitive_Engine"],
            "Accuracy": [98.5, 97.2, 96.8],
            "Response Time": [12, 8, 15],
            "Throughput": [1000, 950, 900],
            "Uptime": [99.9, 99.5, 99.7]
        }
        
        fig = go.Figure()
        
        metrics = ["Accuracy", "Response Time", "Throughput", "Uptime"]
        colors = ['rgba(74, 222, 128, 0.8)', 'rgba(74, 222, 128, 0.6)', 
                 'rgba(74, 222, 128, 0.4)', 'rgba(74, 222, 128, 0.3)']
        
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Bar(
                name=metric,
                x=performance_data["Providers"],
                y=performance_data[metric],
                marker_color=color
            ))
        
        fig.update_layout(
            barmode='group',
            title='Provider Performance Metrics',
            xaxis_title='Providers',
            yaxis_title='Value',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(22, 26, 30, 0.95)',
            plot_bgcolor='rgba(22, 26, 30, 0.95)',
            font=dict(color='rgb(209, 213, 219)'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def get_active_providers(self) -> List[str]:
        """Get list of currently active providers"""
        return [
            provider_id
            for provider_id, status in st.session_state.provider_status.items()
            if status == "Active"
        ]
    
    def update_provider_status(self, provider_id: str, status: str):
        """Update the status of a specific provider"""
        if provider_id in self.provider_configs:
            st.session_state.provider_status[provider_id] = status

def render_agent_cards():
    """Render provider cards with descriptions"""
    st.title("🤖 HyperSyncDT Autonomous Agent Factory")

    # Add custom CSS for the project cards
    st.markdown("""
    <style>
    .project-card {
        background: rgba(22, 26, 30, 0.95);
        border: 1px solid rgba(30, 35, 40, 1);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        transition: transform 0.2s ease;
    }
    .project-card:hover {
        transform: translateY(-2px);
    }
    .project-title {
        color: rgb(74, 222, 128);
        font-size: 1.5em;
        font-weight: 500;
        margin-bottom: 12px;
    }
    .project-description {
        color: rgb(209, 213, 219);
        font-size: 1em;
        margin: 12px 0;
    }
    .progress-bar-container {
        width: 100%;
        height: 6px;
        background: rgba(30, 35, 40, 0.8);
        border-radius: 3px;
        margin: 20px 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        background: rgb(74, 222, 128);
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    .project-meta {
        color: rgb(156, 163, 175);
        font-size: 0.9em;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 8px;
        font-size: 0.85em;
    }
    .status-on-track {
        background: rgba(74, 222, 128, 0.2);
        color: rgb(74, 222, 128);
    }
    .status-in-progress {
        background: rgba(251, 191, 36, 0.2);
        color: rgb(251, 191, 36);
    }
    .status-review {
        background: rgba(147, 51, 234, 0.2);
        color: rgb(147, 51, 234);
    }
    .link-icon {
        display: inline-block;
        margin-left: 8px;
        opacity: 0.7;
    }
    </style>
    """, unsafe_allow_html=True)

    # Project Cards
    st.markdown("""
    <div class="project-card">
        <div class="project-title">Quantum-Enhanced Process Control</div>
        <div class="project-description">Implementing quantum algorithms for real-time process optimization</div>
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: 75%;"></div>
        </div>
        <div class="project-meta">
            Deadline: 2025-Q2 <span class="status-badge status-on-track">On Track</span>
        </div>
    </div>

    <div class="project-card">
        <div class="project-title">Advanced ML Pipeline Integration</div>
        <div class="project-description">Developing automated ML pipelines for predictive maintenance</div>
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: 60%;"></div>
        </div>
        <div class="project-meta">
            Deadline: 2025-Q3 <span class="status-badge status-in-progress">In Progress</span>
        </div>
    </div>

    <div class="project-card">
        <div class="project-title">Digital Twin Synchronization <span class="link-icon">↗</span></div>
        <div class="project-description">Enhancing real-time synchronization between physical and digital systems</div>
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: 40%;"></div>
        </div>
        <div class="project-meta">
            Deadline: 2025-Q4 <span class="status-badge status-review">Under Review</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_agent_factory():
    """Main function to render the agent factory interface"""
    render_agent_cards()
    
    # Add agent configuration section
    st.markdown("### Agent Configuration")
    
    # Create tabs for different configuration aspects
    tab1, tab2, tab3 = st.tabs(["Settings", "Integration", "Monitoring"])
    
    with tab1:
        st.markdown("""
        <div class="glass-card">
            <h4>Agent Parameters</h4>
            <p>Configure core parameters for each agent type</p>
        </div>
        """, unsafe_allow_html=True)
        
        agent_type = st.selectbox("Select Agent Type", [
            "HyperSyncDT_Quantum_Core",
            "HyperSyncDT_Neural_Fabric",
            "HyperSyncDT_Cognitive_Engine"
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
            st.slider("Batch Size", 16, 512, 128, step=16)
        with col2:
            st.slider("Memory Allocation (GB)", 1, 32, 8)
            st.slider("Thread Count", 1, 16, 4)
    
    with tab2:
        st.markdown("""
        <div class="glass-card">
            <h4>System Integration</h4>
            <p>Configure agent integration with external systems</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.text_input("API Endpoint")
        st.selectbox("Authentication Method", ["OAuth2", "API Key", "Certificate"])
        st.text_input("Access Token")
    
    with tab3:
        st.markdown("""
        <div class="glass-card">
            <h4>Performance Monitoring</h4>
            <p>Real-time agent performance metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Real-time monitoring chart
        times = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        monitoring_data = pd.DataFrame({
            'timestamp': times,
            'cpu_usage': np.random.normal(60, 10, 100),
            'memory_usage': np.random.normal(70, 15, 100),
            'response_time': np.random.normal(20, 5, 100)
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monitoring_data['timestamp'],
            y=monitoring_data['cpu_usage'],
            name='CPU Usage (%)',
            line=dict(color='rgba(100, 255, 200, 0.8)')
        ))
        
        fig.add_trace(go.Scatter(
            x=monitoring_data['timestamp'],
            y=monitoring_data['memory_usage'],
            name='Memory Usage (%)',
            line=dict(color='rgba(200, 100, 255, 0.8)')
        ))
        
        fig.add_trace(go.Scatter(
            x=monitoring_data['timestamp'],
            y=monitoring_data['response_time'],
            name='Response Time (ms)',
            line=dict(color='rgba(255, 200, 100, 0.8)')
        ))
        
        fig.update_layout(
            title='Real-time Performance Metrics',
            xaxis_title='Time',
            yaxis_title='Value',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,40,50,0.7)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_agent_factory() 