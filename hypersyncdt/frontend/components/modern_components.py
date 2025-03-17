import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

class ModernComponents:
    @staticmethod
    def inject_custom_css():
        """Inject custom CSS for modern UI components."""
        st.markdown("""
            <style>
            .modern-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                border: 1px solid rgba(255, 255, 255, 0.18);
                transition: all 0.3s ease;
                animation: float 6s ease-in-out infinite;
            }
            .modern-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.45);
            }
            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            .card-header h3 {
                color: #ff6b6b;
                margin: 0;
                font-size: 1.2em;
                font-weight: 600;
            }
            .status-badge {
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: 500;
            }
            .status-badge.active {
                background: rgba(46, 213, 115, 0.2);
                color: #2ed573;
            }
            .status-badge.inactive {
                background: rgba(255, 71, 87, 0.2);
                color: #ff4757;
            }
            .card-description {
                color: #a4b0be;
                margin-bottom: 20px;
                font-size: 0.9em;
                line-height: 1.5;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
            }
            .metric {
                text-align: center;
                padding: 10px;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 10px;
            }
            .metric-label {
                display: block;
                color: #a4b0be;
                font-size: 0.8em;
                margin-bottom: 5px;
            }
            .metric-value {
                display: block;
                color: #ff6b6b;
                font-size: 1.1em;
                font-weight: 600;
            }
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
                100% { transform: translateY(0px); }
            }
            .upload-zone {
                border: 2px dashed rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                background: rgba(255, 255, 255, 0.03);
                transition: all 0.3s ease;
            }
            .upload-zone:hover {
                border-color: #ff6b6b;
                background: rgba(255, 107, 107, 0.05);
            }
            .modern-button {
                background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 10px 20px;
                font-weight: 600;
                transition: all 0.3s ease;
                text-align: center;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
            }
            .modern-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
            }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_floating_card(title: str, description: str, metrics: Dict[str, Any], status: str = "active"):
        """Render a modern floating card with metrics."""
        st.markdown(f"""
            <div class="modern-card">
                <div class="card-header">
                    <h3>{title}</h3>
                    <span class="status-badge {status.lower()}">{status}</span>
                </div>
                <p class="card-description">{description}</p>
                <div class="metrics-grid">
                    {ModernComponents._render_metrics(metrics)}
                </div>
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def _render_metrics(metrics: Dict[str, Any]) -> str:
        """Helper method to render metrics HTML."""
        metrics_html = ""
        for label, value in metrics.items():
            metrics_html += f"""
                <div class="metric">
                    <span class="metric-label">{label}</span>
                    <span class="metric-value">{value}</span>
                </div>
            """
        return metrics_html

    @staticmethod
    def render_upload_zone(label: str = "Drop your files here or click to upload"):
        """Render a modern upload zone."""
        st.markdown(f"""
            <div class="upload-zone">
                <p>{label}</p>
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_performance_chart(data: pd.DataFrame, x_col: str, y_cols: list, colors: list):
        """Render a modern performance comparison chart."""
        fig = go.Figure()
        
        for y_col, color in zip(y_cols, colors):
            fig.add_trace(go.Bar(
                name=y_col,
                x=data[x_col],
                y=data[y_col],
                marker_color=color,
                hovertemplate=f"{y_col}: %{{y}}<extra></extra>"
            ))
        
        fig.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor='rgba(255,255,255,0.18)',
                borderwidth=1
            ),
            margin=dict(t=30, l=0, r=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True) 