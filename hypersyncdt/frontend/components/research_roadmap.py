import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import altair as alt
import json
from datetime import datetime, timedelta, date
import random
import time
# New imports for 3D visualization and Redis
import plotly.io as pio

# Handle optional plotly events import
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except ImportError:
    HAS_PLOTLY_EVENTS = False
    def plotly_events(fig, *args, **kwargs):
        st.warning("Interactive events are disabled. Install streamlit-plotly-events for full functionality.")
        st.plotly_chart(fig, use_container_width=True)
        return []
import requests
import base64
import re  # Add import for regular expressions
try:
    import redis
except ImportError:
    redis = None

# Define a class for the research roadmap
class ResearchRoadmap:
    def __init__(self):
        self.timeline = []
        self.notes = []
        self.collaborations = []
        self.media_tools = []

    def add_milestone(self, date_obj, description):
        self.timeline.append({'date': date_obj, 'description': description})

    def add_note(self, note):
        self.notes.append(note)

    def add_collaboration(self, team, description):
        self.collaborations.append({'team': team, 'description': description})

    def add_media_tool(self, tool_name, description):
        self.media_tools.append({'tool_name': tool_name, 'description': description})

# Initialize the research roadmap
roadmap = ResearchRoadmap()

# Add initial milestones
roadmap.add_milestone(date(2024, 1, 1), "Project Kickoff")
roadmap.add_milestone(date(2024, 6, 1), "First Publication Submission")
roadmap.add_milestone(date(2025, 1, 1), "Patent Application")
roadmap.add_milestone(date(2025, 6, 1), "Conference Presentation")
roadmap.add_milestone(date(2026, 1, 1), "Final Report and Deployment")

# Add collaboration tools
roadmap.add_collaboration("Notion", "Note-taking and collaboration platform with shareable notes.")
roadmap.add_collaboration("Teams", "Live chat and meeting platform for team collaboration.")
roadmap.add_collaboration("Slack", "Messaging app for team communication and collaboration.")

# Add media tools
roadmap.add_media_tool("Screen Recorder", "Capture screen activities for documentation.")
roadmap.add_media_tool("Speech to Text", "Convert spoken words into text for easy documentation.")
roadmap.add_media_tool("Video Streaming", "Stream and record video meetings.")

# PhD Research Roadmap data
def load_phd_roadmap_data():
    # This would normally load from a JSON file, but for now we'll define it inline
    roadmap_data = {
        "timeline": [
            {
                "Task": "Literature Review",
                "Start": "2025-01-01",
                "Finish": "2025-03-31",
                "Resource": "Research",
                "Description": "Comprehensive review of digital twin technologies",
                "Complete": 0
            },
            {
                "Task": "Methodology Development",
                "Start": "2025-04-01",
                "Finish": "2025-06-30",
                "Resource": "Research",
                "Description": "Development of machine tool dynamics-based digital twin methodology",
                "Complete": 0
            },
            {
                "Task": "First Prototype Implementation",
                "Start": "2025-07-01",
                "Finish": "2025-09-30",
                "Resource": "Development",
                "Description": "Implementation of first prototype of the digital twin system",
                "Complete": 0
            },
            {
                "Task": "Initial Testing",
                "Start": "2025-10-01",
                "Finish": "2025-12-31",
                "Resource": "Testing",
                "Description": "Testing of the initial digital twin prototype",
                "Complete": 0
            },
            {
                "Task": "First Publication Development",
                "Start": "2025-07-01",
                "Finish": "2025-09-30",
                "Resource": "Publications",
                "Description": "Preparation of first journal article on digital twin methodology",
                "Complete": 0
            },
            {
                "Task": "Industry Partner Engagement",
                "Start": "2025-01-01",
                "Finish": "2025-06-30",
                "Resource": "Stakeholder Engagement",
                "Description": "Engagement with industry partners for requirements gathering",
                "Complete": 0
            },
            {
                "Task": "Enhanced Algorithm Development",
                "Start": "2026-01-01",
                "Finish": "2026-06-30",
                "Resource": "Research",
                "Description": "Development of enhanced algorithms for the digital twin",
                "Complete": 0
            },
            {
                "Task": "System Integration",
                "Start": "2026-07-01",
                "Finish": "2026-12-31",
                "Resource": "Development",
                "Description": "Integration of the digital twin with manufacturing systems",
                "Complete": 0
            },
            {
                "Task": "Performance Evaluation",
                "Start": "2027-01-01",
                "Finish": "2027-03-31",
                "Resource": "Testing",
                "Description": "Comprehensive evaluation of digital twin performance",
                "Complete": 0
            },
            {
                "Task": "Second Publication Development",
                "Start": "2026-07-01",
                "Finish": "2026-12-31",
                "Resource": "Publications",
                "Description": "Preparation of second journal article on system integration",
                "Complete": 0
            },
            {
                "Task": "Patent Application Preparation",
                "Start": "2027-01-01",
                "Finish": "2027-06-30",
                "Resource": "Publications",
                "Description": "Preparation and submission of patent application",
                "Complete": 0
            },
            {
                "Task": "Final Framework Development",
                "Start": "2027-07-01",
                "Finish": "2028-06-30",
                "Resource": "Development",
                "Description": "Development of the final digital twin framework",
                "Complete": 0
            },
            {
                "Task": "Validation Studies",
                "Start": "2028-07-01",
                "Finish": "2028-12-31",
                "Resource": "Testing",
                "Description": "Final validation of the digital twin framework",
                "Complete": 0
            },
            {
                "Task": "Dissertation Writing",
                "Start": "2028-07-01",
                "Finish": "2029-03-31",
                "Resource": "Publications",
                "Description": "Writing and submission of PhD dissertation",
                "Complete": 0
            }
        ],
        "colors": {
            "Research": "rgb(46, 137, 205)",
            "Development": "rgb(114, 44, 121)",
            "Testing": "rgb(198, 47, 105)",
            "Publications": "rgb(58, 149, 136)",
            "Stakeholder Engagement": "rgb(107, 127, 135)"
        },
        "milestones": [
            {
                "title": "Initial Framework Design",
                "date": "Q1 2025",
                "description": "Complete the initial design of the digital twin framework",
                "completion_percentage": 0,
                "status": "Planned",
                "deliverables": ["Framework documentation", "Initial design diagrams"]
            },
            {
                "title": "First Prototype",
                "date": "Q3 2025",
                "description": "Complete the first prototype of the digital twin system",
                "completion_percentage": 0,
                "status": "Planned",
                "deliverables": ["Prototype software", "Technical documentation"]
            },
            {
                "title": "First Journal Publication",
                "date": "Q4 2025",
                "description": "Submit first journal article on digital twin methodology",
                "completion_percentage": 0,
                "status": "Planned",
                "deliverables": ["Journal manuscript", "Supporting data"]
            },
            {
                "title": "Enhanced Algorithm Implementation",
                "date": "Q2 2026",
                "description": "Complete implementation of enhanced algorithms",
                "completion_percentage": 0,
                "status": "Planned",
                "deliverables": ["Algorithm documentation", "Performance analysis"]
            },
            {
                "title": "System Integration Completed",
                "date": "Q4 2026",
                "description": "Complete integration with manufacturing systems",
                "completion_percentage": 0,
                "status": "Planned",
                "deliverables": ["Integrated system", "Integration report"]
            },
            {
                "title": "Patent Application",
                "date": "Q2 2027",
                "description": "Submit patent application for novel digital twin aspects",
                "completion_percentage": 0,
                "status": "Planned",
                "deliverables": ["Patent application", "Technical drawings"]
            },
            {
                "title": "Final Framework",
                "date": "Q2 2028",
                "description": "Complete the final digital twin framework",
                "completion_percentage": 0,
                "status": "Planned",
                "deliverables": ["Final framework", "User documentation"]
            },
            {
                "title": "PhD Dissertation",
                "date": "Q1 2029",
                "description": "Complete and submit PhD dissertation",
                "completion_percentage": 0,
                "status": "Planned",
                "deliverables": ["Dissertation", "Defense presentation"]
            }
        ],
        "publications": [
            {
                "title": "Novel Digital Twin Approach for Tool Condition Monitoring",
                "type": "Journal Article",
                "journal": "Journal of Manufacturing Systems",
                "target_date": "2025-09-30",
                "status": "Planned"
            },
            {
                "title": "Machine Learning Integration in Digital Twins for Smart Manufacturing",
                "type": "Conference Paper",
                "journal": "International Conference on Smart Manufacturing",
                "target_date": "2026-03-31",
                "status": "Planned"
            },
            {
                "title": "System Integration Framework for Digital Twins in Manufacturing",
                "type": "Journal Article",
                "journal": "IEEE Transactions on Industrial Informatics",
                "target_date": "2026-12-31",
                "status": "Planned"
            },
            {
                "title": "Patent: Digital Twin Framework for Tool Wear Prediction",
                "type": "Patent",
                "journal": "Patent Office",
                "target_date": "2027-06-30",
                "status": "Planned"
            },
            {
                "title": "Performance Evaluation of Physics-Informed Neural Networks in Digital Twins",
                "type": "Journal Article",
                "journal": "Journal of Intelligent Manufacturing",
                "target_date": "2027-12-31",
                "status": "Planned"
            },
            {
                "title": "Comprehensive Digital Twin Framework for Smart Manufacturing",
                "type": "Journal Article",
                "journal": "CIRP Annals - Manufacturing Technology",
                "target_date": "2028-06-30",
                "status": "Planned"
            },
            {
                "title": "PhD Dissertation: Digital Twin Technology for Smart Manufacturing",
                "type": "Dissertation",
                "journal": "University Repository",
                "target_date": "2029-03-31",
                "status": "Planned"
            }
        ],
        "resources": [
            {"category": "Research", "allocation": 40},
            {"category": "Development", "allocation": 30},
            {"category": "Testing", "allocation": 15},
            {"category": "Publications", "allocation": 10},
            {"category": "Stakeholder Engagement", "allocation": 5}
        ]
    }
    return roadmap_data

# New function to connect to Redis for news feed
def connect_to_redis():
    """Attempt to connect to Redis server"""
    if redis is None:
        return None
    
    try:
        # Try to connect to a local Redis instance (in production this would be configured differently)
        r = redis.Redis(host='localhost', port=6379, db=0)
        # Test the connection
        r.ping()
        return r
    except (redis.ConnectionError, redis.ResponseError):
        return None

# New function to fetch research news feeds
def fetch_research_news():
    """Fetch news from research papers and digital twin manufacturers"""
    # In a real application, this would fetch from APIs or Redis
    # For demonstration, we'll use mock data
    r = connect_to_redis()
    
    if r:
        # Try to get news from Redis
        news_items = r.get('research_news')
        if news_items:
            return json.loads(news_items)
    
    # Mock data as fallback
    return [
        {
            "title": "Advanced Digital Twin Framework for Machining Processes",
            "source": "Journal of Manufacturing Science and Engineering",
            "date": "2025-02-10",
            "url": "https://example.com/dt-machining",
            "summary": "New framework enables real-time optimization of machining parameters through digital twin integration.",
            "category": "Research"
        },
        {
            "title": "Siemens Launches Enhanced Digital Twin Platform for Manufacturing",
            "source": "Siemens Press Release",
            "date": "2025-01-15",
            "url": "https://siemens.com/press/dt-platform",
            "summary": "New platform integrates AI capabilities for predictive maintenance and process optimization.",
            "category": "Industry"
        },
        {
            "title": "Machine Learning Approaches for Digital Twin Synchronization",
            "source": "IEEE Transactions on Industrial Informatics",
            "date": "2025-01-28",
            "url": "https://ieee.org/transactions/dt-ml",
            "summary": "Novel ML algorithms improve real-time synchronization between physical assets and digital models.",
            "category": "Research"
        },
        {
            "title": "GE Digital Introduces New Data Acquisition System for Digital Twins",
            "source": "GE Digital",
            "date": "2025-02-05",
            "url": "https://ge.com/digital/dt-data",
            "summary": "System provides high-frequency data collection for more accurate digital representations.",
            "category": "Industry"
        },
        {
            "title": "Multi-physics Modeling for Comprehensive Digital Twin Applications",
            "source": "Journal of Systems Engineering",
            "date": "2025-02-18",
            "url": "https://example.com/multi-physics-dt",
            "summary": "Integration of multiple physical models enhances prediction accuracy in complex systems.",
            "category": "Research"
        }
    ]

# New function to fetch digital twin manufacturer data
def fetch_manufacturer_data():
    """Fetch data about available digital twin systems from manufacturers"""
    return [
        {
            "name": "Siemens Mindsphere",
            "category": "Full Platform",
            "capabilities": ["Asset Management", "Predictive Maintenance", "Process Optimization"],
            "integration": "High",
            "contact": "siemens.com/mindsphere"
        },
        {
            "name": "GE Digital Twin",
            "category": "Industrial IoT",
            "capabilities": ["Performance Monitoring", "Failure Prediction", "Optimization"],
            "integration": "Medium",
            "contact": "ge.com/digital/digital-twin"
        },
        {
            "name": "ANSYS Twin Builder",
            "category": "Simulation",
            "capabilities": ["Multi-physics Simulation", "System Modeling", "Predictive Maintenance"],
            "integration": "Medium",
            "contact": "ansys.com/twin-builder"
        },
        {
            "name": "PTC ThingWorx",
            "category": "IoT Platform",
            "capabilities": ["Connected Product Management", "AR Integration", "Analytics"],
            "integration": "High",
            "contact": "ptc.com/thingworx"
        },
        {
            "name": "Dassault Systèmes 3DEXPERIENCE",
            "category": "3D Platform",
            "capabilities": ["3D Visualization", "Collaborative Design", "Simulation"],
            "integration": "High",
            "contact": "3ds.com/3dexperience"
        }
    ]

# New function to create the 3D animated timeline
def create_3d_timeline(data):
    """Create an interactive 3D timeline visualization for the research roadmap"""
    df = pd.DataFrame(data["timeline"])
    
    # Convert dates to datetime objects
    df['Start'] = pd.to_datetime(df['Start'])
    df['Finish'] = pd.to_datetime(df['Finish'])
    
    # Calculate duration and progress for animation effects
    df['Duration'] = (df['Finish'] - df['Start']).dt.days
    
    # Create a 3D Timeline with Plotly
    fig = go.Figure()
    
    # Define colors for different categories
    colors = data["colors"]
    
    # Helper function to extract RGB values from color string
    def extract_rgb(color_str):
        # Use regular expression to find all numbers in the string
        rgb_values = re.findall(r'\d+', color_str)
        if len(rgb_values) >= 3:
            return int(rgb_values[0]), int(rgb_values[1]), int(rgb_values[2])
        # Fallback to a default color if parsing fails
        return 100, 100, 100
    
    # Calculate positions for 3D view (create stair-step effect)
    z_positions = {}
    z_counter = 0
    for resource in df['Resource'].unique():
        z_positions[resource] = z_counter
        z_counter += 2
    
    # Add traces for each task - these create the initial static view
    for i, row in df.iterrows():
        # Determine color based on resource
        color = colors[row['Resource']]
        
        # Determine z-position based on resource category (creates the stair-step effect)
        z_pos = z_positions[row['Resource']]
        
        # Determine height based on task importance
        height = 0.8
        
        # Create 3D bar for task
        fig.add_trace(go.Mesh3d(
            x=[row['Start'], row['Start'], row['Finish'], row['Finish'], 
               row['Start'], row['Start'], row['Finish'], row['Finish']],
            y=[0, 1, 1, 0, 0, 1, 1, 0],
            z=[z_pos, z_pos, z_pos, z_pos, z_pos+height, z_pos+height, z_pos+height, z_pos+height],
            color=color,
            opacity=0.8,
            flatshading=True,
            name=row['Task'],
            showlegend=False,
            hoverinfo="text",
            hovertext=f"<b>{row['Task']}</b><br>" + 
                      f"Start: {row['Start'].strftime('%Y-%m-%d')}<br>" +
                      f"End: {row['Finish'].strftime('%Y-%m-%d')}<br>" +
                      f"Category: {row['Resource']}<br>" +
                      f"Description: {row['Description']}"
        ))
        
        # Add text annotation for task name (positioned in 3D space)
        fig.add_trace(go.Scatter3d(
            x=[(row['Start'] + (row['Finish'] - row['Start'])/2)],
            y=[0.5],
            z=[z_pos + height + 0.1],
            mode='text',
            text=[row['Task']],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            showlegend=False
        ))
    
    # Set the layout with improved camera settings
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Timeline",
                showgrid=True,
                gridcolor='rgba(150, 150, 150, 0.2)',
                showticklabels=False,
                showspikes=False
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                showticklabels=False,
                showspikes=False
            ),
            zaxis=dict(
                title="Categories",
                showgrid=True,
                gridcolor='rgba(150, 150, 150, 0.2)',
                showticklabels=False,
                showspikes=False
            ),
            aspectmode='manual',
            aspectratio=dict(x=3, y=0.5, z=1),
            camera=dict(
                eye=dict(x=1.5, y=-2, z=1),
                up=dict(x=0, y=0, z=1)
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="3D Research Timeline (Interactive & Animated)",
        height=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Add legend for different categories
    for resource, color in colors.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=resource
        ))
    
    # Generate unique float patterns for each task
    # We'll create unique frequency and amplitude values for each task
    task_float_params = {}
    for i, row in df.iterrows():
        # Create unique floating parameters for each task
        task_float_params[i] = {
            # Vertical floating (z-axis)
            'z_freq': 0.5 + 0.5 * np.random.random(),  # Frequency between 0.5-1.0
            'z_amp': 0.2 + 0.2 * np.random.random(),   # Amplitude between 0.2-0.4
            'z_phase': 2 * np.pi * np.random.random(),  # Random phase offset
            
            # Slight horizontal floating (x-axis)
            'x_freq': 0.3 + 0.4 * np.random.random(),  # Slower frequency
            'x_amp': 0.02 + 0.02 * np.random.random(),  # Smaller amplitude
            'x_phase': 2 * np.pi * np.random.random(),  # Random phase offset
            
            # Y-axis subtle wobble
            'y_freq': 0.7 + 0.3 * np.random.random(),
            'y_amp': 0.01 + 0.01 * np.random.random(),
            'y_phase': 2 * np.pi * np.random.random(),
        }
    
    # Enhanced animation frames with more dynamic effects
    frames = []
    num_steps = 120  # More steps for smoother animation
    
    # Calculate camera positions for rotation effect
    camera_x_positions = [1.5 + 0.5 * np.sin(2 * np.pi * i / num_steps) for i in range(num_steps)]
    camera_y_positions = [-2 + 0.3 * np.cos(2 * np.pi * i / num_steps) for i in range(num_steps)]
    camera_z_positions = [1 + 0.2 * np.sin(np.pi * i / num_steps) for i in range(num_steps)]
    
    for step in range(num_steps):
        frame_data = []
        
        # Calculate animation progress (0 to 1)
        progress = step / num_steps
        
        for i, row in df.iterrows():
            z_pos = z_positions[row['Resource']]
            height = 0.8
            
            # Get floating parameters for this task
            float_params = task_float_params[i]
            
            # Calculate floating offsets
            # Z-axis floating (vertical)
            z_offset = float_params['z_amp'] * np.sin(float_params['z_freq'] * 2 * np.pi * step / num_steps + float_params['z_phase'])
            
            # X-axis subtle drift (horizontal)
            x_offset = float_params['x_amp'] * np.sin(float_params['x_freq'] * 2 * np.pi * step / num_steps + float_params['x_phase'])
            
            # Y-axis subtle wobble
            y_offset = float_params['y_amp'] * np.sin(float_params['y_freq'] * 2 * np.pi * step / num_steps + float_params['y_phase'])
            
            # Enhanced bouncing effect with sine wave
            bounce_factor = 0
            pulse_factor = 0
            
            # Apply floating effect based on completion status
            if row.get('Complete', 0) >= 100:
                # More energetic movement for completed items
                pulse_factor = 0.25 * (np.sin(4 * np.pi * step / num_steps) + 1) / 2
                height_mod = height + z_offset + pulse_factor
                
                # Dynamic color for completed items - pulse between green and bright cyan
                pulse_color = f'rgba({100 + 100 * pulse_factor}, {220 + 35 * pulse_factor}, {150 + 100 * pulse_factor}, 0.9)'
                color = pulse_color
            else:
                # Regular floating for incomplete items
                height_mod = height + z_offset
                
                # Base color with slight variation for visual interest
                base_color = colors[row['Resource']]
                # Extract RGB values properly using regex
                r, g, b = extract_rgb(base_color)
                color_variation = 15 * np.sin(2 * np.pi * (step + i * 5) / num_steps)
                color = f'rgba({min(255, max(0, r + color_variation))}, {min(255, max(0, g + color_variation))}, {min(255, max(0, b + color_variation))}, 0.8)'
            
            # Calculate positions with offsets
            # We can't directly add float offsets to Timestamp objects, so we use the original timestamps
            # and apply the offset in the mesh coordinates
            x_start = row['Start']
            x_finish = row['Finish']
            y_base = 0 + y_offset
            y_top = 1 + y_offset
            
            # Create 3D bar with enhanced floating effects
            frame_data.append(go.Mesh3d(
                # Apply the x_offset when creating the mesh coordinates
                x=[x_start + pd.Timedelta(days=x_offset*30), 
                   x_start + pd.Timedelta(days=x_offset*30), 
                   x_finish + pd.Timedelta(days=x_offset*30), 
                   x_finish + pd.Timedelta(days=x_offset*30), 
                   x_start + pd.Timedelta(days=x_offset*30), 
                   x_start + pd.Timedelta(days=x_offset*30), 
                   x_finish + pd.Timedelta(days=x_offset*30), 
                   x_finish + pd.Timedelta(days=x_offset*30)],
                y=[y_base, y_top, y_top, y_base,
                   y_base, y_top, y_top, y_base],
                z=[z_pos + z_offset, z_pos + z_offset, z_pos + z_offset, z_pos + z_offset, 
                   z_pos + height_mod, z_pos + height_mod, z_pos + height_mod, z_pos + height_mod],
                color=color,
                opacity=0.8 + 0.1 * np.sin(np.pi * step / (num_steps/4)),  # Pulsing opacity
                flatshading=True,
                name=row['Task'],
                showlegend=False,
                hoverinfo="text",
                hovertext=f"<b>{row['Task']}</b><br>" + 
                          f"Start: {row['Start'].strftime('%Y-%m-%d')}<br>" +
                          f"End: {row['Finish'].strftime('%Y-%m-%d')}<br>" +
                          f"Category: {row['Resource']}<br>" +
                          f"Description: {row['Description']}"
            ))
            
            # Update text label position to follow the floating block
            # Calculate the midpoint between start and finish dates and apply the offset
            midpoint = x_start + (x_finish - x_start)/2
            text_x = midpoint + pd.Timedelta(days=x_offset*30)
            text_y = y_top/2
            text_z = z_pos + height_mod + 0.1
            
            # Add floating text label
            frame_data.append(go.Scatter3d(
                x=[text_x],
                y=[text_y],
                z=[text_z],
                mode='text',
                text=[row['Task']],
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                showlegend=False
            ))
        
        # Update camera position for dynamic rotation effect
        camera_update = dict(
            scene=dict(
                camera=dict(
                    eye=dict(
                        x=camera_x_positions[step],
                        y=camera_y_positions[step],
                        z=camera_z_positions[step]
                    )
                )
            )
        )
        
        frames.append(go.Frame(data=frame_data, layout=camera_update, name=f"step{step}"))
    
    fig.frames = frames
    
    # Enhanced animation controls
    updatemenus = [dict(
        type="buttons",
        buttons=[dict(label="▶ Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 50, "redraw": True},
                                   "fromcurrent": True,
                                   "mode": "immediate",
                                   "transition": {"duration": 50}}]),
                 dict(label="❚❚ Pause",
                      method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}]),
                 dict(label="↻ Reset View",
                      method="relayout",
                      args=[{"scene.camera.eye": dict(x=1.5, y=-2, z=1)}])],
        direction="right",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.1,
        xanchor="right",
        y=0,
        yanchor="top",
        bgcolor="rgba(50, 50, 50, 0.7)",
        font=dict(color="white")
    )]
    
    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 12},
            "prefix": "Animation Frame: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 50, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [
                    [f"step{k}"],
                    {"frame": {"duration": 50, "redraw": True},
                     "mode": "immediate",
                     "transition": {"duration": 50}}
                ],
                "label": str(k),
                "method": "animate"
            }
            for k in range(0, num_steps, 10)  # Show fewer steps in slider for clarity
        ]
    }]
    
    fig.update_layout(updatemenus=updatemenus, sliders=sliders)
    
    return fig

# Create news feed visualization
def render_news_feed():
    """Render the news feed section for research papers and manufacturer updates"""
    st.subheader("Research & Industry News Feed")
    
    # Tabs for different news categories
    news_tab1, news_tab2 = st.tabs(["Research Papers", "Manufacturer Updates"])
    
    with news_tab1:
        st.markdown("### Latest Research in Digital Twin Technology")
        
        # Fetch and display news items
        news_items = fetch_research_news()
        research_news = [item for item in news_items if item["category"] == "Research"]
        
        # Display news feed with custom styling
        for i, news in enumerate(research_news):
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(70, 150, 180, 0.1); border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid rgba(100, 200, 250, 0.6);">
                    <h4 style="color: rgba(100, 220, 250, 0.9); margin-top: 0;">{news['title']}</h4>
                    <p style="font-size: 0.8em; color: rgba(200, 200, 200, 0.8);">Source: {news['source']} | Published: {news['date']}</p>
                    <p>{news['summary']}</p>
                    <a href="{news['url']}" target="_blank" style="color: rgba(100, 200, 250, 0.8);">Read more</a>
                        </div>
                """, unsafe_allow_html=True)
    
    with news_tab2:
        st.markdown("### Digital Twin Manufacturer Updates")
        
        # Fetch and display manufacturer updates
        news_items = fetch_research_news()
        industry_news = [item for item in news_items if item["category"] == "Industry"]
        
        # Display industry news
        for i, news in enumerate(industry_news):
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(80, 120, 160, 0.1); border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid rgba(120, 180, 220, 0.6);">
                    <h4 style="color: rgba(120, 200, 220, 0.9); margin-top: 0;">{news['title']}</h4>
                    <p style="font-size: 0.8em; color: rgba(200, 200, 200, 0.8);">Source: {news['source']} | Published: {news['date']}</p>
                    <p>{news['summary']}</p>
                    <a href="{news['url']}" target="_blank" style="color: rgba(120, 180, 220, 0.8);">Read more</a>
                    </div>
                """, unsafe_allow_html=True)
        
        # Display manufacturer information
        st.markdown("### Available Digital Twin Systems")
        mfg_data = fetch_manufacturer_data()
        
        for i, mfg in enumerate(mfg_data):
            with st.expander(f"{mfg['name']} - {mfg['category']}"):
                st.markdown(f"""
                **Capabilities:**
                """)
                for cap in mfg['capabilities']:
                    st.markdown(f"- {cap}")
                
                st.markdown(f"""
                **Integration Level:** {mfg['integration']}
                
                **Contact:** [{mfg['contact']}](https://{mfg['contact']})
                """)

# Create a section for data acquisition
def render_data_acquisition():
    """Render the section for data acquisition from manufacturers and digital twins"""
    st.subheader("Data Acquisition & Integration")
    
    # Tabs for different data acquisition methods
    acq_tab1, acq_tab2, acq_tab3 = st.tabs(["Manufacturer APIs", "Digital Twin Integration", "Custom Sensors"])
    
    with acq_tab1:
        st.markdown("### Connect to Manufacturer APIs")
        
        # List available manufacturer APIs
        st.markdown("""
        <div style="background: rgba(70, 70, 70, 0.1); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
            <h4 style="margin-top: 0;">Available Manufacturer APIs</h4>
            <p>These APIs allow direct access to manufacturer data and systems.</p>
                </div>
                """, unsafe_allow_html=True)
    
        # Create mock API connection interface
        api_options = [
            "Siemens Mindsphere API", 
            "GE Digital Twin API", 
            "ANSYS Twin Builder API",
            "PTC ThingWorx API",
            "Dassault Systèmes 3DEXPERIENCE API"
        ]
        
        selected_api = st.selectbox("Select Manufacturer API", api_options)
        api_key = st.text_input("API Key", type="password")
        endpoint = st.text_input("API Endpoint", value=f"https://api.{selected_api.lower().replace(' ', '')}.com/v1")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Categories**")
            data_categories = ["Asset Information", "Telemetry Data", "Historical Performance", "Maintenance Records"]
            selected_categories = []
            for category in data_categories:
                if st.checkbox(category):
                    selected_categories.append(category)
        
        with col2:
            st.markdown("**Update Frequency**")
            update_freq = st.radio("", ["Real-time", "Hourly", "Daily", "Weekly"])
        
        if st.button("Connect to API"):
            if api_key:
                st.success(f"Successfully connected to {selected_api} with access to {', '.join(selected_categories)}!")
                
                # Display mock data stream
                st.markdown("### Live Data Stream")
                placeholder = st.empty()
                
                for i in range(5):
                    data_json = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "device_id": f"DT-{random.randint(1000, 9999)}",
                        "temperature": round(random.uniform(60, 85), 1),
                        "pressure": round(random.uniform(95, 105), 2),
                        "vibration": round(random.uniform(0.1, 0.5), 3),
                        "status": random.choice(["Normal", "Warning", "Normal", "Normal"])
                    }
                    
                    placeholder.json(data_json)
                    time.sleep(0.5)
            else:
                st.error("API Key is required to connect!")
    
    with acq_tab2:
        st.markdown("### Digital Twin Integration")
        
        # Explain digital twin integration
        st.markdown("""
        <div style="background: rgba(70, 70, 70, 0.1); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
            <h4 style="margin-top: 0;">Digital Twin Integration Options</h4>
            <p>Connect your research to existing digital twin systems.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show integration options
        integration_options = [
            "Direct API Integration",
            "Data Lake Connection",
            "Message Queue System",
            "Real-time Streaming",
            "File-based Exchange"
        ]
        
        selected_integration = st.radio("Select Integration Method", integration_options)
        
        if selected_integration == "Direct API Integration":
            st.markdown("""
            **Configuration Requirements:**
            - API Endpoints
            - Authentication Details
            - Rate Limits
            - Data Format Specifications
            """)
        elif selected_integration == "Data Lake Connection":
            st.markdown("""
            **Configuration Requirements:**
            - Storage Connection String
            - Data Structure Schema
            - Access Permissions
            - ETL Process Definition
            """)
        elif selected_integration == "Message Queue System":
            st.markdown("""
            **Configuration Requirements:**
            - Queue Server Details
            - Topic/Queue Names
            - Message Format
            - Subscription Details
            """)
        elif selected_integration == "Real-time Streaming":
            st.markdown("""
            **Configuration Requirements:**
            - Streaming Service Endpoints
            - Stream Keys
            - Data Schemas
            - Processing Rules
            """)
        elif selected_integration == "File-based Exchange":
            st.markdown("""
            **Configuration Requirements:**
            - File Location/FTP Details
            - File Naming Convention
            - File Format Specifications
            - Exchange Frequency
            """)
        
        # Demo integration
        connection_string = st.text_area("Connection Configuration", 
                                         value='{"endpoint": "https://dt-api.example.com", "auth": "****", "format": "JSON"}')
        
        if st.button("Test Integration"):
            st.success("Integration test successful! Connected to digital twin system.")
            
            # Show sample data flow diagram
            st.markdown("### Data Flow Visualization")
            
            # Create a simple Sankey diagram to show data flow
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = "rgba(50, 50, 50, 0.5)", width = 0.5),
                    label = ["Physical Asset", "Sensors", "Edge Device", "Cloud Platform", 
                             "Digital Twin", "Analytics Engine", "Research Application"],
                    color = "rgba(100, 200, 250, 0.8)"
                ),
                link = dict(
                    source = [0, 0, 1, 2, 3, 3, 4, 5],
                    target = [1, 2, 2, 3, 4, 5, 5, 6],
                    value = [10, 5, 15, 20, 15, 5, 10, 15],
                    color = "rgba(100, 200, 250, 0.3)"
                )
            )])
            
            fig.update_layout(
                title_text="Digital Twin Data Flow",
                font=dict(size=10, color="white"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with acq_tab3:
        st.markdown("### Custom Sensor Network")
        
        # Explain custom sensor integration
        st.markdown("""
        <div style="background: rgba(70, 70, 70, 0.1); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
            <h4 style="margin-top: 0;">Custom Sensor Configuration</h4>
            <p>Setup and configure custom sensors to feed into your research digital twin.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create sensor configuration interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sensor Types**")
            sensor_types = {
                "Temperature": st.checkbox("Temperature Sensors", value=True),
                "Pressure": st.checkbox("Pressure Sensors", value=True),
                "Vibration": st.checkbox("Vibration Sensors"),
                "Position": st.checkbox("Position Sensors"),
                "Acoustic": st.checkbox("Acoustic Sensors"),
                "Current": st.checkbox("Current Sensors"),
                "Voltage": st.checkbox("Voltage Sensors"),
                "Flow": st.checkbox("Flow Sensors")
            }
            
            # Count selected sensors
            selected_sensors = sum(1 for value in sensor_types.values() if value)
        
        with col2:
            st.markdown("**Network Configuration**")
            network_type = st.radio("Network Type", ["Wired", "Wireless", "Hybrid"])
            protocol = st.selectbox("Communication Protocol", ["MQTT", "OPC UA", "Modbus", "BACnet", "Custom"])
            frequency = st.slider("Sampling Frequency (Hz)", 1, 100, 10)
        
        # Show estimated setup
        st.markdown(f"""
        ### Configuration Summary
        
        - **Sensors Selected:** {selected_sensors}
        - **Network Type:** {network_type}
        - **Protocol:** {protocol}
        - **Sampling Frequency:** {frequency} Hz
        - **Estimated Data Volume:** {selected_sensors * frequency * 60 * 60 / 1000:.2f} MB/hour
        """)
        
        if st.button("Deploy Sensor Configuration"):
            st.success(f"Sensor configuration deployed! {selected_sensors} sensors have been configured.")

def render_phd_roadmap():
    """Render the PhD Research Roadmap tab with timeline and milestones"""
    # Load the roadmap data
    data = load_phd_roadmap_data()
    
    # Create tabs for different views
    phd_tab1, phd_tab2, phd_tab3, phd_tab4 = st.tabs(["Research Timeline", "Milestones", "Publications", "Resources"])
    
    with phd_tab1:
        st.subheader("PhD Research Timeline")
        
        # Add filtering options
        categories = list(data["colors"].keys())
        selected_categories = st.multiselect(
            "Filter by category",
            categories,
            default=categories,
            key="timeline_filter"
        )
        
        # Filter the timeline data
        if selected_categories:
            filtered_timeline = [item for item in data["timeline"] if item["Resource"] in selected_categories]
            filtered_data = data.copy()
            filtered_data["timeline"] = filtered_timeline
            gantt_chart = create_gantt_chart(filtered_data)
        else:
            gantt_chart = create_gantt_chart(data)
        
        # Display the Gantt chart
        st.plotly_chart(gantt_chart, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; margin-top: 20px;">
            <h4 style="color: rgba(100, 255, 200, 0.9);">About the Timeline</h4>
            <p>This Gantt chart shows the planned research activities from 2025 to 2029, spanning literature review, 
            methodology development, implementation, testing, and publication activities. The timeline highlights 
            key dependencies and parallel activities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with phd_tab2:
        st.subheader("Key Milestones")
        
        # Display milestone cards
        create_milestone_cards(data)
        
        # Add a timeline visualization of milestones
        milestone_df = pd.DataFrame(data["milestones"])
        
        # Add quarter dates for visualization
        quarter_map = {
            "Q1 2025": "2025-01-01", "Q2 2025": "2025-04-01", "Q3 2025": "2025-07-01", "Q4 2025": "2025-10-01",
            "Q1 2026": "2026-01-01", "Q2 2026": "2026-04-01", "Q3 2026": "2026-07-01", "Q4 2026": "2026-10-01",
            "Q1 2027": "2027-01-01", "Q2 2027": "2027-04-01", "Q3 2027": "2027-07-01", "Q4 2027": "2027-10-01",
            "Q1 2028": "2028-01-01", "Q2 2028": "2028-04-01", "Q3 2028": "2028-07-01", "Q4 2028": "2028-10-01",
            "Q1 2029": "2029-01-01", "Q2 2029": "2029-04-01", "Q3 2029": "2029-07-01", "Q4 2029": "2029-10-01"
        }
        
        milestone_df["date_value"] = milestone_df["date"].map(quarter_map)
        milestone_df["date_value"] = pd.to_datetime(milestone_df["date_value"])
        
        # Create an end date 3 months after the start date for better visualization
        milestone_df["end_date"] = milestone_df["date_value"] + pd.DateOffset(months=3)
        
        # Add jitter to y-positions for better spacing when multiple milestones are in the same quarter
        milestone_df["y_position"] = range(len(milestone_df))
        
        # Create custom hover text
        milestone_df["hover_text"] = milestone_df.apply(
            lambda row: f"<b>{row['title']}</b><br>" +
                       f"Date: {row['date']}<br>" +
                       f"Status: {row['status']}<br>" + 
                       f"Completion: {row['completion_percentage']}%<br>" +
                       f"Description: {row['description']}", 
            axis=1
        )
        
        # Create the timeline visualization with enhanced design
        fig = go.Figure()
        
        # Define status colors with higher contrast and better visibility
        status_colors = {
            "Planned": "rgba(70, 180, 255, 0.9)",        # Bright blue
            "In Progress": "rgba(255, 170, 50, 0.9)",    # Orange
            "Completed": "rgba(50, 200, 100, 0.9)",      # Green
            "Delayed": "rgba(255, 100, 100, 0.9)",       # Red
            "At Risk": "rgba(255, 50, 50, 0.9)"          # Bright red
        }
        
        # Add milestone markers with improved styling
        for i, row in milestone_df.iterrows():
            marker_color = status_colors.get(row["status"], "rgba(150, 150, 150, 0.9)")
            
            # Add milestone marker
            fig.add_trace(go.Scatter(
                x=[row["date_value"]],
                y=[row["title"]],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=16,
                    color=marker_color,
                    line=dict(width=2, color="white")
                ),
                name=row["status"],
                text=row["hover_text"],
                hoverinfo="text",
                showlegend=False
            ))
        
        # Create milestone labels with better positioning
        fig.add_trace(go.Scatter(
            x=milestone_df["date_value"],
            y=milestone_df["title"],
            mode="text",
            text=milestone_df["title"],
            textposition="middle left",
            textfont=dict(size=12, color="white"),
            showlegend=False
        ))
        
        # Add a legend for status colors
        for status, color in status_colors.items():
            if status in milestone_df["status"].values:
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=status
                ))
        
        # Customize the timeline layout
        fig.update_layout(
            title="Milestone Timeline",
            xaxis=dict(
                title="Timeline",
                type="date",
                tickformat="%b %Y",  # Format as "Jan 2025"
                tickfont=dict(size=12),
                gridcolor="rgba(255, 255, 255, 0.1)",
                showgrid=True
            ),
            yaxis=dict(
                title="",
                autorange="reversed"  # Reverse the y-axis so first milestone is at the top
            ),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(color="white"),
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.1)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with phd_tab3:
        st.subheader("Publications & Patents")
        
        # Create publication timeline
        pub_chart = create_publications_timeline(data)
        st.plotly_chart(pub_chart, use_container_width=True)
        
        # Add publication details
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; margin-top: 20px;">
            <h4 style="color: rgba(100, 255, 200, 0.9);">Publication Strategy</h4>
            <p>The publication strategy includes journal articles, conference papers, and a patent application.
            The research is structured to produce high-impact publications in leading journals in the field of
            digital twin technology and smart manufacturing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display publication list
        pub_df = pd.DataFrame(data["publications"])
        
        # Add a table with more details
        st.markdown("### Publication Plan")
        st.table(pub_df[["title", "type", "journal", "target_date", "status"]])
    
    with phd_tab4:
        st.subheader("Resource Allocation")
        
        # Create resource allocation chart
        resource_chart = create_resource_allocation(data)
        st.plotly_chart(resource_chart, use_container_width=True)
        
        # Add resource allocation details
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; margin-top: 20px;">
            <h4 style="color: rgba(100, 255, 200, 0.9);">Resource Distribution</h4>
            <p>This visualization shows the planned distribution of resources across different categories of the PhD research.
            The majority of resources are allocated to research and development activities, with appropriate allocations
            for testing, publications, and stakeholder engagement.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add quarterly resource requirement details
        st.markdown("### Key Resource Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: rgba(100, 255, 200, 0.9);">Computing Resources</h4>
                <ul>
                    <li>High-performance computing cluster</li>
                    <li>GPU resources for deep learning</li>
                    <li>Cloud storage for dataset management</li>
                    <li>Real-time simulation servers</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: rgba(100, 255, 200, 0.9);">Human Resources</h4>
                <ul>
                    <li>Faculty advisor time</li>
                    <li>Research collaborator engagement</li>
                    <li>Industry partner participation</li>
                    <li>Technical support staff</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def render_research_roadmap():
    """Render the Research & Development Roadmap page"""
    # Set page title
    st.title("Research & Development Roadmap")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Current Projects", 
        "Future Initiatives", 
        "Research Impact", 
        "PhD Research Timeline",
        "Data Acquisition",
        "Research News Feed"
    ])
    
    with tab1:
        st.header("Current Projects")
        
        # Display active research projects
        active_projects = [
            {
                "title": "Digital Twin for Machine Tool Dynamics",
                "progress": 75,
                "description": "Development of a digital twin system for machine tool dynamics with predictive capabilities.",
                "deadline": "2025-12-31",
                "status": "On Track"
            },
            {
                "title": "Real-time Process Monitoring System",
                "progress": 60,
                "description": "Implementation of a real-time monitoring system for manufacturing processes using sensor networks.",
                "deadline": "2025-10-15",
                "status": "On Track"
            },
            {
                "title": "Edge Computing Framework for Digital Twins",
                "progress": 40,
                "description": "Development of an edge computing framework to support digital twin applications in manufacturing.",
                "deadline": "2026-03-20",
                "status": "Needs Attention"
            }
        ]
        
        for project in active_projects:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(project["title"])
                    st.write(project["description"])
                    st.progress(project["progress"] / 100)
                    st.write(f"**Deadline:** {project['deadline']} | **Status:** {project['status']}")
                
                with col2:
                    if project["status"] == "On Track":
                        st.markdown("🟢")
                    elif project["status"] == "Needs Attention":
                        st.markdown("🟠")
                    else:
                        st.markdown("🔴")
            
            st.markdown("---")
    
    with tab2:
        st.header("Future Initiatives")
        
        # Future research initiatives
        future_initiatives = [
            {
                "title": "AI-Enhanced Digital Twin",
                "start_date": "2026-01-01",
                "end_date": "2027-12-31",
                "description": "Integration of advanced AI capabilities into digital twin framework for enhanced prediction and optimization."
            },
            {
                "title": "Cross-Domain Digital Twin Integration",
                "start_date": "2026-06-01",
                "end_date": "2028-06-30",
                "description": "Development of methods for integrating digital twins across multiple domains and systems."
            },
            {
                "title": "Quantum Computing for Digital Twins",
                "start_date": "2027-01-01",
                "end_date": "2029-12-31",
                "description": "Exploration of quantum computing applications for complex digital twin simulations."
            }
        ]
        
        for initiative in future_initiatives:
            with st.expander(initiative["title"]):
                st.write(f"**Start Date:** {initiative['start_date']}")
                st.write(f"**End Date:** {initiative['end_date']}")
                st.write(f"**Description:** {initiative['description']}")
    
    with tab3:
        st.header("Research Impact Analysis")
        
        # Impact metrics
        metrics = {
            "Publications": {"current": 15, "target": 25},
            "Patents Filed": {"current": 8, "target": 12},
            "Industry Collaborations": {"current": 6, "target": 10},
            "Research Grants (M€)": {"current": 2.5, "target": 5.0}
        }
        
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        
        for (metric, values), col in zip(metrics.items(), cols):
            with col:
                progress = (values["current"] / values["target"]) * 100
                st.metric(
                    metric,
                    f"{values['current']} / {values['target']}",
                    f"{progress:.1f}% of target"
                )
        
        # Impact visualization
        impact_data = {
            "Categories": ["Technical", "Economic", "Social", "Environmental"],
            "Current Impact": [85, 70, 60, 75],
            "Projected Impact": [95, 85, 80, 90]
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=impact_data["Current Impact"],
            theta=impact_data["Categories"],
            fill='toself',
            name='Current Impact',
            line_color='rgba(100, 255, 200, 0.8)',
            fillcolor='rgba(100, 255, 200, 0.2)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=impact_data["Projected Impact"],
            theta=impact_data["Categories"],
            fill='toself',
            name='Projected Impact',
            line_color='rgba(100, 200, 255, 0.8)',
            fillcolor='rgba(100, 200, 255, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Research Impact Assessment",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Render the PhD Research Timeline
        render_phd_roadmap()
        
        # Add the new 3D animated timeline
        st.subheader("3D Interactive Timeline Visualization")
        data = load_phd_roadmap_data()
        fig_3d = create_3d_timeline(data)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; margin-top: 20px;">
            <h4 style="color: rgba(100, 255, 200, 0.9);">About the 3D Timeline</h4>
            <p>This interactive 3D visualization allows you to explore the PhD research milestones in an immersive way.
            Completed milestones will bounce and change color when hovered over. The timeline is structured as a floating
            staircase where each step represents a different research category.</p>
            <p>Use your mouse to rotate, pan, and zoom the visualization. Click and drag to rotate, scroll to zoom,
            and right-click and drag to pan.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        # Render the data acquisition section
        render_data_acquisition()
    
    with tab6:
        # Render the news feed section
        render_news_feed()

    # Add interactive elements for adding new milestones, notes, and collaborations
    new_milestone_date = st.date_input("New Milestone Date")
    new_milestone_description = st.text_input("New Milestone Description")
    if st.button("Add Milestone"):
        roadmap.add_milestone(new_milestone_date, new_milestone_description)
        st.success("Milestone added!")

    new_note = st.text_area("New Note")
    if st.button("Add Note"):
        roadmap.add_note(new_note)
        st.success("Note added!")

    new_collaboration_team = st.text_input("New Collaboration Team")
    new_collaboration_description = st.text_input("New Collaboration Description")
    if st.button("Add Collaboration"):
        roadmap.add_collaboration(new_collaboration_team, new_collaboration_description)
        st.success("Collaboration added!")

# New function to create Gantt charts for the PhD research timeline
def create_gantt_chart(data):
    """Create a Gantt chart visualization using Plotly"""
    if isinstance(data, dict) and "timeline" in data:
        df = pd.DataFrame(data['timeline'])
    else:
        # Handle case where a direct timeline list is passed
        df = pd.DataFrame(data)
    
    # Ensure required columns are present
    required_cols = ['Task', 'Start', 'Finish', 'Resource']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column '{col}' for Gantt chart")
            return None
    
    # Create the Gantt chart
    colors = data.get('colors', {}) if isinstance(data, dict) else {}
    fig = ff.create_gantt(df, colors=colors, index_col='Resource', 
                         show_colorbar=True, group_tasks=True)
    
    # Update layout
    fig.update_layout(
        title="PhD Research Roadmap (2025-2029)",
        height=600,
        xaxis_title="Timeline",
        legend_title="Categories",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    # Add hover information
    fig.update_traces(hoverinfo="text", hovertext=df['Description'] if 'Description' in df.columns else None)
    
    return fig

# Function to create resource allocation visualization
def create_resource_allocation(data):
    """Create a visualization for resource allocation across different categories"""
    if not isinstance(data, dict) or "resources" not in data:
        st.error("Invalid data format for resource allocation")
        return None
    
    # Extract resource data
    resources = data['resources']
    resource_df = pd.DataFrame(resources)
    
    # Sort by allocation (descending)
    resource_df = resource_df.sort_values('allocation', ascending=False)
    
    # Create pie chart for resource allocation
    pie_fig = go.Figure(data=[go.Pie(
        labels=resource_df['category'],
        values=resource_df['allocation'],
        hole=0.4,
        textinfo='label+percent',
        marker=dict(
            colors=[
                'rgb(46, 137, 205)',   # Research
                'rgb(114, 44, 121)',   # Development
                'rgb(198, 47, 105)',   # Testing
                'rgb(58, 149, 136)',   # Publications
                'rgb(107, 127, 135)'   # Stakeholder Engagement
            ],
            line=dict(color='rgba(0,0,0,0)', width=2)
        ),
        pull=[0.1 if cat == resource_df['category'].iloc[0] else 0 for cat in resource_df['category']],
        hovertemplate='<b>%{label}</b><br>Allocation: %{value}%<br>%{percent}'
    )])
    
    # Create a bar chart for comparison
    bar_fig = go.Figure(data=[go.Bar(
        x=resource_df['category'],
        y=resource_df['allocation'],
        marker_color=[
            'rgb(46, 137, 205)',   # Research
            'rgb(114, 44, 121)',   # Development
            'rgb(198, 47, 105)',   # Testing
            'rgb(58, 149, 136)',   # Publications
            'rgb(107, 127, 135)'   # Stakeholder Engagement
        ],
        text=resource_df['allocation'].astype(str) + '%',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Allocation: %{y}%'
    )])
    
    # Create subplots to show both visualizations
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "xy"}]],
        subplot_titles=("Allocation Percentage", "Allocation by Category")
    )
    
    # Add pie chart to the first subplot
    fig.add_trace(pie_fig.data[0], row=1, col=1)
    
    # Add bar chart to the second subplot
    fig.add_trace(bar_fig.data[0], row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title="PhD Research Resource Allocation",
        height=500,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(t=80, l=40, r=40, b=40)
    )
    
    # Update y-axis for bar chart
    fig.update_yaxes(
        title_text="Allocation (%)",
        range=[0, 100],
        gridcolor='rgba(200, 200, 200, 0.2)',
        row=1, col=2
    )
    
    # Update x-axis for bar chart
    fig.update_xaxes(
        tickangle=45,
        row=1, col=2
    )
    
    return fig

# Function to create publication timeline visualization
def create_publications_timeline(data):
    """Create a timeline visualization for publications and patents"""
    if not isinstance(data, dict) or "publications" not in data:
        st.error("Invalid data format for publications timeline")
        return None
    
    # Extract publication data
    publications = data['publications']
    pub_df = pd.DataFrame(publications)
    
    # Convert target_date to datetime for proper sorting
    pub_df['target_date'] = pd.to_datetime(pub_df['target_date'])
    
    # Sort by date
    pub_df = pub_df.sort_values('target_date')
    
    # Create color mapping for publication types
    color_map = {
        'Journal Article': 'rgb(46, 137, 205)',
        'Conference Paper': 'rgb(114, 44, 121)',
        'Patent': 'rgb(198, 47, 105)',
        'Book Chapter': 'rgb(58, 149, 136)',
        'Dissertation': 'rgb(107, 127, 135)'
    }
    
    # Add a jitter value to create vertical separation between items with same date
    pub_df['y_position'] = range(len(pub_df))
    
    # Create the figure
    fig = go.Figure()
    
    # Add publication markers
    for i, pub in pub_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[pub['target_date']],
            y=[pub['y_position']],
            mode='markers+text',
            marker=dict(
                size=20,
                color=color_map.get(pub['type'], 'gray'),
                symbol={
                    'Journal Article': 'circle',
                    'Conference Paper': 'diamond',
                    'Patent': 'star',
                    'Book Chapter': 'square',
                    'Dissertation': 'triangle-up'
                }.get(pub['type'], 'circle')
            ),
            text=pub['type'][0],  # First letter of type
            textposition="middle center",
            textfont=dict(color='white', size=10),
            name=pub['type'],
            hovertemplate=(
                f"<b>{pub['title']}</b><br>"
                f"Type: {pub['type']}<br>"
                f"Journal/Venue: {pub['journal']}<br>"
                f"Target Date: {pub['target_date'].strftime('%Y-%m-%d')}<br>"
                f"Status: {pub['status']}"
            )
        ))
    
    # Add a line connecting all publications
    fig.add_trace(go.Scatter(
        x=pub_df['target_date'],
        y=pub_df['y_position'],
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=2),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title="Publication & Patent Timeline",
        xaxis=dict(
            title="Timeline",
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            showticklabels=False,
            zeroline=False,
            showgrid=False
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        hovermode="closest"
    )
    
    # Add legend for publication types
    for pub_type, color in color_map.items():
        if pub_type in pub_df['type'].values:
            fig.add_trace(go.Scatter(
                x=[pub_df['target_date'].min()],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                    symbol={
                        'Journal Article': 'circle',
                        'Conference Paper': 'diamond',
                        'Patent': 'star',
                        'Book Chapter': 'square',
                        'Dissertation': 'triangle-up'
                    }.get(pub_type, 'circle')
                ),
                name=pub_type,
                showlegend=True
            ))
    
    return fig

# Function to create milestone cards for the PhD research timeline
def create_milestone_cards(data):
    """Create interactive milestone cards for each milestone in the timeline"""
    if not isinstance(data, dict) or "milestones" not in data:
        st.error("Invalid data format for milestone cards")
        return
    
    milestones = data['milestones']
    
    # Create a grid layout for milestone cards (3 columns)
    cols = st.columns(3)
    
    for i, milestone in enumerate(milestones):
        # Cycle through columns
        col_idx = i % 3
        
        # Create styled expander for each milestone
        with cols[col_idx]:
            # Determine status color
            status_color = {
                "Completed": "green",
                "In Progress": "blue",
                "Planned": "gray",
                "Delayed": "orange",
                "At Risk": "red"
            }.get(milestone.get("status", "Planned"), "gray")
            
            # Create an expander with emoji and formatting
            with st.expander(f"📌 {milestone.get('title', 'Milestone')} - {milestone.get('date', 'TBD')}"):
                st.markdown(f"**Description:** {milestone.get('description', 'No description available')}")
                
                # Show progress bar if completion percentage is available
                completion = milestone.get('completion_percentage', 0)
                st.progress(completion / 100)
                
                # Show status with color coding
                st.markdown(f"**Status:** <span style='color:{status_color};font-weight:bold;'>{milestone.get('status', 'Planned')}</span>", 
                            unsafe_allow_html=True)
                
                # Show deliverables if available
                deliverables = milestone.get('deliverables', [])
                if deliverables:
                    st.markdown("**Deliverables:**")
                    for d in deliverables:
                        st.markdown(f"- {d}")

if __name__ == "__main__":
    render_research_roadmap() 