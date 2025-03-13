import streamlit as st
import time
from factory_components import youtube_manager

def render_youtube_stream():
    """Render the YouTube Live Streaming interface."""
    st.title("Media Tool: Youtube Stream")
    
    # Main layout with two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # YouTube Live Stream section
        st.header("YouTube Live Stream")
        
        # Placeholder for the live stream preview
        stream_container = st.container()
        with stream_container:
            # YouTube logo as placeholder when not streaming
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; 
                height: 400px; background-color: #000; border-radius: 10px;">
                    <div style="background-color: #FF0000; border-radius: 20px; 
                    width: 80px; height: 60px; display: flex; justify-content: center; align-items: center;">
                        <div style="border-style: solid; border-width: 15px 0 15px 30px; 
                        border-color: transparent transparent transparent #FFFFFF;"></div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Create tabs for stream info, chat, and analytics
        info_tab, chat_tab, analytics_tab = st.tabs(["Stream Info", "Live Chat", "Analytics"])
        
        with info_tab:
            st.subheader("Live Viewers")
            viewers_col1, viewers_col2 = st.columns([1, 2])
            
            with viewers_col1:
                st.metric(label="", value="24", delta="+8")
            
            with viewers_col2:
                st.markdown("**Watch Time**")
                st.metric(label="", value="48 minutes", delta="+12%")
                
            st.markdown("**Engagement**")
            st.metric(label="", value="86%", delta="+5%")
        
        with chat_tab:
            st.markdown("### Live Chat")
            st.markdown("_Chat messages will appear here_")
        
        with analytics_tab:
            st.markdown("### Real-time Analytics")
            st.bar_chart({"Viewers": [5, 10, 15, 24, 22, 18, 20, 24]})
    
    with col2:
        # Stream Controls section
        st.header("Stream Controls")
        
        # Stream status dropdown
        st.markdown("**Stream Status**")
        stream_status = st.selectbox(
            label="",
            options=["Live", "Scheduled", "Offline"],
            index=0,
            key="stream_status"
        )
        
        # Stream control buttons
        if stream_status == "Live":
            if st.button("End Stream", key="end_stream", use_container_width=True):
                with st.spinner("Ending stream..."):
                    youtube_manager.end_broadcast()
                st.success("Stream ended successfully")
                st.session_state.stream_status = "Offline"
                st.rerun()
        else:
            if st.button("Start Stream", key="start_stream", use_container_width=True):
                with st.spinner("Starting stream..."):
                    # Create a new broadcast
                    youtube_manager.authenticate()
                    broadcast_id = youtube_manager.create_broadcast(
                        "HyperSyncDT Manufacturing Demo", 
                        "Live demonstration of the HyperSyncDT manufacturing system"
                    )
                    
                    # Create a new stream
                    stream_details = youtube_manager.create_stream()
                    
                    # Bind the broadcast to the stream
                    youtube_manager.bind_broadcast()
                    
                    # Start the broadcast
                    youtube_manager.start_broadcast()
                    
                st.success("Stream started successfully")
                st.session_state.stream_status = "Live"
                
                # Show stream key and URL
                with st.expander("Stream Details"):
                    st.code("rtmp://a.rtmp.youtube.com/live2", language=None)
                    st.code("simulated-stream-key-789", language=None)
        
        # Settings section
        st.header("Settings")
        
        # Privacy settings
        st.markdown("**Privacy**")
        privacy_col1, privacy_col2, privacy_col3 = st.columns(3)
        
        with privacy_col1:
            st.radio("", ["Public"], key="privacy_public", label_visibility="collapsed")
        
        with privacy_col2:
            st.radio("", ["Unlisted"], key="privacy_unlisted", label_visibility="collapsed")
        
        with privacy_col3:
            st.radio("", ["Private"], key="privacy_private", label_visibility="collapsed")
        
        # DVR option
        st.checkbox("Enable DVR", value=True)
        
        # Auto-start option
        st.checkbox("Auto-start when live")
        
        # Stream delay slider
        st.markdown("**Stream Delay**")
        st.markdown("5 sec")
        delay_value = st.slider("", min_value=0, max_value=30, value=5, label_visibility="collapsed")
        
if __name__ == "__main__":
    render_youtube_stream() 