import streamlit as st
import sys
import importlib

# Force reload modules
if 'components.interactive_header' in sys.modules:
    importlib.reload(sys.modules['components.interactive_header'])

# Import the header class
from components.interactive_header import AdvancedInteractiveHeader

# Configure the page
st.set_page_config(
    page_title="Header Test",
    page_icon="🧪",
    layout="wide"
)

st.title("Interactive Header Test")

# Create a fresh instance of the header
header = AdvancedInteractiveHeader()

# Display available methods for debugging
st.write("## Available Methods")
methods = [method for method in dir(header) if callable(getattr(header, method)) and not method.startswith("__")]
st.write(methods)

# Test calling the render method
st.write("## Testing Header Render")
try:
    header.render()
    st.success("Header rendered successfully!")
except Exception as e:
    st.error(f"Error rendering header: {str(e)}")
    st.error(f"Error type: {type(e)}")
    
    # Check if render_quick_actions exists
    st.write("### Checking for render_quick_actions method")
    if hasattr(header, 'render_quick_actions'):
        st.write("✅ Method exists in the class")
        
        # Check if method is callable
        try:
            method = getattr(header, 'render_quick_actions')
            st.write(f"Method details: {method}")
            st.write(f"Is callable: {callable(method)}")
        except Exception as e:
            st.error(f"Error getting method: {str(e)}")
    else:
        st.error("❌ Method does not exist in the class")
        
    # Display the source code of render_quick_actions if possible
    st.write("### Method Source Code")
    try:
        import inspect
        method = getattr(header, 'render_quick_actions')
        source = inspect.getsource(method)
        st.code(source, language="python")
    except Exception as e:
        st.error(f"Could not get source: {str(e)}") 