import streamlit as st

# Set page configuration at the very beginning - must be called first
st.set_page_config(
    page_title="Test App - No Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# No dashboard initialization here

import pandas as pd
import numpy as np

st.title("Test App - No Dashboard")
st.write("This is a test app to verify that all dashboard components have been removed.")

# Add a simple UI element
st.button("Click me")

# Display some random data
data = pd.DataFrame(
    np.random.randn(10, 3),
    columns=['A', 'B', 'C']
)
st.dataframe(data)

# Main function that would normally include navigation etc.
def main():
    st.sidebar.title("Navigation")
    st.sidebar.write("This is a test sidebar with no dashboard components.")
    
if __name__ == "__main__":
    main() 