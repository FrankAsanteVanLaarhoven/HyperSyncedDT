import streamlit as st

print("Starting script...")  # Debug print

def main():
    st.write("Header Test")
    
    # Basic layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Column 1")
        if st.button("Test Button"):
            st.success("It works!")
    
    with col2:
        st.write("Column 2")
        st.metric("Sample Metric", 42)

if __name__ == "__main__":
    main()
    print("Script completed")  # Debug print
