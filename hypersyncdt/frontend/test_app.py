import streamlit as st

def main():
    st.title("Test App")
    st.write("Hello World!")
    
    if st.button("Click me"):
        st.success("Button clicked!")

if __name__ == "__main__":
    main() 