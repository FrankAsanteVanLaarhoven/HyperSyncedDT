print("Script starting")  # Debug 1
import streamlit as st
print("Streamlit imported")  # Debug 2

print("Before page config")  # Debug 3
st.set_page_config(
    page_title="HyperSyncDT",
    page_icon="🏭",
    layout="wide"
)
print("After page config")  # Debug 4

def main():
    print("Main function started")  # Debug 5
    st.write("Basic test")
    
    # Just try one button
    if st.button("Test Button"):
        print("Button clicked")  # Debug 6
        st.write("Button was clicked!")

print("Before main check")  # Debug 7
if __name__ == "__main__":
    print("Running main")  # Debug 8
    main()
print("Script completed")  # Debug 9
