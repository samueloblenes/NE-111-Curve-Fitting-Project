import streamlit as st

# Page formating
st.set_page_config(
    page_title="NE 111 Project",
    page_icon="📊",
    layout="wide",  # optional: "centered" or "wide"
    initial_sidebar_state="expanded"  # optional
)

page_title = "📊 Curve Fitting Web App"
st.title(page_title)

# Create Tabs
tab1, tab2, tab3 = st.tabs(["Auto Fit", "Manual Fit", "Settings"])


# Tab1, Auto curve fitting
with tab1:
    st.header("Auto Curve Fitting")

    col3, col4 = st.columns(2)
    
    st.divider()
    col1, col2 = st.columns([1,3])
    col1.subheader("Data")
    col2.subheader("Distribution")
  
# Tab2 Manual curve fitting
with tab2:    
    st.header("Manual Curve Fitting")
    
    col3, col4 = st.columns(2)

    st.divider()
    col1, col2 = st.columns([1,3])
    col1.subheader("Data")
    col2.subheader("Distribution")

# Tab3 Settings
with tab3:
    st.header("Settings")

