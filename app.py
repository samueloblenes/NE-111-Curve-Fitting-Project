import streamlit as st
import pandas as pd

# Page formating
st.set_page_config(
    page_title="NE 111 Project",
    page_icon="📊",
    layout="wide",  # optional: "centered" or "wide"
    initial_sidebar_state="expanded"  # optional
)

page_title = "📊 Curve Fitting Web App"
st.title(page_title)

# create description text box
st.markdown(
    """
    <style>
        .description-box {
            background-color: #4ECDC4;
            padding: 1.2em;
            border-radius: 10px;
            border-left: 6px solid #FFE66D;
            font-size: 1.1em;
            line-height: 1.6em;
            color: #333;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Render description text box
st.markdown(f"<div class='description-box'>{"Select either Auto fit, or manual fit. bellow and enter your data, or a CSV file to begin"}</div>", unsafe_allow_html=True)


# Create Tabs
tab1, tab2, tab3 = st.tabs(["Auto Fit", "Manual Fit", "Settings"])


# Tab1, Auto curve fitting
with tab1:
    st.header("Auto Curve Fitting")

    col3, col4 = st.columns(2)

# Data entry section
    with col3:
        if st.selectbox("Choose to enter data manualy or upload a CSV file",("Manual entry","Upload CSV file")) == "Manual entry":
            df = pd.DataFrame(columns=["column 1", "column2"]) # create the data frame

            edited_df = st.data_editor(df) # make the data fram editable
            edited_df = st.data_editor(df, num_rows="dynamic") # allow aditiion of rows
            
# Graph display section
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

