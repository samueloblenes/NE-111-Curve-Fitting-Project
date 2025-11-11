import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


########## Tab1, Auto curve fitting ##########
with tab1:
    #Initializing variables
    if "Dataconfirmed" not in st.session_state:
        st.session_state.Dataconfirmed = False # Session state varaible to keep track of if the user has confirmed the entered data
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["X-Axis", "Y-Axis"]) # session state variable to store the data being entered

    

# Data entry section
    col3, col4 = st.columns(2)
    
    with col3:
        entry_method = st.selectbox("Choose to enter data manualy or upload a CSV file",("Manual entry","Upload CSV file"))

        # Manual entry mode
        if  entry_method == "Manual entry":
            st.session_state.df = pd.DataFrame(columns=["X-Axis", "Y-Axis"]) # create the data frame as a sesion state variable do it remains constant
            edited_df = st.data_editor(st.session_state.df, num_rows="dynamic") # make the data frame editable 

            # Confirm entered data, if there is no data entered, display an error and ask the user to input data
            col3_1, col3_2 = st.columns(2)
            with col3_1:
                st.write("Click confirm to update the graph")
                confirm_clicked = st.button("Confirm")
                if confirm_clicked:
                    # Remove rows where ALL cells are None, to check if there are actually any numerical values, not just a bunch of aded empty rows
                    cleaned_df = edited_df.dropna(how="all")
                    # Check if at least one cell is not empty
                    if not cleaned_df.empty and cleaned_df.notna().any().any():
                        st.session_state.df = edited_df.copy()
                        st.session_state.Dataconfirmed = True
                    else:
                       st.error("Please enter some data to confirm") #if not data has been enetred (the data frame only contains None values or no values) display this error message
                    
            # Clear entered data
            with col3_2:
                 st.write("Click clear to clear all entered data")
                 if st.button("Clear"):
                    st.session_state.df = pd.DataFrame(columns=["column 1", "column2"]) # Reset pandas dataframe 
                    st.session_state.Dataconfirmed = False # Set confirmation variable to False
            
# Graph display section
    st.divider()
    
    if st.session_state.Dataconfirmed and not st.session_state.df.empty: # If data is confirmed and the dataframe is not empty, display the graph and table
        col1, col2 = st.columns([1,3])
        col1.subheader("Data")
        col2.subheader("Distribution")

        with col1:
            st.dataframe(st.session_state.df)

        with col2:
            df_to_plot = st.session_state.df.copy() # define the dataframe to plot
            df_to_plot = df_to_plot.iloc[:, 1:] # remove first column in the data frame
            for col in df_to_plot.columns:
                df_to_plot[col] = pd.to_numeric(df_to_plot[col], errors='coerce')
            df_to_plot = df_to_plot.dropna()
            if df_to_plot.empty:
                st.error("No numeric data available to plot.") # Error message if no data is enetred and the program proceeds to try and graph
            else:
                fig, ax = plt.subplots()
                df_to_plot.plot(ax=ax)
                st.pyplot(fig)

    else: 
        st.write("Once you enter and confirm your data a graph will apear here") # if data is not confirmed, display this message
        
########## Tab2 Manual curve fitting ##########
with tab2:    
    
    col3, col4 = st.columns(2)

    st.divider()
    col1, col2 = st.columns([1,3])
    col1.subheader("Data")
    col2.subheader("Distribution")

# Tab3 Settings
with tab3:
    pass

