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
    #Initializing variable
    if "Dataconfirmed" not in st.session_state:
        st.session_state.Dataconfirmed = False # Session state varaible to keep track of if the user has confirmed the entered data
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["X-Axis", "Y-Axis"]) # session state variable to store the data being entered

    col3, col4 = st.columns(2)

# Data entry section
    with col3:
        entry_method = st.selectbox("Choose to enter data manualy or upload a CSV file",("Manual entry","Upload CSV file"))

        # Manual entry mode
        if  entry_method == "Manual entry":
            st.session_state.df = pd.DataFrame(columns=["column 1", "column2"]) # create the data frame as a sesion state variable do it remains constant
            edited_df = st.data_editor(st.session_state.df, num_rows="dynamic") # make the data frame editable 

            # Show mesages based on confirmation
            if not st.session_state.Dataconfirmed:
                st.write("Click confrim to display the graph")
            else:
                st.write("Click Clear to enter a new dataset")
        
            col3_1, col3_2 = st.columns(2)

            # Confirm entered data, if there is no data entered, display an error and ask the user to input data 
            with col3_1:
                confirm_clicked = st.button("Confirm")
                if confirm_clicked:
                    if not edited_df.empty:
                        st.session_state.df = edited_df.copy()
                        st.session_state.Dataconfirmed = True
                    else:
                        st.session_state.Dataconfirmed = False
                        st.markdown("""
                            <style>.error-box {
                                background-color: #FF746C;
                                padding: 1.2em;
                                border-radius: 10px;
                                border-left: 6px solid #FFE66D;
                                font-size: 1.1em;
                                line-height: 1.6em;
                                color: #333;
                                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
                            </style>""", unsafe_allow_html=True)
                        st.markdown("<div class='error-box'>Please enter some data to confirm</div>", unsafe_allow_html=True)
                    
            # Clear entered data
            with col3_2:
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
                st.warning("No numeric data available to plot after conversion.")
            else:
                fig, ax = plt.subplots()
                df_to_plot.plot(ax=ax)
                st.pyplot(fig)

    else: 
        st.write("Enter and confirm your data to view the graph") # if data is not confirmed, display this message
        
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

