import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Defining function that accepts a pandas dataframe and a distribution, then returns the fitted dataframe

def fit(df, dist_name, num_points, x_col = "X-Axis", y_col = 'Y-Axis'):
    x_axis = df[x_col].dropna().values # get data from the X-Axis columns, remove None values
    y_axis = df[y_col].dropna().values # get data from the Y-Axis columns, remove None values

    distribution = getattr(stats, dist_name) #get the distribution from the name passed to the function

    params = distribution.fit(y_axis) # Fits the distribution to the cureve, Gives estimated paramaters

    x_fit = np.linspace(np.min(x_axis), np.max(x_axis), num_points) # create evenly spaces points for the x-axis, num_points controls how many points

    #checks whether the given distribution has a pdf method (used for continuous distributions) or a pmf method (used for discrete distributions). 
    #compute the fitted probability values at the points x_fit using the parameters stored in params and the correct method.
    if hasattr(distribution, 'pdf'):
        y_fit = distribution.pdf(x_fit, *params)
    elif hasattr(distribution, 'pmf'):
        y_fit = distribution.pmf(x_fit, *params)

    #store fited date in a pandas dataframe
    fit_df = pd.DataFrame({x_col: x_fit, y_col: y_fit}) # fit data
    orig_df = df.copy() # Entered data

    return orig_df, fit_df
    


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

    col3, col4 = st.columns(2)
    
    # Data entry section
    with col3:
        entry_method = st.selectbox("Choose to enter data manualy or upload a CSV file",("Manual entry","Upload CSV file"))

        # Manual entry mode
        if  entry_method == "Manual entry":
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
                    st.session_state.df = pd.DataFrame(columns=edited_df.columns) # Reset pandas dataframe 
                    st.session_state.Dataconfirmed = False # Set confirmation variable to False
                     
    # configure curve fitting and graph apearance
    with col4:
        st.text("Configure curve fitting")
        
        num_points = st.number_input(
            "Curve resolution",
            value=300,
            step=1,
            format="%d"
        )
        
        st.text("increasing the curve resolution provides a smoother fitted curve")
        
        dist_name = st.selectbox(
            "Choose a distribution", 
            ["norm", "expon", "gamma", "beta", "uniform", 
            "weibull_min", "poisson", "binom", "chi2", "lognorm"]
        )
        
        st.divider()
      
            
# Graph display section
    st.divider()
    
    if st.session_state.Dataconfirmed and not st.session_state.df.empty: # If data is confirmed and the dataframe is not empty, display the graph and table
        col1, col2 = st.columns([1,3])
        col1.subheader("Data")
        col2.subheader("Distribution")

        with col1:
            st.dataframe(st.session_state.df)

        with col2:
            # prepare/clean entered date
            df_to_plot = st.session_state.df.copy() # define the dataframe to plot
            for col in df_to_plot.columns:
                df_to_plot[col] = pd.to_numeric(df_to_plot[col], errors='coerce')
            df_to_plot = df_to_plot.dropna()

            # if no numerical data was entered, display and error
            if df_to_plot.empty:
                st.error("No numeric data available to plot.") # Error message if no data is enetred and the program proceeds to try and graph
                
            else:
                orig_df, fit_df = fit(df_to_plot, dist_name, int(num_points))

                fig, ax = plt.subplots()
                ax.hist(orig_df["Y-Axis"], bins=30, density=True, alpha=0.5, label="Data Histogram") # create histogram of the entered data
                ax.plot(fit_df["X-Axis"], fit_df["Y-Axis"], color='red', lw=2, label="Fitted Curve") # Overlay the fitted curve
                # Display in Streamlit
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

