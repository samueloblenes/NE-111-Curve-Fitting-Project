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
    
# Defining function that handles data input

def data_entry(entry_method, unique_prefix): 
    input_df = pd.DataFrame(columns=["X-Axis", "Y-Axis"])
    if entry_method == "Manual entry":
        input_df = st.data_editor(st.session_state.df, num_rows="dynamic", key=f"{unique_prefix}_editor") #unique prefix gives a different key to the widgets, was inititally going to sue So i could have seperat ones on each tab but ended up scrapping that idea
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key=f"{unique_prefix}_uploader")
        if uploaded_file != None:
            df_uploaded = pd.read_csv(uploaded_file) # Read the CSV file to a pandas DataFrame
            num_cols = df_uploaded.shape[1] # check if dataframe has the correct dimensions
            
            if num_cols == 0:
                st.error("No columns found in uploaded file.")
            elif num_cols == 1: # for only 1 column treat as Y-Axis, create default X-Axis
                df_uploaded.columns = ['Y-Axis'] # rename columns
                df_uploaded['X-Axis'] = range(1, len(df_uploaded) + 1)
                input_df = df_uploaded[['X-Axis', 'Y-Axis']]  # reorder columns
                st.warning("Only one column found, assumed it is Y-Axis, X-Axis assigned as sequential integers starting from 1.")  
            elif num_cols == 2:
                df_uploaded.columns = ['X-Axis', 'Y-Axis']
                input_df = df_uploaded
            else:
                st.error("Uploaded CSV has more than 2 columns, please enter a file containing only 2 columns for x and y data respectively")

            st.write("Uploaded Data:")
            st.dataframe(input_df)

    return input_df

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
tab1, tab2= st.tabs(["Auto Fit", "Manual Fit"])

#Initializing session state variables for things that I dont want reset everythime strealit updated
if "Dataconfirmed" not in st.session_state:
    st.session_state.Dataconfirmed = False # Session state varaible to keep track of if the user has confirmed the entered data
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["X-Axis", "Y-Axis"]) # session state variable to store the data being entered
if "dist_name" not in st.session_state: # initialize selected sistribution so that it ramins constant across manual and auto tabs
    st.session_state.dist_name = "norm"
if "num_points" not in st.session_state:
    st.session_state.num_points = 300 



########## Data entry ##########
entry_method = st.selectbox("Choose to enter data manualy or upload a CSV file",("Manual entry","Upload CSV file"),key="auto_entry_method")
input_df = data_entry(entry_method, "auto") # call data entry function

# Confirm entered data, if there is no data entered, display an error and ask the user to input data
col1, col2 = st.columns(2)
with col1:
    st.write("Click confirm to update the graph")
    confirm_clicked = st.button("Confirm")
    if confirm_clicked:
        # Remove rows where ALL cells are None, to check if there are actually any numerical values, not just a bunch of aded empty rows
        cleaned_df = input_df.dropna() #remove None values from dataframe
        if not cleaned_df.empty and cleaned_df.notna().any().any(): # Check if at least one cell is not empty
            st.session_state.df = cleaned_df
            st.session_state.Dataconfirmed = True
        else:
           st.error("Please enter some data to confirm") #if not data has been enetred (the data frame only contains None values or no values) display this error message
            
# Clear entered data
with col2:
     st.write("Click clear to clear all entered data")
     if st.button("Clear"):
        st.session_state.df = pd.DataFrame(columns=["X-Axis", "Y-Axis"]) # Reset pandas dataframe 
        st.session_state.Dataconfirmed = False # Set confirmation variable to False
        st.rerun() # force streamlit to rerun so that the input table is cleared imediatly

st.session_state.dist_name = st.selectbox(
        "Choose a distribution", 
        ["norm", "expon", "gamma", "beta", "uniform", 
        "weibull_min", "poisson", "binom", "chi2", "lognorm"]
    )


########## Tab1 Auto curve fitting ##########
with tab1:    
    st.text("Auto curv fitting. create default paramters and make so that if this tab is selected fitting is done with the default paramater, include some plot appearance customization")
             
########## Tab2, Manual curve fitting ##########
with tab2:
    st.text("Configure curve fitting for manual mode, have sliders and option for every posible fitting paramater")
    
    st.session_state.num_points = st.number_input(
        "Curve resolution",
        value=300,
        step=1,
        format="%d"
    )
    
    st.text("increasing the curve resolution provides a smoother fitted curve")
        
########## Graph display section ##########
st.divider()

if st.session_state.Dataconfirmed and not st.session_state.df.empty: # If data is confirmed and the dataframe is not empty, display the graph and table
    col1, col2 = st.columns([1,3])
    col1.subheader("Data")
    col2.subheader("Distribution")

    # prepare/clean entered date
    df_to_plot = st.session_state.df.copy() # define the dataframe to plot
    for col in df_to_plot.columns:
        df_to_plot[col] = pd.to_numeric(df_to_plot[col], errors='coerce')
        df_to_plot = df_to_plot.dropna()
    
    orig_df, fit_df = fit(df_to_plot, st.session_state.dist_name, int(st.session_state.num_points))
    
    with col1:
        st.dataframe(orig_df)
        st.dataframe(fit_df)
    
    with col2:
        # if no numerical data was entered, display and error
        if df_to_plot.empty:
            st.error("No numeric data available to plot.") # Error message if no data is enetred and the program proceeds to try and graph
            
        else:
           
            fig, ax = plt.subplots()
            ax.hist(orig_df["Y-Axis"], bins=30, density=True, alpha=0.5, label="Data Histogram") # create histogram of the entered data
            ax.plot(fit_df["X-Axis"], fit_df["Y-Axis"], color='red', lw=2, label="Fitted Curve") # Overlay the fitted curve
            # Display in Streamlit
            st.pyplot(fig)
    
else: 
    st.write("Once you enter and confirm your data a graph will apear here") # if data is not confirmed, display this message

        


