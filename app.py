################################################################################
# Samuel O'Blenes NE 111 Final project
#
# Description:
#    This program is a curve fitting web app made using the streamlit library.
#    It accepts data either through manuall entry or by uploading a .CSV file 
#    and allows the user to configure a number of setting
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


########## Function definitions ##########

# Defining function that accepts a pandas dataframe and a distribution, then returns the fitted dataframe

def fit(df, dist_name, xi = None, xf = None, num_points =  300, x_col = "X-Axis", y_col = 'Y-Axis', fixed_loc=None, fixed_scale=None): 
    x_axis = df[x_col].dropna().values # get data from the X-Axis columns, remove None values
    y_axis = df[y_col].dropna().values # get data from the Y-Axis columns, remove None values

    distribution = getattr(stats, dist_name) #get the distribution from the name passed to the function

    params = distribution.fit(y_axis, floc = fixed_loc, fscale = fixed_scale)  #gives the paramaters 
 
    if xi is None or xf is None: # determine  min and max values depending on if the user entered them in manual mode or is in auto mode
        x_min, x_max = np.min(x_axis), np.max(x_axis)
    else:
        x_min, x_max = xi, xf 
    
    x_fit = np.linspace(x_min, x_max, num_points) # create evenly spaced points for the x-axis, num_points controls how many points
    y_fit = distribution.pdf(x_fit, *params) # .pdf method for continuous distribution

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

    return input_df

# defining function that plots the entered data and the fit data

def plot(data_confirmed, dataframe, dist_name, num_points  = 300, xi = None, xf = None, fixed_loc = None, fixed_scale = None): 
    if data_confirmed and not dataframe.empty: # If data is confirmed and the dataframe is not empty, display the graph and table
        col1, col2 = st.columns([1,3])
        col1.subheader("Data")
        col2.subheader("Distribution")
    
        # prepare/clean entered date
        df_to_plot = dataframe # define the dataframe to plot
        for col in df_to_plot.columns:
            df_to_plot[col] = pd.to_numeric(df_to_plot[col], errors='coerce')
            df_to_plot = df_to_plot.dropna()
        
        orig_df, fit_df = fit(df_to_plot, dist_name, xi = xi, xf = xf, num_points = int(num_points), fixed_loc = fixed_loc, fixed_scale = fixed_scale) 
        
        with col1:
            st.write("Entered Data")
            st.dataframe(orig_df)
            st.write("Fit data")
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

# Samuel O'Blenes
########## Page formating/ setup ##########
st.set_page_config(
    page_title="NE 111 Project",
    page_icon="📊",
    layout="wide",  # optional: "centered" or "wide"
    initial_sidebar_state="expanded"  # optional
)

page_title = "📊 Curve Fitting Web App"
st.title(page_title)

########## Session state variables ##########

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
st.subheader("Data Entry")
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
        "lognorm", "weibull_min", "chi2", "laplace", "cauchy"]
    )

latex_formulas = [
    # norm: scipy.stats.norm (location=μ, scale=σ)
    r"f(x\,|\,\mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
    # expon: scipy.stats.expon (loc, scale=1/λ)
    r"f(x\,|\,\lambda) = \lambda e^{-\lambda x} \quad x \geq 0",
    # gamma: scipy.stats.gamma (shape=a, loc, scale=θ)
    r"f(x\,|\,a,\theta) = \frac{1}{\Gamma(a)\theta^a} x^{a-1} \exp\left(-\frac{x}{\theta}\right) \quad x \geq 0",
    # beta: scipy.stats.beta (a=α, b=β, loc, scale)
    r"f(x\,|\,\alpha,\beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1},\quad 0 < x < 1",
    # uniform: scipy.stats.uniform (loc=a, scale=b-a)
    r"f(x\,|\,a,b) = \frac{1}{b-a} \quad a \leq x \leq b",
    # lognorm: scipy.stats.lognorm (s=σ, loc, scale=exp(μ))
    r"f(x\,|\,\mu,\sigma) = \frac{1}{x \sigma \sqrt{2\pi }} \exp\left(-\frac{(\ln x - \mu )^2}{2\sigma^2}\right),\quad x>0",
    # weibull_min: scipy.stats.weibull_min (c=shape, loc, scale)
    r"f(x\,|\,c) = c x^{c-1} e^{-x^{c}},\quad x \geq 0",
    # chi2: scipy.stats.chi2 (df=k, loc, scale)
    r"f(x\,|\,k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2-1} e^{-x/2},\quad x \geq 0",
    # laplace: scipy.stats.laplace (loc=μ, scale=b)
    r"f(x\,|\,\mu, b) = \frac{1}{2b} \exp\left(-\frac{|x-\mu|}{b}\right)",
    # cauchy: scipy.stats.cauchy (loc=x0, scale=γ)
    r"f(x\,|\,x_0,\gamma) = \frac{1}{\pi \gamma} \left[ 1 + \left(\frac{x-x_0}{\gamma}\right)^2 \right]^{-1}",
]

st.latex(latex_formulas[ ["norm", "expon", "gamma", "beta", "uniform", 
        "lognorm", "weibull_min", "chi2", "laplace", "cauchy"].index(st.session_state.dist_name)])


st.divider()
st.subheader("Configure graph and fit settings")

# Create Tabs
tab1, tab2= st.tabs(["Auto Fit", "Manual Fit"])

########## Tab1 Auto curve fitting ##########
with tab1:    
    st.text("Auto curv fitting. create default paramters and make so that if this tab is selected fitting is done with the default paramater, include some plot appearance customization")

    st.divider()
    plot(st.session_state.Dataconfirmed,st.session_state.df, st.session_state.dist_name, st.session_state.num_points) # call plot function to display graph 
########## Tab2, Manual curve fitting ##########
with tab2:
    st.text("Configure curve fitting settings")
    
    st.session_state.num_points = st.slider("Curve resolution", max_value = 300, value=100, step=1)
    
    xi, xf = st.slider(
        label = "Select range of x values", 
        value = (0,25)
    )
    
    fixed_loc = None 
    
    fixed_scale = None
    
    st.divider()

    st.text("Configure graph appearance")
    
    st.divider()
    plot(st.session_state.Dataconfirmed,st.session_state.df, st.session_state.dist_name, st.session_state.num_points, xi = xi, xf = xf, fixed_loc = fixed_loc, fixed_scale = fixed_scale) # call plot function to display graph 


# Samuel O'Blenes NE 111 Final project


