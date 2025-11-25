################################################################################
# Samuel O'Blenes NE 111 Final project
#
# Description:
#    This program is a curve fitting web app made using the streamlit library.
#    It accepts data either through manuall entry or by uploading a .CSV file 
#    and allows the user to configure a number of setting
################################################################################

from turtle import color
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Samuel O'Blenes
########## Function definitions ##########

# Defining function that accepts a pandas dataframe and a distribution, then returns the fitted dataframe
def fit(df, dist_name, xi = None, xf = None, num_points =  300, x_col = "X-Axis", y_col = 'Y-Axis'): 
    x_axis = df[x_col].dropna().values # get data from the X-Axis columns, remove None values
    y_axis = df[y_col].dropna().values # get data from the Y-Axis columns, remove None values

    distribution = getattr(stats, dist_name) #get the distribution from the name passed to the function
        
    params = distribution.fit(y_axis)  #gives the paramaters 
 
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
def plot(data_confirmed, dataframe, dist_name, plt_appearance_settings, hist_appearance_settings, axis_settings, num_points  = 300, xi = None, xf = None): 
    if data_confirmed and not dataframe.empty: # If data is confirmed and the dataframe is not empty, display the graph and table
        
        # prepare/clean entered date
        df_to_plot = dataframe # define the dataframe to plot
        df_to_plot["X-Axis"] = pd.to_numeric(df_to_plot["X-Axis"])
        df_to_plot["Y-Axis"] = pd.to_numeric(df_to_plot["Y-Axis"])
        
        orig_df, fit_df = fit(df_to_plot, dist_name, xi = xi, xf = xf, num_points = int(num_points)) 
        
        # if no numerical data was entered, display and error
        if df_to_plot.empty:
            st.error("No numeric data available to plot.") # Error message if no data is enetred and the program proceeds to try and graph
            
        else:
            
            fig, ax = plt.subplots()
            ax.hist(orig_df["Y-Axis"], bins=30, density=True, **hist_appearance_settings) # create histogram of the entered data, unpack settings dictionary
            
            if hist_appearance_settings["orientation"] == "horizontal":
                ax.plot(fit_df["Y-Axis"], fit_df["X-Axis"], **plt_appearance_settings) # Overlay the fitted curve over a hprizontal histogram, unpack settings dictionary
            else:
                ax.plot(fit_df["X-Axis"], fit_df["Y-Axis"], **plt_appearance_settings) # Overlay the fitted curve over a normal vertical histogram, unpack settings dictionary

            # Set axis labels and title from the axis_setting dictionary
            ax.set_xlabel(axis_settings["xlabel"])
            ax.set_ylabel(axis_settings["ylabel"])
            ax.set_title(axis_settings["title"])
            
            
            # set axis limits with some padding based on entered data
            
            nonzero = fit_df["Y-Axis"].astype(float) != 0
            if nonzero.any():
                last_nonzero_idx = nonzero[nonzero].index[-1]
                df_trimmed = fit_df.loc[:last_nonzero_idx]
            else:
                df_trimmed = fit_df.iloc[0:0]


            # Depending on the orientation of the histogram, set the appropriate axis limits, 
            if hist_appearance_settings["orientation"] == "horizontal":
                ax.set_ylim(0, max(df_trimmed["X-Axis"]) * 1.1)
            else:
                ax.set_xlim(0, max(df_trimmed["X-Axis"]) * 1.1) 

            # Display in Streamlit
            st.pyplot(fig)
            
        col1, col2 = st.columns(2)
        col1.subheader("Data")
        col2.subheader("Distribution")

        with col1:
            st.write("Entered Data")
            st.dataframe(orig_df)
        with col2:
            st.write("Fit data")
            st.dataframe(fit_df)
    else: 
        st.write("Once you enter and confirm your data a graph will apear here") # if data is not confirmed, display this message

# defining functions that collect graph appearance settings
def get_combined_plot_settings(key):
    
    colours = st.expander("Colour settings", expanded=False)
    dim = st.expander("Line and bar settings", expanded=False)
    title_axis = st.expander("Title and axis labels", expanded=False)
    
    
    
    with colours:
        colcol1, colcol2 = st.columns(2)
        with colcol1:
            color = st.color_picker("Bar Color", "#1f77b4", key=key+"_hist_color")
        with colcol2:
            edgecolor = st.color_picker("Edge Color", "#000000", key=key+"_hist_edge")
            
        dist_color = st.color_picker("Distribution Line Color", "#ff7f0e", key=key+"_dist_color")
        
        alpha = st.slider("Histogram Transparency", 0.0, 1.0, 0.8, key=key+"_hist_alpha")
        dist_alpha = st.slider("Distribution Transparency", 0.0, 1.0, 1.0, key=key+"_dist_alpha")
        
    with dim:
        histtype = st.selectbox("Histogram Type", options=["bar", "barstacked", "step", "stepfilled"], key=key+"_hist_histtype")
        plotstyle = st.selectbox("Plot Style", options=["solid", "dashed", "dashdot", "dotted"], key=key+"_plot_style")
        orientation = st.selectbox("Orientation", options=["vertical", "horizontal"], key=key+"_hist_orientation")
        linewidth = st.slider("Edge Line Width", 0.0, 5.0, 1.0, key=key+"_hist_line")
        rwidth = st.slider("Relative Bar Width", 0.1, 1.0, 0.8, key=key+"_hist_rwidth")
        dist_linewidth = st.slider("Distribution Line Width", 0.0, 5.0, 2.0, key=key+"_dist_linewidth")
        
    with title_axis:
        title = st.text_input("Plot Title", "Title", key=key+"_title")
        xlabel = st.text_input("X-axis Label", "x-label", key=key+"_xlabel")
        ylabel = st.text_input("Y-axis Label", "y-label", key=key+"_ylabel")
        label = st.text_input("Histogram Label", "Histogram", key=key+"_hist_label")
        dist_label = st.text_input("Distribution Label", "Distribution", key=key+"_dist_label")
    
        
    appearance_settings = {
        "color": color,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
        "alpha": alpha,
        "rwidth": rwidth,
        "histtype": histtype,
        "orientation": orientation,
        "label": label
    }
    
    axis_settings = {
        "xlabel": xlabel,
        "ylabel": ylabel,
        "title": title,
    }
    
    dist_appearance_settings = {
        "color": dist_color,
        "linewidth": dist_linewidth,
        "alpha": dist_alpha,
        "label": dist_label,
        "linestyle": plotstyle
    }

    return appearance_settings, axis_settings, dist_appearance_settings
    

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
        
        temp_df = input_df.copy()
        if temp_df["X-Axis"].isna().all():# If all X-Axis missing, auto-generate
            temp_df["X-Axis"] = range(1, len(temp_df) + 1)
            st.warning("No X-Axis data entered. X values have been assigned as sequential integers starting from 1.")
            
        cleaned_df = temp_df.dropna(subset=["Y-Axis"]) #remove None values from dataframe
        
        if not temp_df["X-Axis"].astype(str).str.isnumeric().any() or not temp_df["Y-Axis"].astype(str).str.isnumeric().any() and temp_df["X-Axis"].isnull().any(): # Check for non-numeric value
            st.error("Non-numeric values detected in the data. Please enter only numeric values.")
        elif not cleaned_df.empty and cleaned_df.notna().any().any(): # Check if at least one cell is not empty
            if cleaned_df.shape[0] < temp_df.shape[0]:
                st.warning("Some rows with missing Y-Axis values have been removed.") # warn user if any rows were removed due to missing Y-Axis values
            st.session_state.df = cleaned_df
            st.session_state.Dataconfirmed = True
            st.info("Data confirmed successfully!")
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
    r"f(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)", # norm
    r"f(x) = \exp(-x), \quad x \geq 0", # expon
    r"f(x, a) = \frac{1}{\Gamma(a)} x^{a-1} \exp(-x), \quad x \geq 0", # gamma
    r"f(x, a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} x^{a-1}(1-x)^{b-1},\quad 0 < x < 1", # beta
    r"f(x) = 1, \quad 0 \leq x \leq 1", # uniform
    r"f(x,s) = \frac{1}{s x \sqrt{2\pi}} \exp\left(-\frac{(\ln x)^2}{2s^2}\right),\quad x > 0", # lognorm
    r"f(x,c) = c x^{c-1} \exp(-x^{c}),\quad x \geq 0", # weibull_min
    r"f(x,k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2-1} e^{-x/2},\quad x \geq 0", # chi2
    r"f(x) = \frac{1}{2} \exp(-|x|)", # laplace
    r"f(x) = \frac{1}{\pi (1 + x^2)}", # cauchy
]
st.latex(latex_formulas[ ["norm", "expon", "gamma", "beta", "uniform", 
        "lognorm", "weibull_min", "chi2", "laplace", "cauchy"].index(st.session_state.dist_name)])


st.divider()
st.subheader("Configure graph and fit settings")

# Create Tabs
tab1, tab2= st.tabs(["Auto Fit", "Manual Fit"])

########## Tab1 Auto curve fitting ##########
with tab1:    
    tab1_col1, tab1_col2 = st.columns([1,3])
    with tab1_col1:
        hist_appearance_settings, axis_settings, dist_appearance_settings = get_combined_plot_settings(key="auto")
   
    with tab1_col2:
        plot(st.session_state.Dataconfirmed, st.session_state.df, st.session_state.dist_name, dist_appearance_settings, hist_appearance_settings, axis_settings, st.session_state.num_points) # call plot function to display graph
     
########## Tab2, Manual curve fitting ##########
with tab2:
    st.text("Configure curve fitting settings")
    
    st.session_state.num_points = st.slider("Curve resolution", max_value = 300, value=100, step=1)
    
    xi, xf = st.slider(
        label = "Select range of x values", 
        value = (0,25)
    )
    
    fix_loc = st.checkbox("Fix location parameter (loc)?")
    if fix_loc:
        fixed_loc = st.number_input("Location (loc) value", value=0.0, step=0.1, format="%.4f")
    else:
        fixed_loc = None

    fix_scale = st.checkbox("Fix scale parameter (scale)?")
    if fix_scale:
        fixed_scale = st.number_input("Scale (scale) value", value=1.0, step=0.1, format="%.4f")
    else:
        fixed_scale = None
    
    st.divider()
        
    tab1_col1, tab1_col2 = st.columns([1,3])
    with tab1_col1:
        hist_appearance_settings, hist_axis_settings, dist_appearance_settings = get_combined_plot_settings(key="man")

    with tab1_col2:
        plot(st.session_state.Dataconfirmed, st.session_state.df, st.session_state.dist_name, dist_appearance_settings, hist_appearance_settings, axis_settings, st.session_state.num_points,xi, xf) # call plot function to display graph 

# Samuel O'Blenes NE 111 Final project


