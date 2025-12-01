################################################################################
# Samuel O'Blenes NE 111 Final project
# Dec 2 2025
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

########## Function definitions ##########

# Defining function that accepts a pandas dataframe and a distribution, then returns the fitted dataframe
def fit(df, dist_name, settings, col="Values"):
    y = df[col].dropna().astype(float).values
    if y.size == 0:
        return df, pd.DataFrame(columns=[col, "PDF"])

    distribution = getattr(stats, dist_name)

    # Extract settings with defaults
    xi = settings.get("xi", None)
    xf = settings.get("xf", None)
    num_points = settings.get("num_points", 300)
    fixed_loc  = settings.get("fixed_loc", None)
    fixed_scale= settings.get("fixed_scale", None)
    fit_kwargs = {}
    if fixed_loc is not None:
        fit_kwargs["floc"] = fixed_loc
    if fixed_scale is not None:
        fit_kwargs["fscale"] = fixed_scale
        
        
    # Extract distribution-specific shape parameters 
    shape_mapping = {
        "fixed_a": "fa",
        "fixed_b": "fb",
        "fixed_s": "fs",
        "fixed_c": "fc",
        "fixed_df": "fdf"
    }
    for settings_key, fit_arg in shape_mapping.items():
        if settings.get(settings_key) is not None:
            fit_kwargs[fit_arg] = settings[settings_key]   
    
    # check weather the user fixed all parameters, in which case fit is not used
    total_required = distribution.numargs + 2  # 
    if len(fit_kwargs) == total_required and all(k.startswith("f") for k in fit_kwargs):
        params = tuple(fit_kwargs[key] for key in sorted(fit_kwargs))
        st.warning("You fixed all distribution parameters, nothing left to optimize and scipy.fit was not used")
    else:
        params = distribution.fit(y, **fit_kwargs)

    if xi is None or xf is None:
        x_min, x_max = float(np.min(y)), float(np.max(y))
    else:
        x_min, x_max = float(xi), float(xf)

    x_fit = np.linspace(x_min, x_max, int(num_points))
    y_fit = distribution.pdf(x_fit, *params)

    fit_df = pd.DataFrame({col: x_fit, "PDF": y_fit})
    return df, fit_df, params
    
# Defining function that handles data input
def data_entry(entry_method, unique_prefix):
    input_df = pd.DataFrame(columns=["Values"])
    if entry_method == "Manual entry":
        # Start from confirmed df, ensure correct columns
        base_df = st.session_state.df
        if list(base_df.columns) != ["Values"]:
            base_df = pd.DataFrame(columns=["Values"])
            
        column_config = {"Values": st.column_config.NumberColumn("Values",format="%.6f")}    

        dynamic_key = st.session_state.get("auto_editor_key", f"{unique_prefix}_editor") # key for data editor, changes each time clear is pressed to try and get the table to reset
        input_df = st.data_editor(base_df, num_rows="dynamic",column_config = column_config, hide_index = True, key = dynamic_key)

    else:
        try:
            uploaded_file = st.file_uploader(
                "Choose a CSV file", type="csv",
                key=f"{unique_prefix}_uploader",
            )
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
        except:
            input_df = pd.DataFrame()
            
    return input_df

# defining function that plots the entered data and the fit data
def plot(data_confirmed, dataframe, dist_name, plt_appearance_settings, hist_appearance_settings, axis_settings, settings): 
    if data_confirmed and not dataframe.empty: # If data is confirmed and the dataframe is not empty, display the graph and table
        
        # prepare/clean entered date
        df_to_plot = dataframe # define the dataframe to plot
        df_to_plot["Values"] = pd.to_numeric(df_to_plot["Values"])
        
        orig_df, fit_df, params = fit(df_to_plot, dist_name, settings=settings, col="Values") # call fit function to get the fitted dataframe
        
        # if no numerical data was entered, display and error
        if df_to_plot.empty:
            st.error("No numeric data available to plot.") # Error message if no data is entered and the program proceeds to try and graph
            
        else:
            
            fig, ax = plt.subplots()
            hist_data, bin_edges, _ = ax.hist(orig_df["Values"], bins=30, density=True, **hist_appearance_settings) # create histogram of the entered data, unpack settings dictionary
            
            # Determine error
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            distribution = getattr(stats, dist_name)
            pdf_at_bins = distribution.pdf(bin_centers, *params)
            errors = np.abs(hist_data - pdf_at_bins)
            avg_error = float(np.mean(errors))
            max_error = float(np.max(errors))
            
            # Prepare parameter text for display
            param_names = []
            if getattr(distribution, "shapes", None):
                param_names.extend([s.strip() for s in distribution.shapes.split(",")])
            param_names.extend(["loc", "scale"])

            param_text_parts = [f"{name} = {value:.4g}" for name, value in zip(param_names, params)]

            # Add fixed shape parameters only
            for name, key in [("a", "fixed_a"), ("b", "fixed_b"), ("s", "fixed_s"), ("c", "fixed_c"), ("df", "fixed_df")]:
                if settings.get(key) is not None:
                    param_text_parts.append(f"{name} = {settings[key]:.4g} (fixed)")

            param_text = ", ".join(param_text_parts)
            
            
            # Plot the fitted curve over the histogram
            if hist_appearance_settings["orientation"] == "horizontal":
                ax.plot(fit_df["PDF"], fit_df["Values"], **plt_appearance_settings) # Overlay the fitted curve over a horizontal histogram, unpack settings dictionary
            else:
                ax.plot(fit_df["Values"], fit_df["PDF"], **plt_appearance_settings) # Overlay the fitted curve over a normal vertical histogram, unpack settings dictionary

            # Set axis labels and title from the axis_setting dictionary
            ax.set_xlabel(axis_settings["xlabel"])
            ax.set_ylabel(axis_settings["ylabel"])
            ax.set_title(axis_settings["title"])
            
            # Set face and edge colors, and boarder thickness
            fig.set_facecolor(axis_settings["facecolor"])
            ax.set_facecolor(axis_settings["bgcolor"])
            fig.set_edgecolor(axis_settings["figedgecolor"])
            fig.set_linewidth(axis_settings["boarderthickness"])
            
            # set axis limits with some padding based on entered data
            nonzero = fit_df["PDF"].astype(float) != 0
            if nonzero.any():
                last_nonzero_idx = nonzero[nonzero].index[-1]
                df_trimmed = fit_df.loc[:last_nonzero_idx]
            else:
                df_trimmed = fit_df.iloc[0:0]

            # Depending on the orientation of the histogram, set the appropriate axis limits, 
            if not df_trimmed.empty:
                if hist_appearance_settings["orientation"] == "horizontal":
                    # horizontal: data on y-axis
                    ax.set_ylim(0, max(df_trimmed["Values"]) * 1.1)
                else:
                    # vertical: data on x-axis
                    ax.set_xlim(0, max(df_trimmed["Values"]) * 1.1)

                # Display legend if enabled
                if axis_settings["legend"]:
                    ax.legend([plt_appearance_settings["label"], hist_appearance_settings["label"]])    

            # Display in Streamlit
            st.pyplot(fig)
            
            # Display error metrics below the plot in html text boxes
            col_err1, col_err2 = st.columns(2)
            with col_err1:
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #ccc;
                        border-radius:4px;
                        padding:8px 12px;
                        background-color:#6d77ffff;
                    ">
                        <strong>Average error</strong><br>
                        {avg_error:.4g}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_err2:
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #ccc;
                        border-radius:4px;
                        padding:8px 12px;
                        background-color:#6d77ffff;
                    ">
                        <strong>Maximum error</strong><br>
                        {max_error:.4g}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # Display fitted parameters below the plot in an html text box
            st.markdown(
                f"""
                <div style="
                    border:1px solid #ccc;
                    border-radius:4px;
                    padding:8px 12px;
                    margin-top:8px;
                    background-color:#6d77ffff;
                ">
                    <strong>Fitting parameters</strong><br>
                    {param_text}
                </div>
                """,
                unsafe_allow_html=True,
            )
            
    else: 
        st.warning("Once you enter and confirm your data a graph will appear here") # if data is not confirmed, display this message

# defining functions that collect graph appearance settings
def get_combined_plot_settings(key):
    
    colours = st.expander("Colour settings", expanded=False)
    dim = st.expander("Line and bar settings", expanded=False)
    title_axis = st.expander("Title and axis labels", expanded=False)
    
    
    
    with colours:
        colcol1, colcol2 = st.columns(2)
        with colcol1:
            color = st.color_picker("Bar Colour", "#1f77b4", key=key+"_hist_color",help="Colour of the histogram bars")
            dist_color = st.color_picker("Distribution Line Colour", "#ff7f0e", key=key+"_dist_color", help="Colour of the distribution curve line")
            bgcolor = st.color_picker("Background Colour", "#ffffff", key=key+"_hist_face",help="Colour of the area inside the plot axes")
        with colcol2:
            edgecolor = st.color_picker("Bar Edge Colour", "#000000", key=key+"_hist_edge")
            facecolor = st.color_picker("Face Colour", "#ffffff", key=key+"_plot_face",help="Colour of the area outside the plot axes")
            figedgecolor = st.color_picker("Border Colour", "#FFFFFF", key=key+"_fig_edge",help="Colour of the border around the plot")            
        
        
        alpha = st.slider("Histogram Transparency", 0.0, 1.0, 0.8, key=key+"_hist_alpha",help="Transparency of the histogram bars")
        dist_alpha = st.slider("Distribution Transparency", 0.0, 1.0, 1.0, key=key+"_dist_alpha",help="Transparency of the distribution curve")
        border_thickness = st.slider("Border Thickness", 0.0, 5.0, 1.0, key=key+"_border_thickness", help="Thickness of the border around the plot")
        
    with dim:
        histtype = st.selectbox("Histogram Type", options=["bar", "barstacked", "step", "stepfilled"], key=key+"_hist_histtype")
        plotstyle = st.selectbox("Plot Style", options=["solid", "dashed", "dashdot", "dotted"], key=key+"_plot_style")
        orientation = st.selectbox("Orientation", options=["vertical", "horizontal"], key=key+"_hist_orientation")
        linewidth = st.slider("Edge Line Width", 0.0, 5.0, 1.0, key=key+"_hist_line", help="Width of the edges of the histogram bars")
        rwidth = st.slider("Relative Bar Width", 0.1, 1.0, 0.8, key=key+"_hist_rwidth")
        dist_linewidth = st.slider("Distribution Line Width", 0.0, 5.0, 2.0, key=key+"_dist_linewidth",help="Width of the distribution curve line")
        
    with title_axis:
        title = st.text_input("Plot Title", "Title", key=key+"_title")
        xlabel = st.text_input("X-axis Label", "x-label", key=key+"_xlabel")
        ylabel = st.text_input("Y-axis Label", "y-label", key=key+"_ylabel")
        label = st.text_input("Histogram Label", "Histogram", key=key+"_hist_label", help="This label will appear in the legend for the histogram")
        dist_label = st.text_input("Distribution Label", "Distribution", key=key+"_dist_label", help="This label will appear in the legend for the distribution curve")
        legend = st.checkbox("Show Legend", value=True, key=key+"_legend",help="Check to display the legend on the plot")
    
        
    appearance_settings = {
        "color": color,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
        "alpha": alpha,
        "rwidth": rwidth,
        "histtype": histtype,
        "orientation": orientation,
        "label": label,
    }
    
    axis_settings = {
        "xlabel": xlabel,
        "ylabel": ylabel,
        "title": title,
        "legend": legend,
        "facecolor": facecolor,
        "figedgecolor": figedgecolor,
        "bgcolor": bgcolor,
        "boarderthickness": border_thickness
    }
    
    dist_appearance_settings = {
        "color": dist_color,
        "linewidth": dist_linewidth,
        "alpha": dist_alpha,
        "label": dist_label,
        "linestyle": plotstyle
    }

    return appearance_settings, axis_settings, dist_appearance_settings
    
########## Page formatting/ setup ##########
st.set_page_config(
    page_title="Samuel O'Blenes NE111 Final Project - Curve Fitting webapp",
    page_icon="📊",
    layout="wide",  # optional: "centered" or "wide"
    initial_sidebar_state="expanded"  # optional
)

st.title("📊 Samuel O'Blenes NE111 Final Project - Curve Fitting webapp")

with st.sidebar:
    st.header("How to use this app")

    st.markdown("""
1. **Enter your data**
- Choose **Manual entry** or **Upload CSV file**.
- For CSVs:
    - Upload a file with **one column** of numeric values.
    - The column will be treated as the data sample.
- For manual entry:
    - Edit the **Values** column in the table.
- Click **Confirm** to clean and lock in the data for plotting.

2. **Select a distribution**
- Use the **Choose a distribution** dropdown.
- The formula shown below the dropdown is the PDF of the selected SciPy distribution.
- The app fits this distribution to the **Values** using SciPy's `.fit` method.

3. **View or refine the fit**
- In **Auto Fit**, the app:
    - Fits the selected distribution to the Values.
    - Plots a histogram of the Values.
    - Overlays the fitted PDF as a smooth curve.
- In **Manual Fit**, you can:
    - Adjust the **curve resolution** (number of points).
    - Set the **x-range** for the PDF.
    - Fix and manually set scale, location, and shape paramaters
        - Note, if you fix all the paramaters, there is nothing left to optomize and scipy.fit will not be used, the distribution will be fit using only your entered paramaters.
    
4. **Customize the plot**
- Use the 3 settings expanders to view and change:
    - Colours and transparency of the histogram and curve.
    - Line style, bar width, and orientation (vertical or horizontal).
    - Plot title and axis labels.

5. **Clear or update data**
- Use the **Clear** button to reset all entered data.
   
**Notes:**
- Rows with **missing values** are automatically removed.
- Only numeric **Values** are allowed, non-numeric entries are removed.
- The plot and fit only update after you click **Confirm**.
- After entering data, you must press the clear button before editing or entering new date and then reconfirm for it to be reflected in the graph. 
    """)


########## Session state variables ##########

#Initializing session state variables for things that I dont want reset every time streamlit updates
if "Dataconfirmed" not in st.session_state:
    st.session_state.Dataconfirmed = False # Session state variable to keep track of if the user has confirmed the entered data
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["Values"]) # session state variable to store the data being entered
if "dist_name" not in st.session_state: 
    st.session_state.dist_name = "norm" # selected distribution
if "num_points" not in st.session_state:
    st.session_state.num_points = 300 # number of points to use for plotting the fitted curve
if "has_non_numeric" not in st.session_state:
    st.session_state.has_non_numeric = False # flag for non numeric data entry

########## Data entry ##########
st.divider()

cola,colb = st.columns(2)

with cola:
    entry_method = st.selectbox("Chose a data entry method",("Manual entry","Upload CSV file"),key="auto_entry_method")
    input_df = data_entry(entry_method, "auto") # call data entry function
    
    col1,col2 = st.columns(2)
    with col1:
        # Confirm entered data, if valid, store in session state, else display error
        confirm_clicked = st.button("Confirm", help = "Update graph" )
        if confirm_clicked:
            temp_df = input_df.copy()
                        
            if len (temp_df.columns) != 1:
                st.error("Please ensure there is exactly one column of data.")
            else:
                # deal with any non numeric characters in the dataframe
                temp_df.columns = ["Values"]  # Handles both CSV and editor
                temp_df["Values"] = pd.to_numeric(temp_df["Values"], errors="coerce")
                cleaned_df = temp_df.dropna(subset=["Values"])
        
                if cleaned_df.shape[0] < temp_df.shape[0]:
                    st.session_state.has_non_numeric = True
                
                if not cleaned_df.empty and cleaned_df["Values"].notna().any():
                    st.session_state.df = cleaned_df
                    st.session_state.Dataconfirmed = True
                else:
                    st.error("Please enter some data to confirm")
                    
            
                
    # display warnings/info messages if The non-numeric flag is true or data is confirmed
    if st.session_state.has_non_numeric:
            st.warning("Non-numeric values were found and have been removed.")
            
    if st.session_state.Dataconfirmed:
            st.info("Data confirmed successfully!")

    with col2:            
        # Clear entered data when clear button is pressed
        if st.button("Clear", help="Clear all data"):
                st.session_state.df = pd.DataFrame(columns=["Values"])
                st.session_state.Dataconfirmed = False
                st.session_state.has_non_numeric = False
                
                if entry_method == "Manual entry":
                    st.session_state["auto_editor_key"] = str(np.random.rand()) # force new key for data editor to reset it
                    for key in ["auto_editor", "auto_uploader", "auto_entry_method"]:
                        st.session_state.pop(key, None)
                st.rerun()

with colb:
    # Distribution selection
    st.session_state.dist_name = st.selectbox(
            "Choose a distribution", 
            ["norm", "expon", "gamma", "beta", "uniform", 
            "lognorm", "weibull_min", "chi2", "laplace", "cauchy"]
        )

    # Display the LaTeX formula for the selected distribution
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

tab1, tab2= st.tabs(["Auto Fit", "Manual Fit"])

########## Tab1 Auto curve fitting ##########
with tab1:    
    tab1_col1, tab1_col2 = st.columns([1,3])
    with tab1_col1:
        hist_appearance_settings, axis_settings, dist_appearance_settings = get_combined_plot_settings(key="auto")
   
    with tab1_col2:
        auto_settings = {
            "num_points": st.session_state.num_points,
            "xi": None,
            "xf": None,
            "fixed_loc": None,
            "fixed_scale": None,
            "fixed_a": None,      
            "fixed_b": None,     
            "fixed_s": None,      
            "fixed_c": None,      
            "fixed_df": None
        }
        plot(st.session_state.Dataconfirmed, st.session_state.df, st.session_state.dist_name, dist_appearance_settings, hist_appearance_settings, axis_settings, auto_settings) # call plot function to display graph
     
########## Tab2, Manual curve fitting ##########
with tab2:
    
    # Define manual fit settings dictionary    
    manual_settings = {
            "num_points": st.session_state.num_points,
            "xi": None,
            "xf": None,
            "fixed_loc": None,
            "fixed_scale": None,
            "fixed_a": None,      
            "fixed_b": None,     
            "fixed_s": None,      
            "fixed_c": None,      
            "fixed_df": None      
        }

    #  shape parameters corresponding to each distributions
    dist_shapes = {
    "gamma": [("a", "fixed_a")],
    "beta": [("a", "fixed_a"), ("b", "fixed_b")],
    "lognorm": [("s", "fixed_s")],
    "weibull_min": [("c", "fixed_c")],
    "chi2": [("df", "fixed_df")]
    }

    colman1, colman2 = st.columns(2)
    with colman1:
    
        fix_loc = st.checkbox("Configure Fix location parameter (loc)?")
        if fix_loc:
            manual_settings["fixed_loc"] = st.number_input("Location (loc) value", value=0.0, step=0.1, format="%.4f")
        else:
            manual_settings["fixed_loc"] = None

        fix_scale = st.checkbox("Configure Fix scale parameter (scale)?")
        if fix_scale:
            manual_settings["fixed_scale"] = st.number_input("Scale (scale) value", value=1.0, step=0.1, format="%.4f")
        else:
            manual_settings["fixed_scale"] = None

        # depending on the selected distribution, display the appropriate shape parameter options
        shapes = dist_shapes.get(st.session_state.dist_name, [])
        for shape_name, setting_key in shapes:
            fix_shape = st.checkbox(f"Fix {shape_name} parameter?")
            if fix_shape:
                manual_settings[setting_key] = st.number_input(f"{shape_name} value", value=1.0)
            else:
                manual_settings[setting_key] = None
           
           
                
    with colman2:
        st.session_state.num_points = st.slider("Curve resolution", max_value = 300, value=100, step=1)
    
        manual_settings["xi"], manual_settings["xf"] = st.slider(
            label="Select range of x values", 
            value=(0,25)
            )
    
    st.divider()
        
    tab1_col1, tab1_col2 = st.columns([1,3])
    with tab1_col1:
        hist_appearance_settings, axis_settings, dist_appearance_settings = get_combined_plot_settings(key="man")

    with tab1_col2:    
        plot(st.session_state.Dataconfirmed, st.session_state.df, st.session_state.dist_name, dist_appearance_settings, hist_appearance_settings, axis_settings, manual_settings)

# Samuel O'Blenes NE 111 Final project


