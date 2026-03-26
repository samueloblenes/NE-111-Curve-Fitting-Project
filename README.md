# Statistical Curve Fitting Tool 📈

This is a streamlit web application for curve fitting that lets users see how different statistical distributions fit their data. This tool allows users to enter data manually or importe it from a CSV file and quickly visualize different probability distributions

## Features
- **Users can input data manually using an interactive input table or import data from a .CSV file**
- **Users can select from several distributions including**:
  - normal
  - exponential
  - gamma
  - beta
  - uniform
  - lognormal
  - Weibull
  - chi‑square
  - Laplace
  - Cauchy
- **“Auto Fit” tab performs standard distribution fitting, while the “Manual Fit” tab allows more control over**:
  - resolution
  - x‑range
  - fixed location, scale, and shape parameters
- **Visulization**
  - the probability density function overlaid on a density‑normalized histogram of the data
- **Plot appearance customization using sliders and drop downs**
  - colours
  - transparency
  - line styles
  - orientation
  - custom titles and axis labels

 ## Dependencies
- matplotlib
- numpy
- scipy
- Streamlit
