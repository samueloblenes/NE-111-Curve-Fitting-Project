# Samuel O'Blenes NE 111 Curve Fitting Project

This web app is a curve fitting tool built with Streamlit that lets users see how different probability distributions fit their data. Data can be entered manually in an editable table or imported from a CSV file. The app accepts CSV files with exactly one column and treats it as a single “Values” column, automatically dropping missing entries and ignoring non‑numeric values during fitting.

Users can choose from several SciPy distributions (including normal, exponential, gamma, beta, uniform, lognormal, Weibull, chi‑square, Laplace, and Cauchy) and see the corresponding probability density function overlaid on a density‑normalized histogram of the data (so bar heights represent probability density, not raw counts). The “Auto Fit” tab performs standard distribution fitting, while the “Manual Fit” tab allows control over resolution, x‑range, and optional fixed location/scale parameters

The app also allows users to fully customize the appearance of the output plot featuring: colour pickers, transparency sliders, line styles, orientation (vertical or horizontal), and custom titles and axis labels. the original and fitted data are displayed in tables below the plot so users can inspect the data alongside the graph.
