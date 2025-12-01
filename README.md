# Samuel O'Blenes NE 111 Curve Fitting Project

This web app is a curve fitting tool built with Streamlit that lets users see how different probability distributions fit their data. Data can be entered manually in an editable table or imported from a CSV file with exactly one column. It automatically drops missing entries and ignors non‑numeric values during fitting.

Users can choose from several SciPy distributions (including normal, exponential, gamma, beta, uniform, lognormal, Weibull, chi‑square, Laplace, and Cauchy) and see the corresponding probability density function overlaid on a density‑normalized histogram of the data (so bar heights represent probability density). The “Auto Fit” tab performs standard distribution fitting, while the “Manual Fit” tab allows control over resolution, x‑range, and fixed location, scale, and shape parameters

The app also allows users to fully customize the appearance of the output plot including: colours, transparency, line styles, orientation (vertical or horizontal), and custom titles and axis labels. The paramaters, average error, and maximum error are displayed below the graph
