import streamlit as st
import Housing_Brief as app
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
import seaborn as sns
from mlxtend.plotting import scatterplotmatrix
import numpy as np
import plotly.graph_objects as go
st.header("Visulizations")
st.divider()
column_to_filter_by = st.selectbox("Choose a column to filter by", app.df.columns)
filter_options = st.multiselect("Filter by", options=app.df[column_to_filter_by].unique())

# Filtering data based on selection
if filter_options:
    filtered_data = app.df[app.df[column_to_filter_by].isin(filter_options)]
else:
    filtered_data = app.df

st.dataframe(filtered_data)
st.write(f"{filtered_data["SalePrice"].count()} results are displayed.")

st.divider()


import plotly.express as px
import pandas as pd

# Assuming you have your data in a DataFrame 'app.df'
fig = px.scatter_matrix(
    app.df, 
    dimensions=app.df.columns, 
    title='Scatterplot Matrix',
    # color=app.df['some_column'],  # Use a categorical column for grouping
    color_discrete_sequence=px.colors.qualitative.Plotly,
    width=1000,  # Width in pixels
    height=800  # Nice color palette
)

fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(opacity=0.7))
fig.update_xaxes(tickangle=45) 

st.plotly_chart(fig)
st.divider()
import plotly.figure_factory as ff
st.subheader('Correlation Matrix')
corr_matrix = app.df.corr()
fig = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    annotation_text=corr_matrix.round(2).values,
    colorscale='Viridis'
)
fig.update_layout(xaxis=dict(tickangle=45))
st.plotly_chart(fig)
st.divider()

st.subheader("Learning Curve")
st.image("Pictures\Learning_Curve_Housing.png")
st.divider()
st.subheader("Validation Curve")
st.image("Pictures\Housing_Validation_Curve.png")
st.write("R2 Score = 0.8412 ± 0.0607")
st.divider()


st.divider()
st.subheader("KMeans++ Elbow Plot")
st.image("Pictures\K_means++.png")
