# streamlit_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os

# Add the app directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.eda_utils import (
    summary_table,
    split_heatmap
)
from app.config import FEATURES

# Page setup
st.set_page_config(page_title="Rainfall Analysis Dashboard", layout="wide")
st.title("🌧️ Rainfall Analysis Dashboard")

# Sidebar
st.sidebar.header("User Inputs")
n_lags = st.sidebar.slider("Number of Lag Days", min_value=1, max_value=10, value=3)

# Data processing
flnm = 'F:\\GitHub\\Kaggle\\RainFall\\data\\train.csv'
df = pd.read_csv(flnm)

# Section: Rainfall Distribution
with st.expander("📊 Rainfall Distribution"):
    st.bar_chart(df['rainfall'].value_counts())

# Section: Summary Table by Rainfall
with st.expander("🧾 Summary Statistics by Rainfall"):
    summary = summary_table(df)
    st.dataframe(summary)


# Section: Heatmap Split by Rainfall
with st.expander("🧊 Heatmap Split by Rain/No Rain"):
    # Plot heatmap
    heatmap_fig = split_heatmap(df)
    st.pyplot(heatmap_fig)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")