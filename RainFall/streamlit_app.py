# streamlit_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os

# Add the app directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.pipeline import (
    load_and_process,
    summary_table,
    dot_product_corr,
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
df = load_and_process(n_lags)

# Section: Rainfall Distribution
with st.expander("📊 Rainfall Distribution"):
    st.bar_chart(df['rainfall'].value_counts())

# Section: Summary Table by Rainfall
with st.expander("🧾 Summary Statistics by Rainfall"):
    summary = summary_table(df)
    st.dataframe(summary)

# Section: Correlation Heatmap
with st.expander("📎 Dot Product Correlation"):
    st.write("Standardized correlation using dot product (same as Pearson on scaled data).")
    corr = dot_product_corr(df[FEATURES].select_dtypes(include='number'))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(corr), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Section: Heatmap Split by Rainfall
with st.expander("🧊 Heatmap Split by Rain/No Rain"):
    # Plot heatmap
    heatmap_fig = split_heatmap(df)
    st.pyplot(heatmap_fig)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")