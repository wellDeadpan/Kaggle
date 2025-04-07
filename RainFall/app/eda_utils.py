import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import base64

def basic_summary(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df.describe(include='all').to_dict()
    }

def plot_histograms(df):
    plots = {}
    for col in df.select_dtypes(include="number").columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plots[col] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    return plots

def value_counts(df):
    return df['rainfall'].value_counts().to_dict()

def summary_table(df):
    return df.groupby('rainfall')[df.columns.difference(['rainfall'])].describe().transpose()

def split_heatmap(df, feature_prefixes=["", "lag"]):
    """
        Plots a heatmap of standardized features, including lagged versions.

        Parameters:
        - df: pandas DataFrame with original and lagged features
        - feature_prefixes: list of strings to filter columns (e.g., ['', 'lag'] to include both original and lagged)

        Returns:
        - Matplotlib figure with annotated heatmap
        """
    sorted_df = df.sort_values(by='rainfall', ascending=False).reset_index(drop=True)

    # Select features that match the prefix
    all_features = [col for col in df.columns if any(p in col for p in feature_prefixes)
                    and col != 'rainfall']

    # Standardize selected features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sorted_df[all_features])
    scaled_df = pd.DataFrame(scaled_data, columns=all_features)
    # Count how many Rain days we have
    days = sorted_df['rainfall'].sum()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        scaled_df.T,
        cmap='coolwarm',
        cbar_kws={'label': 'Standardized Value'},
        xticklabels=False,
        ax=ax
    )

    ax.axvline(x=days, color='black', linestyle='--', linewidth=2)

    for i in range(sorted_df.shape[0]):
        if sorted_df.loc[i, 'rainfall'] == 1:
            ax.plot(i + 0.5, -0.5, marker='o', color='green', markersize=5)
            ax.text(i + 0.5, -0.5, 'R', ha='center', va='center', color='green', fontsize=8)

    ax.set_title('Daily Weather Heatmap (Rainy Days Annotated)', fontsize=16)
    ax.set_ylabel('Features')
    ax.set_xlabel('Days (Rain → No Rain)')
    fig.tight_layout()

    return fig