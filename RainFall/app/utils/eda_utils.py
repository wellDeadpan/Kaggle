import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import base64
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional

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

def pca_plots(
        X: pd.DataFrame,
        y: pd.Series,
        n_components: int = 2,
        figsize: Tuple[int, int] = (10, 7),
        colors: List[str] = ['yellow', 'skyblue'],
        labels: List[str] = ['No Rain', 'Rain'],
        loading_scale: float = 8,
        alpha: float = 0.6) -> Tuple[plt.Figure, PCA]:
    """Create PCA biplot with loadings visualization.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        n_components (int, optional): Number of PCA components. Defaults to 2.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 7).
        colors (List[str], optional): Colors for classes. Defaults to ['yellow', 'skyblue'].
        labels (List[str], optional): Labels for classes. Defaults to ['No Rain', 'Rain'].
        loading_scale (float, optional): Scaling factor for loading arrows. Defaults to 8.
        alpha (float, optional): Transparency for scatter plots. Defaults to 0.6.

    Returns:
        Tuple[plt.Figure, PCA]: Matplotlib figure and fitted PCA object
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Create DataFrame for PCA results
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=[f'PC{i + 1}' for i in range(n_components)]
    )
    pca_df["Target"] = y.values

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Plot scatter points for each class
    for i, label in enumerate([0, 1]):
        subset = pca_df[pca_df["Target"] == label]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            c=colors[i],
            label=labels[i],
            alpha=alpha
        )

    # Add loadings (feature contributions)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    for i, feature in enumerate(X.columns):
        # Plot loading arrows
        plt.arrow(
            0, 0,
            loadings[i, 0] * loading_scale,
            loadings[i, 1] * loading_scale,
            color='red',
            alpha=0.5,
            head_width=0.02
        )

        # Add feature labels
        plt.text(
            loadings[i, 0] * loading_scale,
            loadings[i, 1] * loading_scale,
            feature,
            color='red',
            ha='center',
            va='center'
        )

    # Add labels and title
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    plt.title('PCA Biplot with Loadings')

    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    return fig, pca


def plot_feature_boxplots(
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = 'Rain',
        col_wrap: int = 4,
        height: int = 3,
        show_outliers: bool = False,
        title: str = "Boxplots of Scaled Features by Outcome",
        palette: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
) -> sns.FacetGrid:
    """
    Create faceted boxplots for features grouped by target variable.

    Args:
        X (pd.DataFrame): Feature matrix (scaled or unscaled)
        y (pd.Series): Target variable
        target_name (str, optional): Name of target variable. Defaults to 'Rain'.
        col_wrap (int, optional): Number of plots per row. Defaults to 4.
        height (int, optional): Height of each subplot. Defaults to 3.
        show_outliers (bool, optional): Whether to show outliers. Defaults to False.
        title (str, optional): Main title for the plot.
        palette (str, optional): Color palette for the plots.
        figsize (Tuple[int, int], optional): Figure size. If None, calculated automatically.

    Returns:
        sns.FacetGrid: The resulting facet grid object

    Example:
        >>> g = plot_feature_boxplots(X_scaled, y,
        ...                          title="Weather Features by Rainfall",
        ...                          palette="Set3")
        >>> plt.show()
    """
    try:
        # Create copy of X to avoid modifying original
        X_plot = X.copy()

        # Add target variable, ensuring index alignment
        X_plot[target_name] = y.reset_index(drop=True)

        # Melt DataFrame to long format
        df_long = X_plot.melt(
            id_vars=target_name,
            var_name='Feature',
            value_name='Value'
        )

        # Calculate figure dimensions if not provided
        if figsize is None:
            n_features = len(X.columns)
            n_rows = (n_features + col_wrap - 1) // col_wrap
            figsize = (col_wrap * 4, n_rows * height)

        # Create faceted boxplot
        g = sns.catplot(
            data=df_long,
            x=target_name,
            y='Value',
            col='Feature',
            kind='box',
            col_wrap=col_wrap,
            height=height,
            aspect=1.2,
            sharey=False,
            showfliers=show_outliers,
            palette=palette
        )

        # Adjust layout and add title
        g.fig.set_size_inches(figsize)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(title, fontsize=16)

        # Rotate x-axis labels if needed
        for ax in g.axes:
            ax.tick_params(axis='x', rotation=45)

        # Add feature-specific titles and adjust ylabels
        for ax, feature in zip(g.axes, X.columns):
            # Make feature names more readable
            feature_title = feature.replace('_', ' ').title()
            ax.set_title(feature_title)
            ax.set_ylabel('Value')

        return g

    except Exception as e:
        raise ValueError(f"Error creating boxplots: {str(e)}")


# Additional utility function for adding statistical annotations
def add_statistical_annotations(g: sns.FacetGrid, test_func=None):
    """
    Add statistical test results to the boxplots.

    Args:
        g (sns.FacetGrid): The facet grid object from plot_feature_boxplots
        test_func (callable, optional): Statistical test function to use
    """
    from statannot import add_stat_annotation

    if test_func is None:
        from scipy import stats
        test_func = stats.mannwhitneyu

    for ax in g.axes:
        try:
            data = ax.get_children()[0].get_data()
            feature = ax.get_title()

            add_stat_annotation(
                ax,
                data=g.data[g.data['Feature'] == feature],
                x=g.data.columns[0],
                y='Value',
                box_pairs=[(0, 1)],
                test=test_func,
                text_format='star',
                loc='outside'
            )
        except Exception as e:
            print(f"Could not add statistics for {feature}: {str(e)}")

def plot_correlation_matrix(df, figsize=(10, 8), title="Correlation Matrix (Scaled Data)", annot=True):
    """
    Plots a heatmap of the correlation matrix for the given DataFrame.

    Parameters:
        df (pd.DataFrame): Scaled numeric dataframe
        figsize (tuple): Figure size (width, height)
        title (str): Title of the plot
        annot (bool): Whether to annotate correlation values

    Returns:
        fig: The matplotlib figure object
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap='coolwarm', fmt=".2f", center=0, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig