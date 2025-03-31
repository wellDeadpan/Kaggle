import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from config import FEATURES


def load_and_process(n_lags: int):
    df = pd.read_csv("data/Rainfall.csv")
    df.columns = df.columns.str.strip().str.lower()
    df['rainfall'] = df['rainfall'].astype(str).str.strip().str.lower()
    df['rain_flag'] = df['rainfall'].map({'yes': 1, 'no': 0})
    days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Create a list with the month for each day
    month_labels = [month for month, days in zip(months, days_in_month) for _ in range(days)]
    df['Month'] = month_labels

    print(df.isna().sum())  # Check NaNs
    print(df.applymap(lambda x: not pd.api.types.is_number(x)).sum())  # Non-numeric check
    #print(~np.isfinite(df[FEATURES]).all().all())  # Check for inf/-inf

    for feature in FEATURES:
        for lag in range(1, n_lags + 1):
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

            # Generate moving averages for 3, 4, 5 days
        for window in [3, 4, 5]:
            df[f'{feature}_ma{window}'] = df[feature].rolling(window).mean()
    df.dropna(inplace=True)
    return df


def value_counts(df):
    return df['rainfall'].value_counts().to_dict()


def summary_table(df):
    return df.groupby('rainfall')[df.columns.difference(['rainfall', 'rain_flag'])].describe().transpose()

def dot_product_corr(df):
    df = df[FEATURES]
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    dot_corr = np.dot(X.T, X) / (X.shape[0] - 1)
    sns.heatmap(dot_corr, annot=True, cmap='coolwarm')
    plt.title("Correlation via Dot Product (Standardized Features)")
    plt.show()
    return pd.DataFrame(dot_corr, index=df.columns, columns=df.columns)

def feature_plot(df):
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    figs = []

    for feature in FEATURES:
        g = sns.relplot(
            data=df,
            x='day', y=feature,
            col='Month', kind='line',
            col_wrap=4, height=3, aspect=1.2,
            facet_kws={'sharey': True}
        )
        g.fig.suptitle(f'Daily {feature.capitalize()} by Month', fontsize=16)
        plt.subplots_adjust(top=0.9)
        figs.append(g.fig)

    return figs


def split_heatmap(df, feature_prefixes=["", "lag"]):
    """
        Plots a heatmap of standardized features, including lagged versions.

        Parameters:
        - df: pandas DataFrame with original and lagged features
        - feature_prefixes: list of strings to filter columns (e.g., ['', 'lag'] to include both original and lagged)

        Returns:
        - Matplotlib figure with annotated heatmap
        """
    sorted_df = df.sort_values(by='rain_flag', ascending=False).reset_index(drop=True)

    # Select features that match the prefix
    all_features = [col for col in df.columns if any(p in col for p in feature_prefixes) and col != 'rain_flag'
                    and col != 'rainfall' and col != 'Month']

    # Standardize selected features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sorted_df[all_features])
    scaled_df = pd.DataFrame(scaled_data, columns=all_features)
    # Count how many Rain days we have
    days = sorted_df['rain_flag'].sum()

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
        if sorted_df.loc[i, 'rain_flag'] == 1:
            ax.plot(i + 0.5, -0.5, marker='o', color='green', markersize=5)
            ax.text(i + 0.5, -0.5, 'R', ha='center', va='center', color='green', fontsize=8)

    ax.set_title('Daily Weather Heatmap (Rainy Days Annotated)', fontsize=16)
    ax.set_ylabel('Features')
    ax.set_xlabel('Days (Rain → No Rain)')
    fig.tight_layout()

    return fig


