from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations

def generate_features(df, n_lags=3):
    df = df.copy()
    for feature in ['pressure', 'maxtemp', ...]:
        for lag in range(1, n_lags + 1):
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        df[f'{feature}_ma{n_lags}'] = df[feature].rolling(n_lags).mean()
    return df.dropna()

def generate_all_interactions(df, features):
    """
    Automatically create all pairwise interaction terms for given features.

    Parameters:
        df (pd.DataFrame): Input dataframe
        features (list): Features to generate interactions between

    Returns:
        pd.DataFrame: DataFrame with added interaction features
    """
    df = df.copy()
    for f1, f2 in combinations(features, 2):
        df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def compute_mda_importance(model, X_val, y_val, feature_names):
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    importance_df = importance_df.sort_values(by='importance_mean', ascending=False)
    return importance_df

def plot_mda_importance(importance_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=importance_df,
        y='feature',
        x='importance_mean',
        xerr=importance_df['importance_std'],
        ax=ax,
        orient='h'
    )
    ax.tick_params(axis='y', labelsize=5)
    ax.set_title('Variable Importance (Mean Decrease Accuracy)', fontsize=14)
    ax.set_xlabel('Importance (mean decrease accuracy)')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    return fig

def intx_features(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interact = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    # Convert to DataFrame for easier inspection
    X_interact_df = pd.DataFrame(X_interact, columns=feature_names)
    return X_interact_df




