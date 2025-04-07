import pandas as pd
import io
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


async def read_input_data(file=None,
                          id=None, day=None, pressure=None, maxtemp=None, temparature=None, mintemp=None, dewpoint=None,
                          humidity=None, cloud=None, sunshine=None, winddirection=None, windspeed=None, rainfall=None):
    if file:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    else:
        df = pd.DataFrame([{
            "id": id,
            "day": day,
            "pressure": pressure,
            "temparature": temparature,
            "maxtemp": maxtemp,
            "mintemp": mintemp,
            "dewpoint": dewpoint,
            "humidity": humidity,
            "cloud": cloud,
            "sunshine": sunshine,
            "winddirection": winddirection,
            "windspeed": windspeed,
            "rainfall": rainfall
        }])
    df.columns = df.columns.str.strip().str.lower()
    return df

def generate_features(df, features, n_lags=3):
    df = df.copy()
    for feature in features:
        for lag in range(1, n_lags + 1):
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

            # Generate moving averages for 3, 4, 5 days
        for window in [3, 4, 5]:
            df[f'{feature}_ma{window}'] = df[feature].rolling(window).mean()
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

def intx_features(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interact = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    # Convert to DataFrame for easier inspection
    X_interact_df = pd.DataFrame(X_interact, columns=feature_names)
    return X_interact_df

def prepare_model_data(df, features, n_lags=5):
    df = df.copy()
    df = generate_features(df, features, n_lags)

    # Ensure 'month' is retained for one-hot encoding
    keep_cols = [col for col in df.columns if col.startswith(tuple(f"{f}_lag" for f in features))] + [col for col in df.columns if col.startswith(tuple(f"{f}_ma" for f in features))]
    X = df[keep_cols]
    # Step 3: One-hot encode categorical columns (like 'month')
    y = df['rainfall']
    return X, y

def prepare_test_data(df, features, n_lags=5):
    # Step 1: Load processed data with lag features and 'month'
    #df = load_and_process(flnm, n_lags=n_lags)
    # Step 2: Drop target columns from X
    df = df.copy()
    # Ensure 'month' is retained for one-hot encoding
    idlist = df['id']
    keep_cols = [col for col in df.columns if col.startswith(tuple(f"{f}_lag" for f in features))] + [col for col in df.columns if col.startswith(tuple(f"{f}_ma" for f in features))]
    X = df[keep_cols]
    # Step 3: One-hot encode categorical columns (like 'month')
    return X, idlist

def pca(df):
    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)

    return X_pca, pca