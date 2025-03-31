import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pipeline import load_and_process
from config import FEATURES



def prepare_model_data(n_lags=5):
    # Step 1: Load processed data with lag features and 'month'
    df = load_and_process(n_lags=n_lags)
    # Step 2: Drop target columns from X
    df = df.copy()
    # Ensure 'month' is retained for one-hot encoding
    keep_cols = ['Month'] + [col for col in df.columns if col.startswith(tuple(f"{f}_lag" for f in FEATURES))] + [col for col in df.columns if col.startswith(tuple(f"{f}_ma" for f in FEATURES))]
    X = df[keep_cols]
    # Step 3: One-hot encode categorical columns (like 'month')
    categorical_cols = ['Month']
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    y = df['rain_flag']
    return X_encoded, y

X, y = prepare_model_data(n_lags=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

sample_weights = y_train.map({0: 1, 1: 2})
model = xgb.XGBClassifier(eval_metric='logloss', max_depth=3, colsample_bytree=0.9, subsample=0.9)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]
y_true = y_test.reset_index(drop=True)

# Create sorted DataFrame
results = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
results_sorted = results.sort_values(by='y_proba', ascending=False).reset_index(drop=True)

# Plot
plt.figure(figsize=(12, 3))
plt.plot(results_sorted['y_true'].values, marker='o', linestyle='None')
plt.title("Rainy (1) vs Non-Rainy (0) Days Sorted by Predicted Probability")
plt.xlabel("Sorted Days")
plt.ylabel("True Rainfall Label")
plt.show()

# Compute AUROC
auc_score = roc_auc_score(y_test, y_proba)
print(f"AUROC Score: {auc_score:.3f}")

fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def pca(X_test):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)

    # Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca


def add_pca_loadings(ax, pca, feature_names, scale=10, top_n=None, color='gray', alpha=0.6):
    """
    Draw loading vectors (PCA components) on the PCA scatterplot (biplot).
    :param ax: matplotlib axis
    :param pca: fitted PCA object
    :param feature_names: list of feature names
    :param scale: arrow length scaling factor
    :param top_n: only show top N loadings by magnitude
    :param color: arrow color
    :param alpha: arrow transparency
    """
    components = pca.components_
    loadings = pd.DataFrame(components.T, index=feature_names, columns=['PCA1', 'PCA2'])
    loadings['magnitude'] = np.linalg.norm(loadings[['PCA1', 'PCA2']], axis=1)

    if top_n:
        loadings = loadings.sort_values(by='magnitude', ascending=False).head(top_n)

    for feature, row in loadings.iterrows():
        x, y = row['PCA1'] * scale, row['PCA2'] * scale
        ax.arrow(0, 0, x, y, color=color, alpha=alpha, head_width=0.3, length_includes_head=True)
        ax.text(x * 1.1, y * 1.1, feature, fontsize=4, color=color, ha='center', va='center')


def pca_error(X_pca, y_test, y_pred, y_proba, X_test, pca):
    results = pd.DataFrame(X_test.copy())
    results["Actual"] = y_test
    results["Predicted"] = y_pred
    results["Proba"] = y_proba
    results["PCA1"] = X_pca[:, 0]
    results["PCA2"] = X_pca[:, 1]

    # Label error types
    def get_error_type(row):
        if row["Actual"] == 1 and row["Predicted"] == 0:
            return "False Negative"
        elif row["Actual"] == 0 and row["Predicted"] == 1:
            return "False Positive"
        elif row["Actual"] == 1 and row["Predicted"] == 1:
            return "True Positive"
        else:
            return "True Negative"

    results["Error Type"] = results.apply(get_error_type, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=results,
        x='PCA1', y='PCA2',
        hue='Error Type',
        style='Error Type',
        palette={
            'True Positive': 'green',
            'True Negative': 'blue',
            'False Positive': 'orange',
            'False Negative': 'red'
        },
        ax=ax
    )
    add_pca_loadings(ax, pca, X_test.columns, scale=30)
    ax.set_title("PCA Projection of Model Errors")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return fig, results

def pca_loadings(pca, feature_names):
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=['PCA1', 'PCA2']
    ).sort_values(by='PCA1', key=abs, ascending=False)
    return loadings


def get_top_features_by_error_type(results_df, feature_cols, top_n=10, method="mean_diff"):
    """
    Identify top contributing features for each error type based on their deviation from the rest.
    :param results_df: DataFrame with error labels, feature values, and proba
    :param feature_cols: List of columns used for PCA (original features)
    :param top_n: Number of top features to return
    :param method: 'mean_diff' or 'zscore' or 'absolute'
    :return: Dictionary of DataFrames for each error type
    """
    summary = {}
    error_types = results_df["Error Type"].unique()

    for etype in error_types:
        group = results_df[results_df["Error Type"] == etype]
        rest = results_df[results_df["Error Type"] != etype]

        if method == "mean_diff":
            mean_diff = group[feature_cols].mean() - rest[feature_cols].mean()
            top_features = mean_diff.abs().sort_values(ascending=False).head(top_n)
        elif method == "zscore":
            std = rest[feature_cols].std()
            z_diff = (group[feature_cols].mean() - rest[feature_cols].mean()) / std
            top_features = z_diff.abs().sort_values(ascending=False).head(top_n)
        elif method == "absolute":
            top_features = group[feature_cols].mean().abs().sort_values(ascending=False).head(top_n)

        summary[etype] = top_features.to_frame(name="Score")

    return summary


def compare_loading_differences_by_group(pca, X_scaled, results_df, feature_names):
    """
    Compare PCA loading-weighted means of groups (e.g., FN vs TP, FP vs TN)
    Returns a dictionary of DataFrames for analysis.
    """
    # PCA loadings: shape (n_features, 2)
    loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=['PCA1', 'PCA2'])

    # PCA projection (scores)
    X_pca = pca.transform(X_scaled)
    pca_scores = pd.DataFrame(X_pca, columns=['PCA1_val', 'PCA2_val'])

    # Merge with error types
    df = pd.concat([results_df.reset_index(drop=True), pca_scores], axis=1)

    groups = {
        "FN vs TP": ("False Negative", "True Positive", "desc"),
        "FP vs TN": ("False Positive", "True Negative", "desc"),
        "FN vs TN": ("False Negative", "True Negative", "asc"),
        "FP vs TP": ("False Positive", "True Positive", "asc"),
    }

    results = {}
    for label, (g1, g2, sort_order) in groups.items():
        g1_mean = df[df["Error Type"] == g1][["PCA1_val", "PCA2_val"]].mean()
        g2_mean = df[df["Error Type"] == g2][["PCA1_val", "PCA2_val"]].mean()

        # Delta in PCA space
        delta = g1_mean - g2_mean  # shape (2,)
        # Project delta back to feature space using loadings
        feature_diff = loadings.values @ delta.values.reshape(-1, 1)
        feature_diff = pd.Series(feature_diff.flatten(), index=feature_names, name="Loading Difference")

        # Order based on use case
        feature_diff_df = feature_diff.abs().sort_values(
            ascending=(sort_order == "asc")
        ).to_frame()

        results[label] = feature_diff_df

    return results


def prune_and_evaluate_mda(X_val, y_val, mda_df, model_func, min_features=5):
    """
    Iteratively remove features with lowest MDA (including negative), evaluate AUROC at each step.

    :param X_val: Validation feature set
    :param y_val: Validation labels
    :param mda_df: DataFrame with 'feature' and 'importance_mean'
    :param model_func: Callable that fits a model: (X, y) → model
    :param min_features: Stop pruning when only this many features remain
    :return: history DataFrame with AUROC and feature list at each step
    """
    history = []
    features = mda_df.sort_values(by="importance_mean", ascending=True)['feature'].tolist()
    remaining_features = features.copy()

    for i in range(len(features)):
        if len(remaining_features) < min_features:
            break

        # Train on current feature set
        model = model_func(X_val[remaining_features], y_val)
        y_pred_proba = model.predict_proba(X_val[remaining_features])[:, 1]
        auroc = roc_auc_score(y_val, y_pred_proba)

        history.append({
            'num_features': len(remaining_features),
            'AUROC': auroc,
            'features': remaining_features.copy()
        })

        # Remove the least important remaining feature
        remaining_features.pop(0)

    return pd.DataFrame(history)


def prune_and_evaluate_mda_cv(X, y, mda_df, model_func, min_features=5, cv=5):
    """
    Iteratively remove low-MDA features and evaluate mean AUROC using CV.

    :param X: Feature DataFrame
    :param y: Target vector
    :param mda_df: DataFrame with MDA importance values
    :param model_func: Callable returning an unfitted model
    :param min_features: Minimum number of features to retain
    :param cv: Number of CV folds
    :return: DataFrame of pruning history
    """
    history = []
    features = mda_df.sort_values(by="importance_mean", ascending=True)['feature'].tolist()
    remaining_features = features.copy()

    for i in range(len(features)):
        if len(remaining_features) < min_features:
            break

        model = model_func()
        scores = cross_val_score(
            model,
            X[remaining_features],
            y,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1
        )

        history.append({
            'num_features': len(remaining_features),
            'AUROC_mean': scores.mean(),
            'AUROC_std': scores.std(),
            'features': remaining_features.copy()
        })

        # Remove the least important remaining feature
        remaining_features.pop(0)

    return pd.DataFrame(history)


def run_mda_pruning(X, y, mda_df, model_func=None, min_features=5, cv=5):
    """
    Wrapper for MDA pruning using cross-validated AUROC.
    """
    import xgboost as xgb

    if model_func is None:
        model_func = lambda: xgb.XGBClassifier(eval_metric="logloss", max_depth=4, colsample_bytree=0.8, subsample=0.9)

    history_df = prune_and_evaluate_mda_cv(X, y, mda_df, model_func=model_func, min_features=min_features, cv=cv)
    return history_df


def xgb_oof(X, y, X_test=None, n_splits=5, best_params=None, random_state=42):
    """
    Generate OOF predictions for XGBClassifier.

    Returns:
        oof_preds: array of OOF predictions for training data
        test_preds: mean prediction for test data (if provided)
        models: list of trained models per fold
    """
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []
    auc_scores = []

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"🔁 Fold {fold + 1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state + fold,
            max_depth=4, colsample_bytree=0.8, subsample=0.9
        )
        model.fit(X_train, y_train)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, oof_preds[val_idx])
        auc_scores.append(auc)
        print(f"✅ Fold {fold + 1} AUC: {auc:.4f}")

        if X_test is not None:
            test_preds += model.predict_proba(X_test)[:, 1] / n_splits

        models.append(model)

    print(f"\n📊 Overall OOF AUC: {roc_auc_score(y, oof_preds):.4f}")
    return oof_preds, test_preds, models, auc_scores


def plot_fold_auc(auc_scores):
    folds = list(range(1, len(auc_scores) + 1))
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=folds, y=auc_scores, palette="coolwarm", ax=ax)
    ax.axhline(mean_auc, color='black', linestyle='--', label=f"Mean AUC = {mean_auc:.3f}")
    ax.fill_between(folds,
                    [mean_auc - std_auc] * len(folds),
                    [mean_auc + std_auc] * len(folds),
                    alpha=0.2, color='gray', label=f"±1 std = {std_auc:.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUROC")
    ax.set_title("Fold-wise AUROC Scores")
    ax.legend()
    fig.tight_layout()
    return fig