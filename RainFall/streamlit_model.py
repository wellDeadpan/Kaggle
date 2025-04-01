# streamlit_model.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve, confusion_matrix, classification_report
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler

import sys
import os

# Add the app directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))
from app.model import  (prepare_model_data, prepare_test_data, pca_error, pca_loadings, pca, get_top_features_by_error_type,
                        compare_loading_differences_by_group,run_mda_pruning, xgb_oof, plot_fold_auc, xgb_grid_search, lgbm_grid_search,
                        lgbm_oof, elasticnet_grid_search, fit_and_evaluate_model, elasticnet_oof, optimize_ensemble_weights, plot_roc_curves)
from app.features import compute_mda_importance, plot_mda_importance
from app.features import generate_all_interactions

# --- Streamlit App ---
st.set_page_config(page_title="Rainfall Prediction Model", layout="wide")
st.title("🌧️ Rainfall Prediction with XGBoost")

# Sidebar: n_lags selection
n_lags = 5

# Prepare data
X, y = prepare_model_data("data/train.csv", n_lags)
#X_test = prepare_test_data("data/test.csv", n_lags)

intx_lst = ['humidity_ma3', 'cloud_ma3', 'sunshine_ma3', 'windspeed_ma3']
X = generate_all_interactions(X, intx_lst)
#X_test  = generate_all_interactions(X_test, intx_lst)


# Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

#cols_to_drop = ['humidity_lag1', 'humidity_lag2', 'humidity_lag3', 'humidity_lag4', 'pressure_lag2', 'dewpoint_lag4', 'windspeed_lag5']
#X.drop(columns=cols_to_drop, inplace=True)

#prefixes = ('mintemp_')
#X.drop(columns=[col for col in X.columns if col.startswith(prefixes)], inplace=True)

best_xgb_model, best_xgb_params, best_xgb_score = xgb_grid_search(X_train, y_train)
st.subheader("Best XGB Parameters")
st.json(best_xgb_params)
st.markdown(f"**Best Cross-Validated AUROC:** `{best_xgb_score:.4f}`")

best_lgbm_model, best_lgbm_params, best_lgbm_score = lgbm_grid_search(X_train, y_train)
st.subheader("Best LGBM Parameters")
st.json(best_lgbm_params)
st.markdown(f"**Best Cross-Validated AUROC:** `{best_lgbm_score:.4f}`")

best_enet_model, best_enet_params, best_score = elasticnet_grid_search(X_train, y_train)
st.subheader("Best Elastic Net Parameters")
st.json(best_enet_params)
st.markdown(f"**Best Cross-Validated AUROC:** `{best_score:.4f}`")



model_type = 'xgb'
xgb_model, xgb_auc, xgb_preds, xgb_curve = fit_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, best_xgb_params)

model_type = 'lgbm'
lgbm_model, lgbm_auc, lgbm_preds, lgbm_curve = fit_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, best_lgbm_params)

model_type = 'enet'
enet_model, enet_auc, enet_preds, enet_curve = fit_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, None, best_estimator=best_enet_model)

st.subheader("ROC Curve - XGB")
st.pyplot(xgb_curve)

st.subheader("ROC Curve - LGBM")
st.pyplot(lgbm_curve)

st.subheader("ROC Curve - Elastic Net")
st.pyplot(enet_curve)

st.subheader("Predicted Probability Distribution by Actual Label")
# Prepare results
xgb_res = pd.DataFrame({
    'True Label': y_val.reset_index(drop=True),
    'Predicted Probability': xgb_preds
})

xgb_predclass = xgb_model.predict(X_val)
lgbm_predclass = lgbm_model.predict(X_val)
enet_predclass = enet_model.predict(X_val)

# Convert labels to human-readable form (optional)
xgb_res['True Label'] = xgb_res['True Label'].map({0: 'No Rain', 1: 'Rain'})

# Plot
fig_violin, ax_violin = plt.subplots(figsize=(8, 5))
sns.violinplot(data=xgb_res, x='True Label', y='Predicted Probability', inner=None, palette='pastel', cut=0, ax=ax_violin)
sns.stripplot(data=xgb_res, x='True Label', y='Predicted Probability', jitter=0.2, size=4, color='black', alpha=0.6, ax=ax_violin)

ax_violin.set_title("Distribution of Predicted Rain Probability")
ax_violin.set_ylabel("Predicted Probability (Rain)")
ax_violin.set_xlabel("Actual Label")

st.pyplot(fig_violin)


st.header("🔍 Variable Importance (Mean Decrease Accuracy)")
importance_df = compute_mda_importance(xgb_model, X_val, y_val, X_val.columns)
fig = plot_mda_importance(importance_df)
st.pyplot(fig)

st.header("🔍 Variable Importance (SHAP Values)")
# Create SHAP explainer
explainer = shap.Explainer(xgb_model, X_val)
# Compute SHAP values
shap_values = explainer(X_val, check_additivity=False)
# Visualize
fig = shap.summary_plot(shap_values, X_val)
st.pyplot(fig)

explainer = shap.TreeExplainer(xgb_model)
shap_interaction_values = explainer.shap_interaction_values(X)
with st.expander("🔍 SHAP Feature Interactions"):
    feat1 = st.selectbox("Main Feature", X.columns)
    feat2 = st.selectbox("Interaction Feature", X.columns)
    if st.button("Plot SHAP Interaction"):
        fig, ax = plt.subplots()
        shap.dependence_plot((feat1, feat2), shap_interaction_values, X, show=False)
        st.pyplot(fig)

# Convert X_val back to DataFrame if needed
X_val_df = X_val.copy()
X_val_df["actual"] = y_val
X_val_df["predicted"] = xgb_predclass
X_val_df["proba"] = xgb_preds

st.dataframe(X_val_df.sort_values("proba", ascending=False).head(10))

false_negatives = X_val_df[(X_val_df["actual"] == 1) & (X_val_df["predicted"] == 0)]
false_positives = X_val_df[(X_val_df["actual"] == 0) & (X_val_df["predicted"] == 1)]

false_negatives.sort_values("proba", ascending=False).head(10)
false_positives.sort_values("proba", ascending=True).head(10)

st.subheader("🚫 False Negatives (Missed Rain)")
st.dataframe(false_negatives.sort_values("proba", ascending=False).head(10))

st.subheader("🚫 False Positives (Wrong Rain Predictions)")
st.dataframe(false_positives.sort_values("proba", ascending=True).head(10))

# Error analysis with PCA
X_pca, pca_obj = pca(X_val)
fig, res = pca_error(X_pca, y_val, xgb_predclass, xgb_preds, X_val, pca_obj)

st.subheader("🚫 Error analysis with PCA")
st.dataframe(res.sort_values("Proba", ascending=True).head(10))

st.subheader("PCA Visualization of Prediction Errors")
st.pyplot(fig)

loadings = pca_loadings(pca_obj, X_val.columns)
st.subheader("Top Features Driving PCA1")
st.dataframe(loadings.head(10).style.format("{:.2f}"))

top_features_by_error = get_top_features_by_error_type(res, feature_cols=X_val.columns, top_n=10)

scaled = StandardScaler().fit_transform(X_val)
diff_tables = compare_loading_differences_by_group(pca_obj, scaled, res, X_val.columns)

for label, df in diff_tables.items():
    st.subheader(f"🔍 Feature Loading Differences: {label}")
    st.dataframe(df.style.format("{:.3f}"))

history_df = run_mda_pruning(X_train, y_train, importance_df, model_func=None, min_features=5, cv=5)

st.subheader("AUROC vs. Number of Features")
fig, ax = plt.subplots()
sns.lineplot(data=history_df, x='num_features', y='AUROC_mean', marker='o', ax=ax)
ax.set_xlabel("Number of Features")
ax.set_ylabel("AUROC")
ax.set_title("Feature Pruning Effect on AUROC")
st.pyplot(fig)
