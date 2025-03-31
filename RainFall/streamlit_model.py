# streamlit_model.py

import streamlit as st
import pandas as pd
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
from app.model import  prepare_model_data, pca_error, pca_loadings, pca, get_top_features_by_error_type, compare_loading_differences_by_group,run_mda_pruning, xgb_oof, plot_fold_auc
from app.features import compute_mda_importance, plot_mda_importance
from app.features import generate_all_interactions


# --- Streamlit App ---
st.set_page_config(page_title="Rainfall Prediction Model", layout="wide")
st.title("🌧️ Rainfall Prediction with XGBoost")

# Sidebar: n_lags selection
n_lags = 5

# Prepare data
X, y = prepare_model_data(n_lags)

intx_lst = ['humidity_ma3', 'cloud_ma3', 'sunshine_ma3', 'windspeed_ma3']
X = generate_all_interactions(X, intx_lst)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


# Train Model
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}
sample_weights = y_train.map({0: 1, 1: 1})
model = xgb.XGBClassifier(eval_metric='logloss',random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Define scorer
scorer = make_scorer(roc_auc_score, needs_proba=True)

# GridSearchCV wrapper
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Run the search
grid_search.fit(X_train, y_train, sample_weight=sample_weights)
st.subheader("📈 Best AUROC Score:")
st.text(grid_search.best_score_)
st.subheader("✅ Best Parameters:")
st.text(grid_search.best_params_)

# Predict probabilities and calculate AUROC
y_proba = model.predict_proba(X_test)[:, 1]
# Predict label
y_pred = model.predict(X_test)
auc_score = roc_auc_score(y_test, y_proba)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
ax.plot([0, 1], [0, 1], '--', color='gray')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
ax.grid(True)

# Show results
st.subheader("AUROC Score")
st.metric(label="AUROC", value=f"{auc_score:.3f}")

st.subheader("ROC Curve")
st.pyplot(fig)

st.write("### Test Set Confusion Matrix")
st.text(confusion_matrix(y_test, y_pred))

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))

st.subheader("Fold-wise AUROC Scores")
# Train with fixed validation
oof_preds, test_preds, models, auc_scores = xgb_oof(X, y, X_test=None)
fig = plot_fold_auc(auc_scores)
st.pyplot(fig)

st.subheader("Predicted Probability Distribution by Actual Label")
# Prepare results
results = pd.DataFrame({
    'True Label': y_test.reset_index(drop=True),
    'Predicted Probability': y_proba
})

# Convert labels to human-readable form (optional)
results['True Label'] = results['True Label'].map({0: 'No Rain', 1: 'Rain'})

# Plot
fig_violin, ax_violin = plt.subplots(figsize=(8, 5))
sns.violinplot(data=results, x='True Label', y='Predicted Probability', inner=None, palette='pastel', cut=0, ax=ax_violin)
sns.stripplot(data=results, x='True Label', y='Predicted Probability', jitter=0.2, size=4, color='black', alpha=0.6, ax=ax_violin)

ax_violin.set_title("Distribution of Predicted Rain Probability")
ax_violin.set_ylabel("Predicted Probability (Rain)")
ax_violin.set_xlabel("Actual Label")

st.pyplot(fig_violin)



# After you split/train your model:
# model = RandomForestClassifier().fit(X_train, y_train)
st.header("🔍 Variable Importance (Mean Decrease Accuracy)")
importance_df = compute_mda_importance(model, X_test, y_test, X_test.columns)
fig = plot_mda_importance(importance_df)
st.pyplot(fig)

st.header("🔍 Variable Importance (SHAP Values)")
# Create SHAP explainer
explainer = shap.Explainer(model, X_test)
# Compute SHAP values
shap_values = explainer(X_test, check_additivity=False)
# Visualize
fig = shap.summary_plot(shap_values, X_test)
st.pyplot(fig)

# Convert X_val back to DataFrame if needed
X_val_df = X_test.copy()
X_val_df["actual"] = y_test
X_val_df["predicted"] = y_pred
X_val_df["proba"] = y_proba

false_negatives = X_val_df[(X_val_df["actual"] == 1) & (X_val_df["predicted"] == 0)]
false_positives = X_val_df[(X_val_df["actual"] == 0) & (X_val_df["predicted"] == 1)]

false_negatives.sort_values("proba", ascending=False).head(10)
false_positives.sort_values("proba", ascending=True).head(10)

st.subheader("🚫 False Negatives (Missed Rain)")
st.dataframe(false_negatives.sort_values("proba", ascending=False).head(10))

st.subheader("🚫 False Positives (Wrong Rain Predictions)")
st.dataframe(false_positives.sort_values("proba", ascending=True).head(10))

# Error analysis with PCA
X_pca, pca_obj = pca(X_test)
fig, res = pca_error(X_pca, y_test, y_pred, y_proba, X_test, pca_obj)

st.subheader("🚫 Error analysis with PCA")
st.dataframe(res.sort_values("Proba", ascending=True).head(10))

st.subheader("PCA Visualization of Prediction Errors")
st.pyplot(fig)

loadings = pca_loadings(pca_obj, X_test.columns)
st.subheader("Top Features Driving PCA1")
st.dataframe(loadings.head(10).style.format("{:.2f}"))

top_features_by_error = get_top_features_by_error_type(res, feature_cols=X_test.columns, top_n=10)

scaled = StandardScaler().fit_transform(X_test)
diff_tables = compare_loading_differences_by_group(pca_obj, scaled, res, X_test.columns)

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
