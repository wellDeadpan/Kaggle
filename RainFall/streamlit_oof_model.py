# streamlit_model.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import shap
from sklearn.preprocessing import StandardScaler

import sys
import os

# Add the app directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))
from app.model import  (prepare_model_data, prepare_test_data, xgb_oof, plot_fold_auc, xgb_grid_search, lgbm_grid_search,
                        lgbm_oof, elasticnet_grid_search, fit_and_evaluate_model, elasticnet_oof, optimize_ensemble_weights, plot_roc_curves)
from app.features import generate_all_interactions

# --- Streamlit App ---
st.set_page_config(page_title="Rainfall OOF Prediction Model", layout="wide")
st.title("🌧️ Rainfall Prediction with XGBoost")
# Sidebar: n_lags selection
n_lags = 5
# Prepare data
X, y = prepare_model_data("data/train.csv", n_lags)
X_test, idlist = prepare_test_data("data/test.csv", n_lags)
intx_lst = ['humidity_ma3', 'sunshine_ma3']
X = generate_all_interactions(X, intx_lst)

X_test  = generate_all_interactions(X_test, intx_lst)
st.subheader("🚫 Test datasets")
st.dataframe(X_test.head(10))



best_xgb_model, best_xgb_params, best_xgb_score = xgb_grid_search(X, y)
st.subheader("Best XGB Parameters")
st.json(best_xgb_params)
st.markdown(f"**Best Cross-Validated AUROC:** `{best_xgb_score:.4f}`")

best_lgbm_model, best_lgbm_params, best_lgbm_score = lgbm_grid_search(X, y)
st.subheader("Best LGBM Parameters")
st.json(best_lgbm_params)
st.markdown(f"**Best Cross-Validated AUROC:** `{best_lgbm_score:.4f}`")

best_enet_model, best_enet_params, best_score = elasticnet_grid_search(X, y)
st.subheader("Best Elastic Net Parameters")
st.json(best_enet_params)
st.markdown(f"**Best Cross-Validated AUROC:** `{best_score:.4f}`")


st.subheader("Fold-wise AUROC Scores - XGB")
# Train with fixed validation
oof_preds_xgb, oof_proba_xgb, test_preds_xgb, models_xgb, auc_scores_xgb = xgb_oof(X, y, X_test=X_test)
fig = plot_fold_auc(auc_scores_xgb)
st.pyplot(fig)

oof_preds_lgbm, oof_proba_lgbm, test_preds_lgbm, auc_scores_lgbm = lgbm_oof(X, y, X_test=X_test, n_splits=5, random_state=42)
st.subheader("Fold-wise AUROC Scores - LGBM")
fig = plot_fold_auc(auc_scores_lgbm)
st.pyplot(fig)

#oof_preds_elastic, oof_proba_elastic, test_preds_elastic, auc_scores_elastic = elasticnet_oof(
#    X, y, X_test=X_test, best_estimator=best_enet_model, n_splits=5, random_state=42)
#st.subheader("Fold-wise AUROC Scores - Elastic Net")
#fig = plot_fold_auc(auc_scores_elastic)
#st.pyplot(fig)

st.subheader("OOF Correlation")
#st.text(np.corrcoef(oof_preds_xgb, oof_preds_elastic))
st.text(np.corrcoef(oof_preds_xgb, oof_preds_lgbm))


#ensemble_weights, best_auc = optimize_ensemble_weights(
#    y_true=y,
#    preds_list=[oof_preds_xgb, oof_preds_elastic],
#    step=0.01
#)

#st.subheader("Ensemble AUC")
#st.write(f"📈 **ElasticNet AUROC**:{best_auc:.4f}")

#w_xgb, w_enet = ensemble_weights
#final_test_pred = (
#    w_xgb * test_preds_xgb +
#    w_enet * test_preds_elastic
#)

#st.subheader("Optimal Ensemble Weights")
#st.dataframe(pd.DataFrame({
#    "Model": ["XGB", "ElasticNet"],
#    "Weight": ensemble_weights
#}))



#meta_model = xgb.XGBClassifier(eval_metric='logloss')

#meta_model, oof_stacked, test_stacked, stacked_auc = stack_models(
#    oof_preds_list=[oof_preds, oof_preds_elastic],
#    y_true=y_train,
#    test_preds_list=[test_preds, test_preds_elastic],
#    meta_model=meta_model
#)
#st.subheader("Stacking AUROC")
#st.metric("AUROC", f"{stacked_auc:.4f}")

st.subheader("Model ROC Curve Comparison")

#fig = plot_roc_curves(y, oof_preds_xgb, oof_preds_elastic)
#st.pyplot(fig)

st.subheader("Training AUROC")
XGB_auc = roc_auc_score(y, oof_preds_xgb)
#enet_auc = roc_auc_score(y, oof_preds_elastic)

st.write(f"📈 **XGB AUROC**: {XGB_auc:.4f}")
#st.write(f"📈 **ElasticNet AUROC**: {enet_auc:.4f}")


df_out = pd.DataFrame({
    'id': idlist,  # or replace with your actual ID column name
    'rainfall': test_preds_xgb
})

df_out.to_csv("data\submission.csv", index=False)

