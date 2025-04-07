import joblib
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations

def load_model(path):
    return joblib.load(path)

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

def run_permutation_test(model, X, y, scoring='neg_log_loss', n_repeats=10, random_state=42, top_n=15):
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state
    )

    importances = pd.Series(result.importances_mean, index=X.columns)
    importances = importances.sort_values(ascending=False)

    # Plot
    importances.head(top_n).plot(kind='barh', figsize=(8, 6), title="Permutation Feature Importance")
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Importance (score drop)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return importances

def run_cv_error_trace(model, X, y, scoring='neg_log_loss', cv=5, model_name='Model'):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    losses = -scores  # negate because higher is better for neg_log_loss

    # Plot
    plt.plot(losses, marker='o', label=model_name)
    plt.title(f"Cross-Validation Error Trace - {model_name}")
    plt.xlabel("Fold")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    print(f"Mean log loss: {losses.mean():.4f} ± {losses.std():.4f}")
    return losses

def predict_rain(model, data):
    return model.predict(data)
