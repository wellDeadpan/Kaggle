# -----------------------------
# app/handlers/eda_handler.py
from ..utils import eda_utils as eda  # Relative import based on your structure

class EDAHandler:
    def __init__(self):
        pass  # No internal state yet, so we can leave this empty

    def summarize(self, df):
        return eda.basic_summary(df)

    def outcomecount(self, df, outcome_col='rainfall'):
        return eda.value_counts(df, outcome_col)

    def splitheatmap(self, df, features, outcome_col='rainfall'):
        return eda.split_heatmap(df, features, outcome_col)

    def plot_boxplots(self, df, feature_name, outcome_col='rainfall'):
        return eda.plot_feature_boxplots(df, feature_name, outcome_col)

    def corrmat(self, df):
        return eda.plot_correlation_matrix(df)

    def pca_plots(self, X, y, n_components=2):
        return eda.pca_plots(X, y, n_components)




