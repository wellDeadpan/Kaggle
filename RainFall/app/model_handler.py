# services/model_handler.py

import joblib
import pandas as pd
import model_utils

class ModelHandler:
    def __init__(self, model_path="assets/elastic_net_model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, df: pd.DataFrame):
        processed = model_utils.preprocess(df)
        preds = self.model.predict_proba(processed)[:, 1]  # Probability of rain
        return preds
