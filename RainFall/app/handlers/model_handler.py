# -----------------------------
# app/handlers/model_handler.py
import joblib
from pathlib import Path
import pandas as pd

class ModelHandler:
    def __init__(self, model_dir='models', active_version='v1.0'):
        self.model_dir = Path(model_dir)
        self.active_version = active_version
        self.model = self.load_model(self.active_version)

    def load_model(self, version):
        path = self.model_dir / version / 'model.pkl'
        return joblib.load(path)

    def predict(self, input_df):
        return self.model.predict_proba(input_df)[:, 1]

    def set_version(self, version):
        self.active_version = version
        self.model = self.load_model(version)