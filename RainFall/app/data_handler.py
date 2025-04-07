# services/data_handler.py

import pandas as pd
from io import BytesIO
from fastapi import UploadFile

class DataHandler:
    def __init__(self, file: UploadFile):
        self.file = file
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        content = self.file.file.read()
        return pd.read_csv(BytesIO(content))

    def get_dataframe(self) -> pd.DataFrame:
        return self.df