# services/eda_handler.py

import eda_utils

class EDAHandler:
    def __init__(self, df):
        self.df = df

    def get_summary(self):
        return eda_utils.summarize(self.df)

    def plot_distributions(self):
        return eda_utils.plot_distributions(self.df)  # return filepath or base64
