from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
from handlers.data_handler import DataHandler
from handlers.eda_handler import EDAHandler
from handlers.model_handler import ModelHandler
import utils.eda_utils as eda_utils  # If needed directly
from io import StringIO

app = FastAPI()

data_handler = DataHandler()
eda_handler = EDAHandler()
model_handler = ModelHandler()

# Temporary in-memory store (can be replaced by session manager or state handler)
DATA_STORE = {}


# ------------------------------
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    df = data_handler.process_uploaded_file(file)
    DATA_STORE["df"] = df
    return {"rows": len(df)}


# ------------------------------
@app.post("/eda")
async def run_eda():
    df = DATA_STORE.get("df")
    if df is None:
        return {"error": "No data uploaded yet."}

    summary = eda_handler.summarize(df).to_dict()
    # You can add other functions like:
    # value_counts = eda.value_counts(df)
    # summary_by_out = eda.summary_table(df)
    # heatmap_fig = eda.split_heatmap(df)  # If returns image/fig

    return JSONResponse(content={
        "summary": summary
        # Add other outputs here
    })


# ------------------------------
@app.post("/predict")
async def run_prediction():
    df = DATA_STORE.get("df")
    if df is None:
        return {"error": "No data uploaded yet."}

    predictions = model_handler.predict(df)
    df = df.copy()
    df["rain_proba"] = predictions
    df["will_rain"] = (df["rain_proba"] > 0.5).astype(int)

    return JSONResponse(content=df.to_dict(orient="records"))
