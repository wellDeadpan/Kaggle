from fastapi import FastAPI, Query, Form, UploadFile, File
from fastapi.responses import JSONResponse
import io
import joblib
import pandas as pd
from config import FEATURES
from data_utils import read_input_data
from model_utils import load_model, predict_rain
import eda_utils as eda
from config import FEATURES
from data_handler import DataHandler
from eda_handler import EDAHandler
from model_handler import ModelHandler


app = FastAPI()
# Store uploaded data temporarily (per session would be better for multi-user)
DATA_STORE = {}
model_handler = ModelHandler()

@app.get("/")
def root():
    return {"message": "Rainfall Analysis API"}

@app.post("/Read-data")
async def read_data(
    maxtemp: float = Form(None),
    mintemp: float = Form(None),
    humidity: float = Form(None),
    cloud: float = Form(None),
    sunshine: float = Form(None),
    file: UploadFile = File(None)
):
    df = await read_input_data(file, maxtemp, mintemp, humidity, cloud, sunshine)
    DATA_STORE["df"] = df
    return {"message": "Data uploaded and stored successfully", "rows": len(df)}
@app.post("/eda")
async def run_eda():
    df = DATA_STORE.get("df")
    if df is None:
        return {"error": "No data uploaded yet."}

    summary = eda.basic_summary(df)
    outtbl = eda.value_counts(df)
    summary_by_out = eda.summary_table(df)
    #corr = eda.dot_product_corr(df, features=FEATURES)
    htmp = eda.split_heatmap(df)
    return JSONResponse(content={"summary": summary, "correlation_matrix": corr})
@app.post("/predict")
async def run_prediction():
    df = DATA_STORE.get("df")
    if df is None:
        return {"error": "No data uploaded yet."}

    #preprocessed = preprocess_data(df)
    #predictions = predict_rain(model, preprocessed)
    df["will_rain"] = predictions
    return JSONResponse(content=df.to_dict(orient="records"))