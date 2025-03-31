from fastapi import FastAPI, Query
from pipeline import load_and_process, value_counts, summary_table, dot_product_corr
from config import FEATURES

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Rainfall Analysis API"}

@app.get("/process/")
async def process(n_lags: int = Query(3, ge=1, le=10)):
    df = load_and_process(n_lags)
    counts = value_counts(df)
    summary = summary_table(df)
    correlation = dot_product_corr(df[FEATURES])
    return {
        "rain_counts": counts,
        "summary_table": summary,
        "correlation_dot_product": correlation
    }
