from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_price
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    features: dict


@app.get("/")
def read_root():
    return {'status': 'running'}


@app.post("/predict")
def predict(input_data: InputData) -> dict:
    results = predict_price(input_data.features)
    return results
