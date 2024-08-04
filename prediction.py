from fastapi import FastAPI
import numpy as np
from joblib import load
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os

# Defining the input data model
class PredictInput(BaseModel):
    Cost_of_Living_Index: float
    Rent_Index: float
    Groceries_Index: float
    Restaurant_Price_Index: float
    Local_Purchasing_Power_Index: float

app = FastAPI()

multivariate_model = load('multivariate_model.joblib')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Cost of Living Prediction API!"}

@app.post("/predict")
def predict(data: PredictInput):
    # Converting the input data to a numpy array
    input_data = np.array([
        data.Cost_of_Living_Index,
        data.Rent_Index, 
        data.Groceries_Index, 
        data.Restaurant_Price_Index, 
        data.Local_Purchasing_Power_Index
    ]).reshape(1, -1)

    # Making the prediction
    prediction = multivariate_model.predict(input_data)

    return {"prediction": float(prediction[0])}

@app.get("/predict")
def predict_get():
    return {"detail": "Welcome"}, 405

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join("static", "favicon.ico"))

if __name__ == "__main__":
    client = TestClient(app)

    def test_predict():
        # Test POST request
        data = {
            "Cost_of_Living_Index": 80.0,
            "Rent_Index": 75.0,
            "Groceries_Index": 85.0,
            "Restaurant_Price_Index": 90.0,
            "Local_Purchasing_Power_Index": 70.0
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 200
        assert "prediction" in response.json()

        # Test GET request
        response = client.get("/predict")
        assert response.status_code == 405
        assert response.json() == {"detail": "Welcome"}

    test_predict()
