from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the model and scalers
model = pickle.load(open('app/model.pkl', 'rb'))
minmax_scaler = pickle.load(open('app/minmaxscaler.pkl', 'rb'))
standard_scaler = pickle.load(open('app/standscaler.pkl', 'rb'))

# Define the input data model
class SoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Define the crop dictionary
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya', 7: 'orange',
    8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes', 12: 'mango', 13: 'banana',
    14: 'pomegranate', 15: 'lentil', 16: 'blackgram', 17: 'mungbean', 18: 'mothbeans',
    19: 'pigeonpeas', 20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

@app.post("/predict")
async def predict_crop(data: SoilData):
    try:
        # Convert input data to numpy array
        features = np.array([[
            data.N, data.P, data.K, data.temperature, 
            data.humidity, data.ph, data.rainfall
        ]])
        
        # Apply MinMax scaling
        features_minmax = minmax_scaler.transform(features)
        
        # Apply Standard scaling
        features_scaled = standard_scaler.transform(features_minmax)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Get the crop name
        crop_name = crop_dict[prediction[0]]
        
        return {"predicted_crop": crop_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Crop Recommendation API"}
