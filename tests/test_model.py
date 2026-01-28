import joblib
import numpy as np
import os

def test_model_prediction():
    assert os.path.exists("model/population_model.pkl")

    model = joblib.load("model/population_model.pkl")
    prediction = model.predict(np.array([[2025]]))

    assert prediction[0] > 0
