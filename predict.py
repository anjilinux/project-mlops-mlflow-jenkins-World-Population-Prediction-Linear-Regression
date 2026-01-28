import joblib
import numpy as np

model = joblib.load("model/population_model.pkl")

year = np.array([[2030]])
prediction = model.predict(year)

print(f"Predicted population for 2030: {int(prediction[0])}")
