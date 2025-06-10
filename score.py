# score.py
import joblib
import numpy as np
import json
from azureml.core.model import Model

# score.py
import joblib
import numpy as np
import json
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("pimadiabetes_logistic_regression_model")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        prediction = model.predict(data)

        results = []
        for pred in prediction:
            if pred == 1:
                result = {
                    "prediction": int(pred),
                    "message": "❗ You are predicted to be at HIGH risk of diabetes.\n",
                    "explanation": "Prediction based on factors such as: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age."
                }
            else:
                result = {
                    "prediction": int(pred),
                    "message": "✅ You are predicted to be at LOW risk of diabetes.\n",
                    "explanation": "Prediction based on factors such as: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age."
                }
            results.append(result)

        return results if len(results) > 1 else results[0]

    except Exception as e:
        return {"error": str(e)}

