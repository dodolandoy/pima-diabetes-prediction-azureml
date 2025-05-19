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
        return prediction.tolist()
    except Exception as e:
        return str(e)
