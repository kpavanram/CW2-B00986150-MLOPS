import json
import pandas as pd
import mlflow

def init():
    global model
    model_path = mlflow.pyfunc.get_model_path("model")
    model = mlflow.pyfunc.load_model(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data["input_data"]["data"], columns=data["input_data"]["columns"])
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
