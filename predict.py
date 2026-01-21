import pickle

from typing import Dict, Any
import uvicorn
from fastapi import FastAPI
import xgboost as xgb

app = FastAPI(tittle="salary-prediction")

with open ('model_xgb.bin', 'rb') as f_in:
       (dv, model) = pickle.load(f_in)

def predict_single(worker: Dict[str, Any]):
    X = dv.transform([worker])
    feature_names = list(dv.get_feature_names_out())
    d = xgb.DMatrix(X, feature_names=feature_names)
    pred = np.expm1(model.predict(d))
    return float(pred[0])

@app.post("/predict")
def predict(worker: Dict[str, Any]):
    salary = predict_single(worker)

    result = {
        'suggested_salary': salary,
    }

    return result


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9696)