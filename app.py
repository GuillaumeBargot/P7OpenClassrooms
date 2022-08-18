# 1. Library imports
from array import array
from itertools import product
import uvicorn
from fastapi import FastAPI
import joblib
import re
from lightgbm import LGBMClassifier
import pandas as pd
import logging

# 2. Create the app object
app = FastAPI()

def get_clean_data():
    clean_datas = []
    for i in range(1,10):
        clean_datas.append(pd.read_csv('notebooks/clean_data' + str(i)))
        logging.warning("PDCONCAT" + clean_datas[len(clean_datas)-1].columns)
    return pd.concat(clean_datas, ignore_index=True)

model = joblib.load('model.joblib')
clean_data = get_clean_data()
logging.warning(clean_data.columns)
X = clean_data.drop('TARGET', axis=1)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, stranger'}

@app.get('/predict/{sk_id}')
async def predict(sk_id: int):
    prob = actualy_predict(sk_id)
    return {'message': str(prob)}

def actualy_predict(sk_id: int):
    row = X.loc[X.SK_ID_CURR == sk_id]
    prob = model.predict_proba(row)
    return prob[0]

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)



