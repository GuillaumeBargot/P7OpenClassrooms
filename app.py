# 1. Library imports
import uvicorn
from fastapi import FastAPI
import joblib
import re
from lightgbm import LGBMClassifier
import pandas as pd

# 2. Create the app object
app = FastAPI()

model = joblib.load('model.joblib')
clean_data = pd.read_csv('notebooks/clean_data.csv')
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