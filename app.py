# 1. Library imports
from array import array
from itertools import product
import uvicorn
from fastapi import FastAPI
import joblib
from lightgbm import LGBMClassifier
import pandas as pd


# 2. Create the app object
app = FastAPI()

def get_clean_data(online):
    clean_datas = []
    url1 = 'https://media.githubusercontent.com/media/GuillaumeBargot/P7OpenClassrooms/main/notebooks/clean_data'
    url2 = '.csv'
    for i in range(1,4):
        file = ""
        if(online):
            file = url1 + str(i) + url2
        else:
            file = "notebooks/clean_data" + str(i) + ".csv"
        clean_datas.append(pd.read_csv(file))
    return pd.concat(clean_datas, ignore_index=True)

def get_zip_data():
    url = 'notebooks/clean_data1.zip'
    clean_data = pd.read_csv(url,compression='zip')
    return clean_data

model = joblib.load('model.joblib')
clean_data = get_zip_data()
X = clean_data.drop('TARGET', axis=1)
del clean_data

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



