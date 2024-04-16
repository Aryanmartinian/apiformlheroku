import uvicorn
from fastapi import FastAPI
from meta import var_data
from fastapi.middleware.cors import CORSMiddleware
import pickle
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pickle_in = open("outlier.pkl","rb")
classifier = pickle.load(pickle_in)


@app.get('/')
def index():
    return {'Deployment':'Hello and Welcome to the outliers API'}


@app.post('/predict')
def predict(data:var_data):
    data = data.dict()
    X1=data['X1']
    X2=data['X2']

    prediction = classifier.predict([[X1,X2]])
    if(prediction[0]==1):
        prediction = "Not a Outlier"
    elif(prediction[0]==-1):
        prediction="Outlier"
    return {
        'prediction': prediction
    }

if __name__ =='__main__':
    uvicorn.run(app,host='127.0.0.1',port=5000)