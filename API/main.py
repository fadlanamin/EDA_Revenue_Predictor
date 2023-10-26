from fastapi import FastAPI, Path, Query
from fastapi.responses import JSONResponse
import import_ipynb
from Predictor_Function import rev_predictor

app = FastAPI()

@app.get("/")
async def root():
    return {"message": 'Hello! This is the main page for Revenue Predictor for a Film',
            "instruction": 'type /docs at the end of the address bar to proceed',
             "Author": "Ahmad Fadlan Amin"}


@app.get("/predict/{budget},{popularity},{ratings}")
async def get_data(
                #Setting up the user input
                budget:float = Path(description='Budget of the film')
                ,popularity :int = Path(description='High: 4, Moderately High: 3, Medium: 2, Low: 1', gt=0, lt=5)
                ,ratings:int = Path(description='Quality of the film - Scale from 1 - 10', gt=0, lt=11)
                ):
        
        results = rev_predictor(budget,popularity,ratings)

        #The results is in ndarray format, needs to convert to list so that it can convert to json file
        prediction = {'prediction':results.tolist()} 

        return JSONResponse(prediction) # Results needs to be jsonified