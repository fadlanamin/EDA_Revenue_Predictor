#Importing Library Dataframe
import import_ipynb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import import_ipynb
from Data_Cleaning import df


#Setting up data and model
x_reg2 = df[['budget_adj', 'popularity_code','vote_average']]
y_reg2 = df[['revenue_adj']]
xtrain, xtest, ytrain, ytest = train_test_split(x_reg2, y_reg2, test_size=0.15, random_state=89)

model = LinearRegression()
model.fit(xtrain, ytrain)

#Function for the prediction
def rev_predictor(budget,pop,vote_average) : 
    features = np.array([[budget,pop,vote_average]])
    rev_pred = model.predict(features)
    return rev_pred