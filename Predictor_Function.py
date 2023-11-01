#Importing Library Dataframe
import import_ipynb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import import_ipynb
from Data_Cleaning import df

#Data that will be used for linear regression model is from 1990
df_linreg = df[['release_year', 'budget_adj', 'revenue_adj', 'popularity_code']].sort_values('release_year')
df_linreg = df_linreg[df_linreg['release_year'] > 1989]


#Setting up and Model Training
x_reg = df_linreg[['budget_adj', 'popularity_code']]
y_reg = df_linreg[['revenue_adj']]
xtrain, xtest, ytrain, ytest = train_test_split(x_reg, y_reg, test_size=0.15,  random_state=90)

model = LinearRegression()
model.fit(xtrain, ytrain)

#Function for the prediction
def rev_predictor(budget,pop) : 
    features = np.array([[budget,pop]])
    rev_pred = model.predict(features)
    return rev_pred