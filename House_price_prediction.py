import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib

import streamlit as st


data = pd.read_csv(r"data.csv")

print("The first five rows of the dataset are:\n",data.info(),"\n")
print("The first five rows of the dataset are:\n",data.sample(5),"\n")

print("The shape of the dataset is:",data.shape)

print("The columns in the dataset are:",data.info())

print("Missing data in each column:\n",data.isnull().sum())
data.drop(["city","statezip","country","street","date"],axis=1,inplace=True)
print(data.describe())

data["sqft_basement"].fillna(data["sqft_basement"].median(), inplace=True)
print(data.info())


X=data.drop(columns=["price"])
y=data["price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train) 
model1=Lasso()
model1.fit(X_train,y_train) 
model2=Ridge()
model2.fit(X_train,y_train) 
#changes Need to be done for hyperparameter tuning and feature engineering

y_pred = model.predict(X_test)
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
print("The predicted values are:", mean_squared_error(y_test, y_pred) *100)
print("The R^2 Score for Linear Regression:", r2_score(y_test, y_pred)*100)
print("The Mean Absolute Error for Laso:", mean_absolute_error(y_test, y_pred1))
print("The R^2 Score for Laso:", r2_score(y_test, y_pred1)*100)
print("The Mean Absolute Error for Ridge:", mean_absolute_error(y_test, y_pred2))
print("The R^2 Score for Ridge", r2_score(y_test, y_pred2)*100)


