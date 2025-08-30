import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import streamlit as slt

df = pd.read_csv("WineQT.csv")
print("Data info:")
print(df.describe())
print(df.head())

df.drop(['Id'], axis=1, inplace=True)
print(df.head())
print("Missing values in each column:\n", df.isnull().sum())
x=df.drop(['quality'], axis=1)
y=df['quality']


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("\n Predictions:",y_pred)

print("\n Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("\n R2 Score:", r2_score(y_test, y_pred))
