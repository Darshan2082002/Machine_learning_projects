import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("diamonds.csv")
print(data.head())
print(data.info())
print(data.describe())

data.drop(columns=['clarity'],inplace=True)
data['cut'].replace({'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5},inplace=True)
data['color'].replace({'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7},inplace=True)


print(data.head())
print(data.isnull().sum())

x_train,x_test,y_train,y_test=train_test_split(data.drop(columns=['price']),data['price'],test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R^2 Score:",r2)
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
plt.figure(figsize=(10,6))
sns.histplot(data['price'],bins=30,kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(10,6))
plt.scatter(data['carat'],data['price'],alpha=0.5)
plt.title("Carat vs Price")
plt.xlabel("Carat")
plt.ylabel("Price")
plt.show()
plt.figure(figsize=(10,6))