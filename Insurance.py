import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("insurance.csv")
print(data.columns)
data['sex']=data['sex'].map({'male': 1, 'female': 2})
data['smoker']=data['smoker'].map({'yes':1, 'no':2})
data['region']=data['region'].map({'southwest':1, 'southeast':2,'northwest':3, 'northeast':4})
print(data.head)
print(data.info)
X=data.drop(columns=['charges'])
y=data['charges']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Classification Report:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Accuracy Score:",accuracy_score(y_test,y_pred)*100)