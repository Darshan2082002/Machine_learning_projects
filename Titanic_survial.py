import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import joblib
import streamlit as st 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 

data=pd.read_csv('Titanic-Dataset.csv')
print("The Titanic dataset contains the following information:",data.info(),"\n")
print(" The Titanic dataset contains the following information:",data.head(),"\n")
print("Missing Data",data.isnull().sum(),"\n")

sns.countplot(x='Survived',data=data)
plt.title("Survival Count 0= Not Survived, 1= Survived")
plt.show()

sns.countplot(x='Pclass',hue='Survived',data=data)
plt.title("Survival Count by Passenger Class")
plt.show()

sns.countplot(x='Sex',hue='Survived',data=data)
plt.title("Survival Count by Gender")
plt.show()

#Data cleaning
data.drop(['Ticket','Cabin','Name'],axis=1,inplace=True)

data['Age']= data['Age'].fillna(data['Age'].median())
data['Embarked']=data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare']=data['Fare'].fillna(data['Fare'].median())
print("Missing Data",data.isnull().sum(),"\n")
data['Sex']=LabelEncoder().fit_transform(data['Sex'])
data=pd.get_dummies(data,columns=['Embarked'],drop_first=True)
X=data.drop('Survived',axis=1)
y=data['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=50,test_size=0.3)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))

joblib.dump(model,'titanic_model.joblib')


model=joblib.load('titanic_model.joblib')
st.title("Titanic Survival Prediction")
st.write("Enter the details of the passenger to predict survival")
pclass=st.selectbox("Passenger Class",[1,2,3])
sex=st.selectbox("Sex",['male','female'])
age=st.number_input("Age",min_value=0.0,max_value=100.0,value=25.0)
Embarked=st.selectbox("Port of Embarkation",['C','Q','S'])

sex=1 if sex=='male' else 0
Embarked_C = 1 if Embarked == "C" else 0
Embarked_Q = 1 if Embarked == "Q" else 0
Embarked_S = 1 if Embarked == "S" else 0

features = np.array([[0, pclass, sex, age, 0, 0, 0, Embarked_Q, Embarked_S]])

if st.button("Predict Survival"):
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] * 100
    
    if prediction[0] == 1:
        st.success(f" Passenger is **likely to survive** with probability {probability:.2f}%")
    else:
        st.error(f" Passenger is **not likely to survive** with probability {100 - probability:.2f}%")