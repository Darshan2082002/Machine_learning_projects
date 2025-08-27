import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import joblib
import streamlit as st



data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
print("The sample data in iris is:", df.sample(100),"\n")
df['species']=data.target
# mapping the data 
df['species']=df['species'].map({0:'setosa',1:'versicolor',2:'virginica'})
df.head()
print("How many rows and columns are in the dataset:", df.shape)
print("Class distribution of the target variable (species):", df['species'].value_counts())
print("Information about the dataset:", df.info())
sns.pairplot(df,hue='species')#diag_kind='kde')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.drop('species',axis=1).corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
X=df.drop('species',axis=1)
y=df['species']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


ml=LogisticRegression()
ml.fit(X_train,y_train)
y_pred=ml.predict(X_test)
print("Classification Report:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Accuracy Score:",accuracy_score(y_test,y_pred)*100)
# Save model
joblib.dump(ml,'iris_model.joblib')

# Load model
model = joblib.load("iris_model.joblib")


st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements to predict species")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Prediction
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.success(f"Predicted Species: {prediction[0]}")


