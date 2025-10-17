import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

# data or file integration into  the system to train the model 
data=pd.read_csv('NaivesBayes.csv')

data.head() # to read the header of the  data to understand the target and other columns in it 
x="train" # to check the data. if the model work properly 
y=" test" # to predict the model is responding 

(x_train,y_train,x_test,y_test= train_test_split(x,y,random_sixxe=0.2)
