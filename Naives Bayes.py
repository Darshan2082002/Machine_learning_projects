import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

# data or file integration into  the system to train the model 
data=pd.read_csv('NaivesBayes.csv')

data.head() # to read the header of the  data to understand the target and other columns in it 
