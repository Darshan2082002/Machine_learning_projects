import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("insurance.csv")
print(data.columns)
data['sex']=data['sex'].map({'male': 1, 'female': 2})
data['smoker']=data['smoker'].map({'yes':1, 'no':2})
data['region']=data['region'].map({'southwest':1, 'southeast':2,'northwest':3, 'northeast':4})
print(data.head)
print(data.info)
