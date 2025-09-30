import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("insurance.csv")


data=data['sex'].map({'male': 1, 'female': 2})
data=data['smoker'].map({'yes':0,'no':1})
print(data.head)
print(data.info)