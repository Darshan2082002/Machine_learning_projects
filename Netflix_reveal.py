import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
data=pd.read_csv('netflix_titles.csv', encoding='latin-1')
print(data.info())
print(data.head(5))

encoder=OneHotEncoder(sparse_output=False)