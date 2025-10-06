import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
data=pd.read_csv('netflix_titles.csv', encoding='latin-1')
print(data.info())
print(data.head(5))

df=pd.DataFrame(data)
encoder=OneHotEncoder(sparse_output=False)
encoder_features = encoder.fit_transform(df[['show_id', 'type', 'title', 'director', 'cast', 'country',
                                             'date_added', 'release_year', 'rating', 'duration',
                                             'listed_in', 'description']])
