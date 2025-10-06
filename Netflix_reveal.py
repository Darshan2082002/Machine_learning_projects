import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
data=pd.read_csv('netflix_titles.csv', encoding='latin-1')
print(data.info())
print(data.head(5))

df = pd.DataFrame(data)

categorical_cols = ['type', 'country', 'rating', 'listed_in']


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoder_features = encoder.fit_transform(df[categorical_cols])


encoded_df = pd.DataFrame(
    encoder_features,
    columns=encoder.get_feature_names_out(categorical_cols)
)

final_df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)


import sys
sys.stdout.reconfigure(encoding='utf-8')

print(final_df.tail())