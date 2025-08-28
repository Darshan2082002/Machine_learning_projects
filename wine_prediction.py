import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
import streamlit as slt

df = pd.read_csv("WineQT.csv")
print("Data info:")
print(df.describe(8*13))
print(df.head())