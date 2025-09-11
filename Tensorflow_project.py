import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets

(x_train,y_train),(x_test,y_test)=datasets.fashion_mnist.load_data()

plt.imshow(x_train[0],cmap='gray')
plt.title(y_train[0])
plt.show()

x_train=x_train/255.0
x_test=x_test/255.0
