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

x_train=x_train.reshape((-1,28,28,1))
x_test=x_test.reshape((-1,28,28,1))

print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.2,random_state=42)

