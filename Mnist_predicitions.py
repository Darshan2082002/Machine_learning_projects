import numpy as np
import pandas as pd 
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

plt.imshow(X_train[1], cmap='gray')
plt.title(f"Label: {y_train[0]}")


X_train=X_train/255.0
X_test=X_test/255.0

x_train=X_train.reshape((-1,28,28,1))
X_test=X_test.reshape((-1,28,28,1))
print("Reshaped training data shape:", x_train.shape)
print("Reshaped testing data shape:", X_test.shape)

model=Sequential(
    [
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]
    
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(X_test, y_test), epochs=2)
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Predicted classes:", y_pred_classes)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))  
model.save("digit_classifier.h5")

# Load later
from tensorflow.keras.models import load_model
model = load_model("digit_classifier.h5")

import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("digit_classifier.h5")

def predict(img):
    img = img.convert("L").resize((28,28))
    img = np.array(img).reshape(1,28,28,1) / 255.0
    prediction = np.argmax(model.predict(img))
    return str(prediction)

interface = gr.Interface(fn=predict, inputs="image", outputs="label")
interface.launch()