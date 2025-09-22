import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv("D:\Python project\Machine_learning_projects\exams.csv")


df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['passed'] = (df['average_score'] >= 60).astype(int)


df.drop(['average_score'], axis=1, inplace=True)


df_encoded = pd.get_dummies(df.drop('passed', axis=1), drop_first=True)


X = df_encoded
y = df['passed']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


y_pred = knn.predict(X_test_scaled)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
plt.bar(["KNN"], [accuracy])
plt.ylim(0, 1)  # Accuracy range between 0 and 1
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.show()