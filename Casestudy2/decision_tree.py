import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# Re-define X and y for robust execution in case previous cells were not run
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df = df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])

categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate and train the Decision Tree Classifier on the training data
dtc_metrics = DecisionTreeClassifier(random_state=42)
dtc_metrics.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dtc_metrics.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

from sklearn.tree import DecisionTreeClassifier

# Instantiate the Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42)

# Fit the model to the training data
dtc.fit(X, y)

print("Decision Tree Classifier trained successfully.")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,20))
plot_tree(dtc, filled=True, feature_names=X.columns.tolist(), class_names=['No Attrition', 'Attrition'], max_depth=3, fontsize=10)
plt.title("Decision Tree Visualization (max_depth=3)")
plt.show()