import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('D:\Python project\Machine_learning_projects\Data_set\WA_Fn-UseC_-HR-Employee-Attrition.csv')
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df = df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])

categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtc_metrics = DecisionTreeClassifier(random_state=42)
dtc_metrics.fit(X_train, y_train)


y_pred = dtc_metrics.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

from sklearn.tree import DecisionTreeClassifier


dtc = DecisionTreeClassifier(random_state=42)


dtc.fit(X, y)

print("Decision Tree Classifier trained successfully.")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,20))
plot_tree(dtc, filled=True, feature_names=X.columns.tolist(), class_names=['No Attrition', 'Attrition'], max_depth=3, fontsize=10)
plt.title("Decision Tree Visualization (max_depth=3)")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt


feature_importances = dtc.feature_importances_


features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})


features_df = features_df.sort_values(by='Importance', ascending=False)


print("Top 10 Most Important Features:")
print(features_df.head(10))


plt.figure(figsize=(12, 8))
plt.barh(features_df['Feature'].head(10), features_df['Importance'].head(10), color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances in Decision Tree Classifier')
plt.gca().invert_yaxis()
plt.show()