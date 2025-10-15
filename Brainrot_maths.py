import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. Load Data
data = pd.read_csv('Brain Rot Cases (1).csv')
print(data.head())
print(data.describe())

# 2. Clean column names and remove duplicates
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
data = data.drop_duplicates()

# 3. Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['number']).columns.tolist()

# 4. Fill missing values
fill_map = {}
for col in data.columns:
    if col in categorical_cols:
        if data[col].mode().shape[0] > 0:
            fill_map[col] = data[col].mode()[0]
        else:
            fill_map[col] = ""
    else:
        fill_map[col] = data[col].mean()
data = data.fillna(fill_map)

# 5. Encode categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

print("Cleaned & Encoded Data (first 10 rows):", data_encoded.shape)
print(data_encoded.head(10))

# 6. Select Target Column
candidates = [c for c in data_encoded.columns if ("mental" in c or "focus" in c or "concentrate" in c or "mood" in c or "mental_health" in c)]
if len(candidates) == 0:
    candidates = data_encoded.select_dtypes(include=[np.number]).columns.tolist()
    target_col = candidates[-1]
else:
    target_col = candidates[0]

y = data_encoded[target_col].astype(float)
X = data_encoded.drop(columns=[target_col])

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Closed-form OLS solution
X_train_aug = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
X_test_aug = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

XtX = X_train_aug.T.dot(X_train_aug)
lam = 1e-8
XtX_reg = XtX + lam * np.eye(XtX.shape[0])
beta_closed = np.linalg.inv(XtX_reg).dot(X_train_aug.T).dot(y_train.values)

# 10. Predictions (Closed-form)
y_pred_closed = X_test_aug.dot(beta_closed)

# 11. Metrics for Closed-form
mse_closed = mean_squared_error(y_test, y_pred_closed)
r2_closed = r2_score(y_test, y_pred_closed)

# 12. Sklearn LinearRegression for comparison
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

results = {
    "closed_form_mse": float(mse_closed),
    "closed_form_r2": float(r2_closed),
    "sklearn_mse": float(mse_lr),
    "sklearn_r2": float(r2_lr)
}

results_text = "\n".join([f"{k}: {v}" for k, v in results.items()])
print("Results:\n", results_text)

# --- Add plot comparing predictions vs actual for both models ---
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred_closed, label="Closed-form OLS Prediction", linestyle='--', marker='x')
plt.plot(y_pred_lr, label="Sklearn LinearRegression Prediction", linestyle=':', marker='s')
plt.title("Actual vs Predicted Values")
plt.xlabel("Test Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.tight_layout()
plt.show()
