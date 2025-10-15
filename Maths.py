

import numpy as np

# Step 1: Define Input Data (Fixed Values)
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([[2], [4], [5], [4], [5]], dtype=float)

# Add a bias term (column of ones) for intercept
X_b = np.c_[np.ones((len(X), 1)), X]

print("Input X values:", X.ravel())
print("Actual y values:", y.ravel())
print("\nMatrix with Bias Term (X_b):\n", X_b)

# Step 2: Assume Initial Parameters (Before Optimization)
theta_initial = np.array([[0.0], [0.0]])  # intercept=0, slope=0
y_pred_before = X_b.dot(theta_initial)

# Step 3: Calculate MSE and R² before optimization
mse_before = np.mean((y - y_pred_before) ** 2)
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res_before = np.sum((y - y_pred_before) ** 2)
r2_before = 1 - (ss_res_before / ss_total)

print("\n--- BEFORE OLS OPTIMIZATION ---")
print("Initial Parameters (theta0, theta1):", theta_initial.ravel())
print(f"Mean Squared Error (MSE): {mse_before:.4f}")
print(f"Coefficient of Determination (R2): {r2_before:.4f}")

# Step 4: Apply OLS Formula
theta_ols = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Step 5: Predictions After Optimization
y_pred_after = X_b.dot(theta_ols)

# Step 6: Compute MSE and R² after optimization
mse_after = np.mean((y - y_pred_after) ** 2)
ss_res_after = np.sum((y - y_pred_after) ** 2)
r2_after = 1 - (ss_res_after / ss_total)

# Step 7: Display Results
print("\n--- AFTER OLS OPTIMIZATION ---")
print("Optimized Parameters (theta0, theta1):", theta_ols.ravel())
print(f"Mean Squared Error (MSE): {mse_after:.4f}")
print(f"Coefficient of Determination (R2): {r2_after:.4f}")

# Step 8: Display Final Equation
print(f"\nBest Fit Line Equation:  y = {theta_ols[0,0]:.2f} + {theta_ols[1,0]:.2f}x")
