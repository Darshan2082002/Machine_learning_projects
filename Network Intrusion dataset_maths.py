import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Load and clean data ---
file_name = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(file_name)

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.replace(' ', '_')
    .str.replace('/', '_')
    .str.replace('-', '_')
    .str.replace('.', '_', regex=False)
)

# Select features and target
Y = df['Flow_Duration']
X = df[['Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets']]

# Remove inf/nan
df_ols = pd.concat([X, Y], axis=1)
df_ols = df_ols.replace([np.inf, -np.inf], np.nan).dropna()

# Split data
X = df_ols.drop(columns=['Flow_Duration'])
Y = df_ols['Flow_Duration']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Add constant for intercept
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit OLS model
ols_model = sm.OLS(Y_train, X_train_sm)
ols_results = ols_model.fit()

# --- Plot residuals ---
residuals = ols_results.resid
fitted_values = ols_results.fittedvalues

plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals, alpha=0.5, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Fitted Values Plot', fontsize=16)
plt.xlabel('Fitted Values (Predicted Flow Duration)', fontsize=14)
plt.ylabel('Residuals (Actual - Fitted)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('ols_residuals_vs_fitted_replot.png')

# --- Model Evaluation ---
r_squared = ols_results.rsquared
Y_pred = ols_results.predict(X_test_sm)
mse = mean_squared_error(Y_test, Y_pred)

print(f"R-squared (Training Data): {r_squared:.4f}")
print(f"Mean Squared Error (MSE) on Test Data: {mse:.2f}\n")

# --- Regression Equation: Y = mX + C ---
params = ols_results.params
intercept = params['const']
print("Regression Equation (Y = mX + C):\n")
equation = f"Flow_Duration = {intercept:.4f}"
for col in X.columns:
    m = params[col]
    equation += f" + ({m:.4f} * {col})"
print(equation)

# --- Print M values (slopes only) ---
print("\nM values (slope coefficients for each predictor):")
m_values = params.drop('const')
for feature, m in m_values.items():
    print(f"{feature}: {m:.6f}")

# --- 📊 Plot 1: Bar Chart of M values ---
plt.figure(figsize=(8, 5))
plt.bar(m_values.index, m_values.values, color='skyblue')
plt.title("M Values (Slope Coefficients) from OLS Model", fontsize=16)
plt.ylabel("Coefficient Value (M)", fontsize=14)
plt.xticks(rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("ols_m_values_bar_plot.png")
plt.show()

# --- 📈 Plot 2: Regression Line for one predictor (example: Total_Fwd_Packets) ---
chosen_feature = 'Total_Fwd_Packets'
m = params[chosen_feature]
c = intercept

x_vals = np.linspace(X[chosen_feature].min(), X[chosen_feature].max(), 100)
y_vals = m * x_vals + c

plt.figure(figsize=(8, 5))
plt.scatter(X[chosen_feature], Y, alpha=0.3, s=10, label='Actual Data')
plt.plot(x_vals, y_vals, color='red', label=f"Y = {m:.4f}X + {c:.4f}")
plt.title(f"Regression Line for {chosen_feature}", fontsize=16)
plt.xlabel(chosen_feature, fontsize=14)
plt.ylabel("Flow_Duration", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("ols_regression_line_single_feature.png")
plt.show()
