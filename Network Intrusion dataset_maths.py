import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


file_name = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(file_name)

df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_').str.replace('.', '_', regex=False)


Y = df['Flow_Duration']
X = df[['Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets']]


df_ols = pd.concat([X, Y], axis=1)
df_ols = df_ols.replace([np.inf, -np.inf], np.nan).dropna()

X = df_ols.drop(columns=['Flow_Duration'])
Y = df_ols['Flow_Duration']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)


ols_model = sm.OLS(Y_train, X_train_sm)
ols_results = ols_model.fit()

residuals = ols_results.resid
fitted_values = ols_results.fittedvalues


plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals, alpha=0.5, s=1) 
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Fitted Values Plot', fontsize=16)
plt.xlabel('Fitted Values (Predicted Flow Duration)', fontsize=14)
plt.ylabel('Residuals (Actual - Fitted)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)


plot_filename = 'ols_residuals_vs_fitted_replot.png'
plt.savefig(plot_filename)


r_squared = ols_results.rsquared
Y_pred = ols_results.predict(X_test_sm)
mse = mean_squared_error(Y_test, Y_pred)

print(f"R-squared (Training Data): {r_squared:.4f}")
print(f"Mean Squared Error (MSE) on Test Data: {mse:.2f}")