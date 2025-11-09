import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_name = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(file_name)


df.columns = (
    df.columns.str.strip()
    .str.replace(' ', '_')
    .str.replace('/', '_')
    .str.replace('-', '_')
    .str.replace('.', '_', regex=False)
)

Y = df['Flow_Duration']
X = df[['Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets']]


df_ols = pd.concat([X, Y], axis=1)
df_ols = df_ols.replace([np.inf, -np.inf], np.nan).dropna()


X = df_ols.drop(columns=['Flow_Duration'])
Y = df_ols['Flow_Duration']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(Y_train, X_train_sm)
ols_results = ols_model.fit()


chosen_feature = 'Total_Fwd_Packets'
m = ols_results.params[chosen_feature]   
c = ols_results.params['const']          


x_vals = np.linspace(X[chosen_feature].min(), X[chosen_feature].max(), 100)
y_vals = m * x_vals + c

# --- Plot slope-intercept diagram ---
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, color='orange', linewidth=2, label=f'y = {m:.4f}x + {c:.4f}')

# Draw axes
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# Label slope and intercept visually
plt.text(x_vals[10], y_vals[10] + (y_vals.max()-y_vals.min())*0.05, 'y = mx + c', color='orange', fontsize=12)
plt.text(x_vals.min(), c + (y_vals.max()-y_vals.min())*0.03, f'c = {c:.2f}', color='blue', fontsize=12)
plt.text(x_vals.mean(), y_vals.mean(), f'm = {m:.6f}', color='blue', fontsize=12)

# Formatting
plt.xlabel(chosen_feature)
plt.ylabel('Flow_Duration')
plt.title(f"Slope-Intercept Form for {chosen_feature}\n(y = mx + c)", fontsize=14)
plt.legend()
plt.grid(False)
plt.show()
