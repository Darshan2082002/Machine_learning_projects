import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("diamonds.csv")
print(data.head())
print(data.info())
print(data.describe())

data.drop(columns=['clarity'],inplace=True)
data['cut'].replace({'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5},inplace=True)
data['color'].replace({'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7},inplace=True)


print(data.head())
print(data.isnull().sum())

x_train,x_test,y_train,y_test=train_test_split(data.drop(columns=['price']),data['price'],test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R^2 Score:",r2)
st.title("💎 Diamond Price Prediction App")

st.sidebar.header("Enter Diamond Features")

carat = st.sidebar.number_input("Carat", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
cut = st.sidebar.selectbox("Cut", options={1:"Fair", 2:"Good", 3:"Very Good", 4:"Premium", 5:"Ideal"}.keys(), 
                           format_func=lambda x: {1:"Fair",2:"Good",3:"Very Good",4:"Premium",5:"Ideal"}[x])
color = st.sidebar.selectbox("Color", options={1:"J",2:"I",3:"H",4:"G",5:"F",6:"E",7:"D"}.keys(),
                             format_func=lambda x: {1:"J",2:"I",3:"H",4:"G",5:"F",6:"E",7:"D"}[x])
depth = st.sidebar.number_input("Depth", min_value=40.0, max_value=80.0, value=61.0, step=0.1)
table = st.sidebar.number_input("Table", min_value=40.0, max_value=100.0, value=57.0, step=0.1)
x_val = st.sidebar.number_input("x (length in mm)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
y_val = st.sidebar.number_input("y (width in mm)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
z_val = st.sidebar.number_input("z (depth in mm)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)

# Prediction button
if st.sidebar.button("Predict Price"):
    # Ensure features order matches X.columns
    features = np.array([[carat, cut, color, depth, table, x_val, y_val, z_val]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Price: ${prediction:,.2f}")

# -------------------------------
# Visualizations
# -------------------------------
st.subheader("📊 Data Insights")

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Price distribution
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(data['price'], bins=30, kde=True, ax=ax)
ax.set_title("Price Distribution")
st.pyplot(fig)

# Carat vs Price
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(data['carat'], data['price'], alpha=0.5)
ax.set_xlabel("Carat")
ax.set_ylabel("Price")
ax.set_title("Carat vs Price")
st.pyplot(fig)

# Show what features model expects
st.write("✅ Model trained on features:", list(X.columns))