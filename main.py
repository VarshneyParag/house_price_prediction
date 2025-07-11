import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 🔍 Debug: Check directory and dataset contents
print("Current files:", os.listdir())
print("Inside folder:", os.listdir('house_prices.csv'))

# 📥 Load dataset
df = pd.read_csv('house_prices.csv/train.csv')

# ✅ Select features as per task description
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# 🔧 Drop rows with missing values (if any)
df = df[features + [target]].dropna()

# 🎯 Define input and output
X = df[features]
y = df[target]

# ✂️ Split dataset for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 📈 Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 🤖 Predict on test set
y_pred = model.predict(X_test)

# 📊 Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# 🧠 Enhancement: Plot Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # ideal line
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
