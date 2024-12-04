import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load historical stock price data
stock_data = pd.read_csv("C:\\Users\\shabana\\OneDrive\\Desktop\\programs\\stock_data.csv")


stock_data.columns = stock_data.columns.str.strip()

# Display column names to check for any discrepancies
print("Column Names:", stock_data.columns)

# Specify features and target variable
features = ['Open', 'High', 'Low', 'Prev. Close', 'Change', '% Change']
target = 'Close'

if not all(feature in stock_data.columns for feature in features):
    missing_features = [feature for feature in features if feature not in stock_data.columns]
    print("Missing features in the DataFrame:", missing_features)
    raise ValueError("Features not found in DataFrame")

# Split data into features (X) and target variable (y)
X = stock_data[features]
y = stock_data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = np.mean((predictions - y_test) ** 2)
print("Mean Squared Error:", mse)

# Plot the predictions vs. actual values
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted Stock Prices")
plt.show()