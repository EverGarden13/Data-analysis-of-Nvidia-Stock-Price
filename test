import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('NVDA.csv')

# Convert 'Date' to datetime format and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature selection: Use previous day's closing price to predict next day's closing price
data['Prev_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)  # Drop rows with NaN values

# Define features and target variable
X = data[['Prev_Close']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Support Vector Regressor
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(X_test)

# Evaluate the models
models = ['Linear Regression', 'Random Forest', 'SVR']
predictions = [lr_predictions, rf_predictions, svr_predictions]

for i, model in enumerate(models):
    mse = mean_squared_error(y_test, predictions[i])
    r2 = r2_score(y_test, predictions[i])
    print(f"{model} - MSE: {mse:.2f}, R²: {r2:.2f}")

# Plotting the results
plt.figure(figsize=(15, 5))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, lr_predictions, label='LR Predictions', color='orange', linestyle='dashed')
plt.plot(y_test.index, rf_predictions, label='RF Predictions', color='green', linestyle='dotted')
plt.plot(y_test.index, svr_predictions, label='SVR Predictions', color='red', linestyle='dashdot')
plt.title('Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
