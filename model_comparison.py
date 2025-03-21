import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
df = pd.read_csv('NVDA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Feature engineering
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(window=20).std()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df['RSI'] = df['Returns'].rolling(window=14).apply(
    lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean())))
)

# Drop NaN values
df.dropna(inplace=True)

# Prepare features and target
features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Volatility', 'MA50', 'MA200', 'RSI']
target = 'Close'

X = df[features]
y = df[target]

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# Function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Dictionary to store results
results = {}

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
results['Linear Regression'] = evaluate_model(y_test, lr_pred, 'Linear Regression')

# 2. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.ravel())
rf_pred = rf_model.predict(X_test)
results['Random Forest'] = evaluate_model(y_test, rf_pred, 'Random Forest')

# 3. XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train.ravel())
xgb_pred = xgb_model.predict(X_test)
results['XGBoost'] = evaluate_model(y_test, xgb_pred, 'XGBoost')

# Create figure for plots
plt.figure(figsize=(15, 12))

# Plotting predictions
plt.subplot(2, 1, 1)
test_dates = df.index[-len(y_test):]
plt.plot(test_dates, scaler_y.inverse_transform(y_test), label='Actual', color='blue')
plt.plot(test_dates, scaler_y.inverse_transform(lr_pred), label='Linear Regression', color='red', alpha=0.7)
plt.plot(test_dates, scaler_y.inverse_transform(rf_pred), label='Random Forest', color='green', alpha=0.7)
plt.plot(test_dates, scaler_y.inverse_transform(xgb_pred), label='XGBoost', color='orange', alpha=0.7)
plt.title('NVIDIA Stock Price Predictions - All Models')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)

# Plotting model comparison
plt.subplot(2, 1, 2)
metrics = pd.DataFrame(results).T
metrics.plot(kind='bar', ax=plt.gca())
plt.title('Model Performance Comparison')
plt.xlabel('Models')
plt.ylabel('Metric Values')
plt.legend(title='Metrics')
plt.xticks(rotation=45)

# Adjust layout and save plot
plt.tight_layout()
plt.savefig('plots/model_comparison.png')
plt.close()

# Print feature importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
})
print("\nRandom Forest Feature Importance:")
print(feature_importance.sort_values('Importance', ascending=False))

# Print XGBoost feature importance
xgb_feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb_model.feature_importances_
})
print("\nXGBoost Feature Importance:")
print(xgb_feature_importance.sort_values('Importance', ascending=False)) 