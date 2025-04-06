# Model Parameters Summary

## Traditional Machine Learning Models (ML.ipynb)

### Ridge Regression
- **Algorithm**: RidgeCV with cross-validation
- **Parameters**:
  - alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
  - cv = 5
- **Feature selection**: Top 10 features by correlation with target

### Support Vector Regression (SVR)
- **Algorithm**: SVR with GridSearchCV hyperparameter tuning
- **Parameters**:
  - kernel = 'rbf'
  - C = [0.1, 1, 10] (searched)
  - gamma = ['scale', 'auto', 0.1, 0.01] (searched)
  - epsilon = [0.01, 0.1, 0.2] (searched)
- **Cross-validation**: 5-fold
- **Scoring metric**: Negative mean squared error
- **Feature selection**: Same top 10 features as Ridge

### XGBoost
- **Parameters**:
  - n_estimators = 100
  - learning_rate = 0.01
  - max_depth = 3
  - min_child_weight = 3
  - subsample = 0.8
  - colsample_bytree = 0.8
  - gamma = 1
  - reg_alpha = 0.1
  - reg_lambda = 1.0
  - random_state = 42
- **Feature selection**: Same top 10 features as other models

## Deep Learning Model (LSTM.ipynb)

### Attention LSTM
- **Network architecture**:
  - input_size = number of features (excluding calendar features)
  - hidden_size = 128
  - num_layers = 2
  - dropout = 0.2
- **Training parameters**:
  - sequence_length = 30
  - batch_size = 32
  - learning_rate = 0.001
  - optimizer = Adam
  - early stopping patience = 10
  - lr_scheduler = ReduceLROnPlateau (patience=5, factor=0.5)

## Time Series Model (TS.ipynb)

### ARIMA
- **Model selection**: auto_arima with parameters:
  - seasonal = False
  - test = 'adf' (Augmented Dickey-Fuller test)
  - stepwise = True
  - Optimal order (p,d,q) determined automatically during model training