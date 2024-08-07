import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

# Custom Keras Regressor wrapper
class KerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn=None, **params):
        self.build_fn = build_fn
        if not callable(build_fn):
            raise ValueError("The build_fn must be a callable object.")
        self.set_params(**params)
        self.history = None
        self.model = None

    def fit(self, X, y, **fit_params):
        if self.build_fn is None or not callable(self.build_fn):
            raise ValueError("The build_fn must be a callable object.")
        # Extract build_fn parameters
        build_fn_params = inspect.signature(self.build_fn).parameters
        valid_params = {k: v for k, v in self.get_params().items() if k in build_fn_params}
        self.model = self.build_fn(**valid_params)
        # Add the History callback
        self.history = History()
        self.model.fit(X, y, callbacks=[self.history], **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"build_fn": self.build_fn, **self.__dict__}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Ensure TensorFlow version is up-to-date
print(f'TensorFlow version: {tf.__version__}')

# Load the California Housing dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='MedHouseVal')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Visualize the dataset
sns.pairplot(pd.concat([X, y], axis=1))
plt.show()

# Linear Regression model
linear_reg = LinearRegression()

# Neural Network model
def build_nn_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

nn_model = KerasRegressor(build_fn=build_nn_model, verbose=0)

# Train Linear Regression model
linear_reg.fit(X_train, y_train)

# Train Neural Network model
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Plot training history
plt.plot(nn_model.history.history['loss'], label='train')
plt.plot(nn_model.history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict on the test set with Linear Regression
y_pred_lr = linear_reg.predict(X_test)

# Predict on the test set with Neural Network
y_pred_nn = nn_model.predict(X_test)

# Evaluate Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Linear Regression Mean Squared Error: {mse_lr}')

# Evaluate Neural Network model
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Neural Network Mean Squared Error: {mse_nn}')

# Plot predictions vs actual values
plt.scatter(y_test, y_pred_lr, label='Linear Regression', alpha=0.5)
plt.scatter(y_test, y_pred_nn, label='Neural Network', alpha=0.5)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Tuning Linear Regression model - there's generally not much tuning for Linear Regression
# However, you can cross-validate the model
cv_scores = cross_val_score(linear_reg, X_train, y_train, cv=5)
print(f'Linear Regression Cross-Validated MSE: {np.mean(cv_scores)}')

# Tuning Neural Network model using GridSearchCV
param_grid = {
    'build_fn': [build_nn_model],  # Ensure build_fn is passed
    'optimizer': ['rmsprop', 'adam'],
    'epochs': [50, 100],
    'batch_size': [32, 64]
}

# Recreate the KerasRegressor for GridSearchCV to ensure it works properly
nn_model = KerasRegressor(build_fn=build_nn_model, verbose=0)

grid = GridSearchCV(estimator=nn_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_result = grid.fit(X_train, y_train)

# Summarize results
print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')

# Visualize GridSearchCV results
results = pd.DataFrame(grid_result.cv_results_)
sns.lineplot(x='param_epochs', y='mean_test_score', hue='param_optimizer', style='param_batch_size', data=results)
plt.xlabel('Epochs')
plt.ylabel('Negative Mean Squared Error')
plt.show()
