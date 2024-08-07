# Results

The project includes training both a Linear Regression model and a Neural Network model. The performance of these models is compared using Mean Squared Error (MSE). Additionally, hyperparameter tuning for the Neural Network model is performed using GridSearchCV.

## Key Steps:

1. **Data Loading and Preprocessing**:
    - Load the California Housing dataset.
    - Split the data into training and testing sets.
    - Normalize the data using `StandardScaler`.

2. **Data Visualization**:
    - Visualize relationships between features and the target variable using pair plots.

3. **Model Training**:
    - Train a Linear Regression model.
    - Define and train a Neural Network model using a custom `KerasRegressor` wrapper.

4. **Model Evaluation**:
    - Evaluate the models using Mean Squared Error (MSE).
    - Plot predictions vs. actual values.

5. **Hyperparameter Tuning**:
    - Tune hyperparameters of the Neural Network model using `GridSearchCV`.
    - Visualize grid search results.

### Example Outputs:

- **Linear Regression MSE**: The Mean Squared Error of the Linear Regression model on the test set.
- **Neural Network MSE**: The Mean Squared Error of the Neural Network model on the test set.
- **Training History Plots**: Plots showing the loss over epochs for the Neural Network model.
- **Predictions vs. Actual Values Plots**: Scatter plots comparing the actual and predicted values for both models.
- **Hyperparameter Tuning Results**: Visualization of the hyperparameter tuning process and the best hyperparameters found.

These results provide insights into the performance and effectiveness of both models, as well as the impact of different hyperparameters on the Neural Network model.
