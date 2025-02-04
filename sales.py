import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Load the data
data = pd.read_csv('sales_data.csv', parse_dates=['date'], index_col='date')

# Display the first few rows of the data
print(data.head())

# Plot the sales data
plt.figure(figsize=(12, 6))
plt.plot(data, label='Sales')
plt.title('Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Display the sizes of the training and test sets
print(f'Training set size: {len(train)}')
print(f'Test set size: {len(test)}')

# Fit the model on the training data
model = ExponentialSmoothing(train['sales'], trend='add', seasonal='add', seasonal_periods=12)
fit_model = model.fit()

# Make predictions
predictions = fit_model.forecast(steps=len(test))

# Plot the training data, test data, and predictions
plt.figure(figsize=(12, 6))
plt.plot(train['sales'], label='Train')
plt.plot(test['sales'], label='Test')
plt.plot(predictions, label='Predictions', color='red')
plt.title('Sales Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Calculate the Mean Squared Error
mse = mean_squared_error(test['sales'], predictions)
print(f'Mean Squared Error: {mse}')

# Calculate the Root Mean Squared Error
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Save the model
joblib.dump(fit_model, 'sales_forecasting_model.pkl')

# Load the model
loaded_model = joblib.load('sales_forecasting_model.pkl')

# Make future predictions
future_predictions = loaded_model.forecast(steps=12)

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(data['sales'], label='Historical Sales')
plt.plot(future_predictions, label='Future Predictions', color='red')
plt.title('Future Sales Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
