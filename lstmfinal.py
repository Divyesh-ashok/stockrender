import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from sklearn.metrics import mean_squared_error, r2_score
import math
import yfinance as yf
from datetime import datetime


# Load dataset
start='2021-01-01'
end=datetime.today().strftime('%Y-%m-%d')
stock='TATAPOWER.NS'
data=yf.download(stock,start,end)
print(data.head())


# Normalize the 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'])

# Split into training and testing data
training_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len:]

def create_dataset(data, seq_length=100):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

x_train, y_train = create_dataset(train_data)
x_test, y_test = create_dataset(test_data)

data.shape[0]

y_test.shape[0]

# Define the model
model = Sequential()

# Add Input layer explicitly
model.add(Input(shape=(x_train.shape[1], 1)))

# Add LSTM layers and dropout for regularization
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))

# Add Dense layers
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Make predictions
predictions = model.predict(x_test)

# Inverse transform predictions and actual values to get the stock prices
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE and R-squared
rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
print(f'RMSE: {rmse}')
r2 = r2_score(y_test_actual, predictions)
print(f'R-squared: {r2}')

# Predict the next 100 days
future_predictions = []
last_100_days = scaled_data[-100:]  # Get the last 100 days for prediction

for _ in range(100):
    X_future = last_100_days[-100:].reshape(1, 100, 1)  # Shape it for LSTM
    next_pred = model.predict(X_future)[0, 0]
    future_predictions.append(next_pred)
    last_100_days = np.append(last_100_days, [[next_pred]], axis=0)

# Inverse transform future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Convert the 'Date' column to datetime (ensure the column exists in your dataset)
data=data.reset_index()
data['Date'] = pd.to_datetime(data['Date'])

# Generate dates for the future predictions
last_date = data['Date'].iloc[-1]  # Get the last date from the dataset
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1,101)]  # Add 50 days

# Plot the data with proper dates
plt.figure(figsize=(16, 8))

# Plot actual prices
plt.plot(data['Date'], data['Close'], label='Actual Prices')
close_price=data['Close']
# Plot predicted prices (from test data)
prediction_dates = data['Date'][training_data_len + 100:]  # Match prediction length
plt.plot(prediction_dates, predictions, label='Predicted Prices')

# Plot future predictions
plt.plot(future_dates, future_predictions, label='Next 100 Days Predictions', color='green')

# Add labels and legend
plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()


model.save("lstmfinalmodel.h5")
