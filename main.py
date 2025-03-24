# #####    using closing prices


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()





# Allow CORS (Enable API access from any frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = load_model("lstmfinalmodel.h5")

# Define API input and output models using Pydantic
class PredictionRequest(BaseModel):
    tickers: List[str]  # List of stock tickers
    days: int           # Number of future days to predict

class StockPredictionResponse(BaseModel):
    stock: str                        # Stock ticker
    closing_prices: Dict[str, float]   # Historical closing prices {date: price}
    test_predictions: List[float]      # Test predictions
    test_dates: List[str]      # Test predictions
    future_predictions: List[float]    # Future predictions
    future_dates: List[str]            # Dates for future predictions

class PredictionResponse(BaseModel):
    results: List[StockPredictionResponse]  # List of predictions for each stock

# Helper function to fetch and preprocess stock data
def fetch_and_prepare_stock_data(stock: str, start_date: str, end_date: str, seq_length: int = 100):
    try:
        # Fetch stock data
        data = yf.download(stock, start_date, end_date)

        if data.empty:
            raise ValueError(f"Failed to fetch data for {stock}. The ticker might be invalid or no data is available.")

        # Extract closing prices with date mapping
        closing_prices = data['Close'].to_dict()

        # Scale the 'Close' prices
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # Split into training and testing datasets
        training_data_len = int(len(scaled_data) * 0.8)
        test_data = scaled_data[training_data_len - seq_length:]
        x_test = create_dataset(test_data, seq_length)

        # Prepare the last 100 days for future predictions
        if len(scaled_data) >= seq_length:
            last_100_days = scaled_data[-seq_length:]
        else:
            padding = np.zeros((seq_length - len(scaled_data), 1))
            last_100_days = np.vstack((padding, scaled_data))

        return data, scaler, x_test, last_100_days, closing_prices
    except Exception as e:
        raise ValueError(f"Error processing stock {stock}: {str(e)}")

# Helper function to create datasets
def create_dataset(data, seq_length=100):
    x = []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])
    return np.array(x)

# Define API endpoints
@app.get("/")
def root():
    return {"message": "Stock Price Prediction API is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_prices(request: PredictionRequest):
    start_date = '2021-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    seq_length = 100

    results = []

    for stock in request.tickers:
        try:
            # Fetch and prepare stock data
            data, scaler, x_test, last_100_days, closing_prices = fetch_and_prepare_stock_data(stock, start_date, end_date, seq_length)

            # Predict test data
            test_predictions = model.predict(x_test)
            test_predictions = scaler.inverse_transform(test_predictions).flatten().tolist()
            last_date = pd.to_datetime(data.index[-1])
            data.reset_index(inplace=True)
            data['Date'] = data['Date'].dt.strftime('%Y/%m/%d')
            test_dates=data['Date'][-len(test_predictions):]
            # Predict future data
            future_predictions = []
            for _ in range(request.days):
                X_future = last_100_days[-seq_length:].reshape(1, seq_length, 1)
                next_pred = model.predict(X_future)[0, 0]
                future_predictions.append(next_pred)
                last_100_days = np.append(last_100_days, [[next_pred]], axis=0)

            # Inverse transform future predictions
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()
            print(type(data.index[-1]), data.index[-1])
            # Generate future dates
            future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, request.days + 1)]
            print(closing_prices.items())
            # Add result to response
            results.append(
                StockPredictionResponse(
                    stock=stock,
                    closing_prices = {pd.to_datetime(date).strftime('%Y-%m-%d'): price for date, price in closing_prices[stock].items()},
                    test_predictions=test_predictions,
                    test_dates=test_dates,
                    future_predictions=future_predictions,
                    future_dates=future_dates
                )
            )
            print('test_dates',closing_prices,type(closing_prices))
        except Exception as e:
            results.append(
                StockPredictionResponse(
                    stock=stock,
                    closing_prices={},
                    test_predictions=[],
                    test_dates=[],
                    future_predictions=[],
                    future_dates=[],
                )
            )
            print(f"Error with stock {stock}: {e}")

    return PredictionResponse(results=results)
