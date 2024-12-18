import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

def preprocess_AAPL_data(data):
    # Apple stock split history
    splits = [
        {"date": "2020-08-31", "factor": 4},
        {"date": "2014-06-09", "factor": 7},
        {"date": "2005-02-28", "factor": 2},
        {"date": "2000-06-21", "factor": 2}
    ]

    adjusted_data = {}
    splits = sorted(splits, key=lambda x: x["date"], reverse=True)

    for date_str, values in data.items():
        date = datetime.strptime(date_str, "%Y-%m-%d")
        adjustment_factor = 1.0

        for split in splits:
            if date < datetime.strptime(split["date"], "%Y-%m-%d"):
                adjustment_factor *= split["factor"]

        adjusted_data[date_str] = round(float(values["4. close"]) / adjustment_factor, 4)

    return adjusted_data

def plot_closing_prices_by_year(data):
    # Convert data to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = df["Close"].astype(float)
    df.set_index("Date", inplace=True)

    # Plot closing prices
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Close"], label='Closing Price')

    # Format the X-axis to show only the year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Major ticks every year

    # Add labels and title
    plt.title("AAPL Closing Prices Over 25 Years")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.legend()

    # Rotate X-axis labels for clarity
    plt.xticks(rotation=45)
    plt.show()

def plot_moving_average(data, window=50):
    df = pd.DataFrame(list(data.items()), columns=["Date", "Close"])
    df["Close"] = df["Close"].astype(float)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    df["Rolling Average"] = df["Close"].rolling(window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label='Closing Price')
    plt.plot(df.index, df["Rolling Average"], label=f'{window}-Day Moving Average')
    plt.title("AAPL Closing Price with Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_daily_returns_with_year(data):
    # Convert data to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = df["Close"].astype(float)

    # Calculate daily returns
    df["Daily Return"] = df["Close"].pct_change()

    # Plot daily returns
    plt.figure(figsize=(10, 5))
    plt.scatter(df["Date"], df["Daily Return"], alpha=0.5, s=5, label='Daily Returns')

    # Format the X-axis to show only the year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Major ticks every year

    # Add labels and title
    plt.title("AAPL Daily Returns Distribution")
    plt.xlabel("Year")
    plt.ylabel("Daily Return")
    plt.legend()

    # Rotate X-axis labels for clarity
    plt.xticks(rotation=45)
    plt.show()

with open("AAPL.json", "r") as file:
    raw_APPL_data = json.load(file)

preprocessed_data = preprocess_AAPL_data(raw_APPL_data["Time Series (Daily)"])

# Convert the preprocessed AAPL closing prices into a NumPy array
prices = np.array(list(preprocessed_data.values())).reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 30  # Lookback window of 30 days
X, y = create_sequences(scaled_prices, n_steps)

# Reshape for LSTM input
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define Encoder
encoder_inputs = Input(shape=(n_steps, 1))
encoder = LSTM(50, activation='relu', return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Define Decoder
decoder_inputs = Input(shape=(n_steps, 1))
decoder_lstm = LSTM(50, activation='relu', return_sequences=False)
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Output layer
outputs = Dense(1)(decoder_outputs)

# Build and compile the model
model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer='adam', loss='mse')

model.fit([X_train, X_train], y_train,
          epochs=50, batch_size=32, validation_data=([X_test, X_test], y_test))

predictions = model.predict([X_test, X_test])
predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label="Actual Prices")
plt.plot(predicted_prices, label="Predicted Prices")
plt.title("AAPL Actual vs Predicted Closing Prices")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()


model.summary()
