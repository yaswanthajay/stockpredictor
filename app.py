import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# ---------- Top 100 Companies ----------
top_100_companies = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "META": "Meta (Facebook)",
    "BRK-B": "Berkshire Hathaway",
    "UNH": "UnitedHealth",
    "JNJ": "Johnson & Johnson",
    "V": "Visa",
    "XOM": "ExxonMobil",
    "JPM": "JPMorgan Chase",
    "WMT": "Walmart",
    "MA": "Mastercard",
    "PG": "Procter & Gamble",
    "LLY": "Eli Lilly",
    "HD": "Home Depot",
    "CVX": "Chevron",
    "KO": "Coca-Cola",
    # (Add more if needed)
}

print("\n📊 Top 100 Companies:")
for ticker, name in top_100_companies.items():
    print(f"{ticker:<6} - {name}")

input_name = input("\nEnter the company ticker or name (e.g., AAPL or Apple): ").strip().lower()

ticker = None
for tk, name in top_100_companies.items():
    if input_name == tk.lower() or input_name in name.lower():
        ticker = tk
        break

if ticker is None:
    raise ValueError("❌ Company not in top 100 list. Please enter a valid ticker or name.")

print(f"\n📥 Fetching stock data for {top_100_companies[ticker]} ({ticker})...")
stock = yf.Ticker(ticker)
data = stock.history(period="max")[["Close"]].reset_index()

df = data.rename(columns={"Date": "ds", "Close": "y"})
df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna()

# --- Calculate Moving Averages ---
df['MA7'] = df['y'].rolling(window=7).mean()
df['MA30'] = df['y'].rolling(window=30).mean()

# --- Calculate Daily Returns ---
df['Daily Return'] = df['y'].pct_change()

# --- Plot Historical Price + MA7 + MA30 ---
plt.figure(figsize=(14,6))
plt.plot(df['ds'], df['y'], label='Close Price')
plt.plot(df['ds'], df['MA7'], label='7-Day MA')
plt.plot(df['ds'], df['MA30'], label='30-Day MA')
plt.title(f"Historical Close Price with Moving Averages - {top_100_companies[ticker]}")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Daily Returns ---
plt.figure(figsize=(14,4))
plt.plot(df['ds'], df['Daily Return'])
plt.title(f"Daily Returns (%) - {top_100_companies[ticker]}")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Prophet Forecast for next 7 days (daily) ---
print("📡 Forecasting next 7 days...")
model_7 = Prophet(daily_seasonality=True)
model_7.fit(df[['ds', 'y']])

future_7 = model_7.make_future_dataframe(periods=7)
forecast_7 = model_7.predict(future_7)

model_7.plot(forecast_7)
plt.title(f"7-Day Forecast - {top_100_companies[ticker]}")
plt.xlabel("Date")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Prophet Forecast for next 30 days (monthly) ---
print("📡 Forecasting next 30 days...")
model_30 = Prophet(daily_seasonality=True)
model_30.fit(df[['ds', 'y']])

future_30 = model_30.make_future_dataframe(periods=30)
forecast_30 = model_30.predict(future_30)

model_30.plot(forecast_30)
plt.title(f"30-Day Forecast - {top_100_companies[ticker]}")
plt.xlabel("Date")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Simple directional predictions ---
def direction_check(forecast, days_ahead=1):
    today_price = forecast.loc[forecast.index[-days_ahead-1], 'yhat']
    future_price = forecast.loc[forecast.index[-1], 'yhat']
    return "increase" if future_price > today_price else "decrease"

print("\n🔮 Predicted Price Movement Directions:")
print(f"Tomorrow (1 day ahead): Price will likely {direction_check(forecast_7, 1)}")
print(f"In 7 days (1 week ahead): Price will likely {direction_check(forecast_7, 7)}")
print(f"In 30 days (1 month ahead): Price will likely {direction_check(forecast_30, 30)}")

# --- Print forecasted values for 7 and 30 days ---
print("\n📅 Next 7 days forecast (sample):")
print(forecast_7[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))

print("\n📅 Next 30 days forecast (sample):")
print(forecast_30[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
