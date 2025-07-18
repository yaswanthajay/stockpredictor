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
    # Add more if needed
}

def main():
    st.title("📈 Stock Price Prediction with Prophet")
    st.write("Select a company from the top 100 list to view historical data and forecast.")

    # Create a list for dropdown with ticker and name
    options = [f"{ticker} - {name}" for ticker, name in top_100_companies.items()]
    selection = st.selectbox("Select Company", options)

    # Extract ticker from selection
    ticker = selection.split(" - ")[0]

    st.write(f"Fetching data for **{top_100_companies[ticker]} ({ticker})**...")

    # Fetch stock data
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")[["Close"]].reset_index()

    df = data.rename(columns={"Date": "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()

    # Calculate moving averages
    df['MA7'] = df['y'].rolling(window=7).mean()
    df['MA30'] = df['y'].rolling(window=30).mean()

    # Calculate daily returns
    df['Daily Return'] = df['y'].pct_change()

    # Plot historical price + moving averages
    st.subheader(f"Historical Close Price with Moving Averages - {top_100_companies[ticker]}")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['ds'], df['y'], label='Close Price')
    ax.plot(df['ds'], df['MA7'], label='7-Day MA')
    ax.plot(df['ds'], df['MA30'], label='30-Day MA')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Plot daily returns
    st.subheader(f"Daily Returns (%) - {top_100_companies[ticker]}")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df['ds'], df['Daily Return'])
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Daily Return")
    ax2.grid(True)
    st.pyplot(fig2)

    # Prophet forecasting
    st.subheader("Forecasting with Prophet")

    # 7-day forecast
    st.text("Forecasting next 7 days...")
    model_7 = Prophet(daily_seasonality=True)
    model_7.fit(df[['ds', 'y']])

    future_7 = model_7.make_future_dataframe(periods=7)
    forecast_7 = model_7.predict(future_7)

    fig3 = model_7.plot(forecast_7)
    plt.title(f"7-Day Forecast - {top_100_companies[ticker]}")
    st.pyplot(fig3)

    # 30-day forecast
    st.text("Forecasting next 30 days...")
    model_30 = Prophet(daily_seasonality=True)
    model_30.fit(df[['ds', 'y']])

    future_30 = model_30.make_future_dataframe(periods=30)
    forecast_30 = model_30.predict(future_30)

    fig4 = model_30.plot(forecast_30)
    plt.title(f"30-Day Forecast - {top_100_companies[ticker]}")
    st.pyplot(fig4)

    # Price direction helper
    def direction_check(forecast, days_ahead=1):
        today_price = forecast.loc[forecast.index[-days_ahead-1], 'yhat']
        future_price = forecast.loc[forecast.index[-1], 'yhat']
        return "increase 📈" if future_price > today_price else "decrease 📉"

    st.subheader("Predicted Price Movement Directions:")
    st.write(f"Tomorrow (1 day ahead): Price will likely **{direction_check(forecast_7, 1)}**")
    st.write(f"In 7 days (1 week ahead): Price will likely **{direction_check(forecast_7, 7)}**")
    st.write(f"In 30 days (1 month ahead): Price will likely **{direction_check(forecast_30, 30)}**")

    st.subheader("Next 7 days forecast sample:")
    st.dataframe(forecast_7[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).set_index('ds'))

    st.subheader("Next 30 days forecast sample:")
    st.dataframe(forecast_30[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).set_index('ds'))

if __name__ == "__main__":
    main()
