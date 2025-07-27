from flask import Flask, render_template, jsonify
import json
import os
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    tickers = ['GOOGL', 'TSLA', 'AMZN', 'AAPL', 'MSFT', 'META', 'NVDA']  # เปลี่ยนได้ตามที่มีไฟล์ .json
    return render_template('index.html', tickers=tickers)

@app.route('/data/<ticker>')
def get_prediction_data(ticker):
    filename = f"json/{ticker}.json"
    if not os.path.exists(filename):
        return jsonify({"error": "File not found"}), 404

    with open(filename, 'r') as f:
        data = json.load(f)

    df_pred = pd.DataFrame(data)
    df_pred["Date"] = pd.to_datetime(df_pred["Date"])
    future_dates = df_pred["Date"].astype(str).tolist()
    lstm_prices = df_pred["LSTM_Predicted_Price"].tolist()
    trans_prices = df_pred["Transformer_Predicted_Price"].tolist()
    tcn_prices = df_pred["TCN_GRU_Predicted_Price"].tolist()

    last_close = lstm_prices[0]
    lstm_return_pct = ((lstm_prices[-1] - last_close) / last_close) * 100
    trans_return_pct = ((trans_prices[-1] - last_close) / last_close) * 100
    tcn_return_pct = ((tcn_prices[-1] - last_close) / last_close) * 100

    df_actual = pd.read_csv(f"datasets/{ticker}_dataset.csv", index_col=0, parse_dates=True)
    actual_dates = df_actual.index[-60:].astype(str).tolist()
    actual_prices = df_actual["Close"].values[-60:].tolist()
    latest_date = df_actual.index[-1].strftime('%Y-%m-%d')
    return jsonify({
        "ticker": ticker,
        "actual_dates": actual_dates,
        "actual_prices": actual_prices,
        "future_dates": future_dates,
        "lstm_prices": lstm_prices,
        "trans_prices": trans_prices,
        "tcn_prices": tcn_prices,
        "lstm_return_pct": lstm_return_pct,
        "trans_return_pct": trans_return_pct,
        "tcn_return_pct": tcn_return_pct,
        "latest_date": latest_date
    })
