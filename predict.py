import os, joblib, requests, pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression

FEATURES = ['open_price', 'high_price', 'low_price', 'volume']
MODEL_FILE = 'models.pkl'
OUT_FILE = 'predictions.csv'

def train():
    import kagglehub
    path = kagglehub.dataset_download("fgh09101010/cypto-data")
    df = pd.read_csv(os.path.join(path, '1_coinhistory.csv'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    m_now = LinearRegression().fit(df[FEATURES], df['close_price'])
    df['t'] = df['close_price'].shift(-60)
    d = df.dropna(subset=['t'])
    m_1h = LinearRegression().fit(d[FEATURES], d['t'])
    joblib.dump((m_now, m_1h), MODEL_FILE)

def fetch():
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": "BTCUSDT", "interval": "1m", "limit": 2}, timeout=15)
        k = r.json()[0]
        return float(k[1]), float(k[2]), float(k[3]), float(k[5])
    except Exception as e:
        print("Binance fail, CryptoCompare use ho raha hai:", e)
        r = requests.get("https://min-api.cryptocompare.com/data/v2/histominute",
                         params={"fsym": "BTC", "tsym": "USD", "limit": 2}, timeout=15)
        c = r.json()["Data"]["Data"][0]
        return c["open"], c["high"], c["low"], c["volumefrom"]

if not os.path.exists(MODEL_FILE):
    train()
m_now, m_1h = joblib.load(MODEL_FILE)

o, h, l, v = fetch()
x = pd.DataFrame([[o, h, l, v]], columns=FEATURES)
row = {
    "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    "open": o, "high": h, "low": l, "volume": v,
    "pred_now": round(float(m_now.predict(x)[0]), 2),
    "pred_1hour": round(float(m_1h.predict(x)[0]), 2),
}
print(row)
pd.DataFrame([row]).to_csv(OUT_FILE, mode='a', header=not os.path.exists(OUT_FILE), index=False)

topic = os.environ.get("NTFY_TOPIC")
if topic:
    msg = (f"BTC अभी: ${o:,.0f}\n"
           f"Predicted close: ${row['pred_now']:,.0f}\n"
           f"1 घंटे बाद: ${row['pred_1hour']:,.0f}")
    try:
        requests.post(f"https://ntfy.sh/{topic}", data=msg.encode("utf-8"),
                      headers={"Title": "BTC Daily Prediction"}, timeout=15)
    except Exception as e:
        print("ntfy fail:", e)
