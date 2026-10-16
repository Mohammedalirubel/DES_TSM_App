# DSE Price Forecast (Streamlit)

Phone-first app that forecasts Dhaka Stock Exchange closing prices with the per-share LSTM from `predictive_analytics_LSTM.ipynb`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the "Network URL" it prints on your phone (same Wi-Fi) to try it.

## Put it on your phone as an app
1. Push this folder to a GitHub repo (keep `.streamlit/config.toml`).
2. Go to share.streamlit.io, create an app from the repo, main file `app.py`, Python 3.12.
3. Open the app URL on your phone, then:
   - iPhone (Safari): Share, then Add to Home Screen.
   - Android (Chrome): menu, then Add to Home screen / Install app.

## Notes
- Prices load from DSE via `bdshare` (last 150 days, cached 6 hours). If DSE is down the app offers a retry, CSV upload, or synthetic demo data.
- Models train on demand per share (about 10-20 s the first time) and are cached while the server runs. Streamlit Cloud sleeps idle apps, so the cache resets.
