from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

# Constants used by train_lstm and load_data
WINDOW = 30
TRAIN_RATIO = 0.8
EPOCHS = 20
BATCH_SIZE = 32
STRETCH = {"use_container_width": True}
# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
# ... Keep your normalize(), to_frame(), fetch_dse(), and demo_frame() functions here ...

def load_data() -> tuple[pd.DataFrame, str]:
    source = st.session_state.get("source", "dse")
    if source == "demo":
        return demo_frame(), "demo"
    if source == "csv":
        return st.session_state["csv_frame"], "csv"
    try:
        with st.spinner("Loading latest prices from DSE…"):
            return fetch_dse(date.today().isoformat()), "dse"
    except Exception as exc:
        st.markdown("### Prices didn't load")
        st.write("The Dhaka Stock Exchange feed didn't respond. You can try again or use another source.")
        st.caption(f"Reason ({type(exc).__name__}): {exc}")
        if isinstance(exc, ModuleNotFoundError):
            st.warning(
                f"The package '{exc.name}' is not installed on this server. Add it to requirements.txt "
                "(in the repo root, next to app.py), push, then reboot the app."
            )
        if st.button("Try again", type="primary", **STRETCH):
            st.rerun()
        up = st.file_uploader("Or upload a CSV with DATE, TRADING CODE and CLOSEP* columns", type="csv")
        if up is not None:
            try:
                st.session_state["csv_frame"] = to_frame(normalize(pd.read_csv(up)))
                st.session_state["source"] = "csv"
                st.rerun()
            except Exception as err:
                st.error(f"That file couldn't be read: {err}")
        if st.button("Explore with demo data", **STRETCH):
            st.session_state["source"] = "demo"
            st.rerun()
        st.stop()


# --------------------------------------------------------------------------------------
# Model (Defined at global root level, NOT nested inside load_data)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=10)
def train_lstm(code: str, last_date: str, prices: tuple):
    """Train one LSTM for one share. The scaler is fitted on training rows only."""
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras

    tf.keras.utils.set_random_seed(42)
    arr = np.asarray(prices, dtype="float64").reshape(-1, 1)
    n = len(arr)
    train_size = int(n * TRAIN_RATIO)
    if n < WINDOW + 10 or train_size <= WINDOW + 10:
        raise ValueError(f"{code} has only {n} trading days of data; at least {int((WINDOW + 11) / TRAIN_RATIO)} are needed.")
    
    scaler = MinMaxScaler().fit(arr[:train_size])
    scaled = scaler.transform(arr).ravel()
    X = np.stack([scaled[i - WINDOW : i] for i in range(WINDOW, n)]).astype("float32")[..., None]
    y = scaled[WINDOW:].astype("float32")
    target_idx = np.arange(WINDOW, n)
    train_mask, test_mask = target_idx < train_size, target_idx >= train_size
    
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(WINDOW, 1)),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(X[train_mask], y[train_mask], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=False, verbose=0)
    
    pred = scaler.inverse_transform(model(X[test_mask], training=False).numpy()).ravel()
    actual = arr[WINDOW:, 0][test_mask]
    previous = arr[WINDOW - 1 : -1, 0][test_mask]
    moved = actual != previous
    if moved.any():
        direction = float(np.mean(np.sign(pred - previous)[moved] == np.sign(actual - previous)[moved])) * 100
    else:
        direction = float("nan")
        
    metrics = {
        "mae": float(np.mean(np.abs(pred - actual))),
        "naive_mae": float(np.mean(np.abs(previous - actual))),
        "mape": float(np.mean(np.abs(pred - actual) / actual) * 100),
        "direction": direction,
        "test_days": int(test_mask.sum()),
        "train_loss": float(hist.history["loss"][-1]),
        "val_loss": float(hist.history["val_loss"][-1]),
    }
    return model, scaler, metrics
