"""DSE Price Forecast: a phone-first Streamlit app built from predictive_analytics_LSTM.ipynb.

Same method as the notebook: one small LSTM per share, 20-session window, closing
price only, time-based 80/20 split, recursive multi-step forecast over Sun-Thu sessions.
"""
from __future__ import annotations

import html
import inspect
import re
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Settings (match the notebook)
# --------------------------------------------------------------------------------------
WINDOW = 20
TRAIN_RATIO = 0.80
EPOCHS = 15
BATCH_SIZE = 16
LOOKBACK_DAYS = 730  # approximately two calendar years
WEEKMASK = "Sun Mon Tue Wed Thu"  # DSE trading week (public holidays are not modelled)
HORIZONS = {"Next day": 1, "1 week": 5, "2 weeks": 10, "1 month": 22}

INK, TEAL, AMBER, UP, DOWN, GRID = "#12292C", "#0E7C86", "#D98E04", "#1B7F4B", "#C23B22", "#D5E2E2"

COLUMNS = ["DATE", "TRADING CODE", "LTP*", "HIGH", "LOW", "OPENP*", "CLOSEP*", "YCP", "TRADE", "VALUE (mn)", "VOLUME"]
ALIASES = {
    "date": "DATE", "tradingcode": "TRADING CODE", "symbol": "TRADING CODE", "code": "TRADING CODE",
    "ltp": "LTP*", "high": "HIGH", "low": "LOW", "openp": "OPENP*", "open": "OPENP*",
    "closep": "CLOSEP*", "close": "CLOSEP*", "ycp": "YCP", "trade": "TRADE",
    "value": "VALUE (mn)", "valuemn": "VALUE (mn)", "volume": "VOLUME",
}

# Mutual funds and treasury instruments are not ordinary shares, so they are left out.
EXCLUDED_CODES = {
    "1JANATAMF", "1STPRIMFMF", "AIBL1STIMF", "ATCSLGF", "CAPMBDBLMF", "CAPMIBBLMF", "DBH1STMF",
    "EBL1STMF", "EBLNRBMF", "EXIM1STMF", "FBFIF", "IC3RDNRB", "ICBAGRANI1", "ICBAMCL2ND",
    "ICBEPMF1S1", "ICBSONALI1", "IFILISLMF1", "LANKABA1MF", "LRGLOBMF1", "MBL1STMF", "NCCBLMF1",
    "NLI1STMF", "PHPMF1", "POPULAR1MF", "PRIME1ICBA", "RELIANCE1", "SEMLIBBLSF", "SEMLLECMF",
    "TRUSTB1MF", "VAMLBDG1", "VAMLRBBF",
}
EXCLUDE_PATTERN = r"MF$|MUTUAL|BOND|DEB|TB\d|TNOTE|TBILL"

st.set_page_config(
    page_title="DSE Price Forecast",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Newer Streamlit uses width="stretch"; older versions use use_container_width=True.
if "width" in inspect.signature(st.button).parameters:
    STRETCH = {"width": "stretch"}
else:
    STRETCH = {"use_container_width": True}

# --------------------------------------------------------------------------------------
# Styling: a single phone-width column, one bold forecast card, quiet everything else
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap');
.stApp {{ font-family: 'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.block-container {{ max-width: 620px; padding: 1.1rem 1rem 4rem; }}
footer, #MainMenu {{ visibility: hidden; }}
h1 {{ font-size: 1.65rem !important; font-weight: 800 !important; letter-spacing: -0.02em; padding: 0 0 .15rem !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 700 !important; }}
.sub {{ color: #4B6568; font-size: .95rem; margin: 0 0 .9rem; line-height: 1.35; }}
.stButton > button, .stDownloadButton > button {{ min-height: 3rem; border-radius: 12px; font-weight: 700; }}

.fc {{ border-radius: 18px; overflow: hidden; background: #fff; border: 1px solid {GRID}; margin: .4rem 0 1rem; }}
.fc-main {{ background: linear-gradient(160deg, #0E7C86 0%, #0A565D 100%); color: #fff; padding: 1.1rem 1.2rem 1.15rem; }}
.fc-label {{ font-size: .85rem; opacity: .85; font-weight: 500; }}
.fc-price {{ font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; margin: .2rem 0 .15rem; }}
.fc-delta {{ font-size: 1.05rem; font-weight: 700; }}
.fc-delta.up {{ color: #A6EBC0; }}
.fc-delta.down {{ color: #FFC2B6; }}
.fc-row {{ display: grid; grid-template-columns: repeat(3, 1fr); }}
.fc-row > div {{ padding: .8rem .9rem; border-right: 1px solid {GRID}; }}
.fc-row > div:last-child {{ border-right: 0; }}
.k {{ display: block; font-size: .72rem; color: #4B6568; font-weight: 500; }}
.v {{ display: block; font-size: 1.1rem; font-weight: 700; color: {INK}; }}
.s {{ display: block; font-size: .72rem; color: #6F8587; }}
.stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: .6rem; margin: .3rem 0 .8rem; }}
.stats > div {{ background: #fff; border: 1px solid {GRID}; border-radius: 12px; padding: .65rem .8rem; }}
.note {{ background: #FFF6E0; border-left: 4px solid {AMBER}; border-radius: 8px; padding: .65rem .8rem; font-size: .9rem; margin: .5rem 0; }}
.demo {{ background: #E8F1FF; border-left: 4px solid #3B6FD1; border-radius: 8px; padding: .65rem .8rem; font-size: .9rem; margin: .3rem 0 .7rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Tidy a DSE response into the standard columns (same logic as the notebook's API)."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = raw.reset_index().copy()
    df.columns = [ALIASES.get(re.sub(r"[^a-z0-9]", "", str(c).lower()), str(c)) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    if not {"DATE", "TRADING CODE", "CLOSEP*"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["TRADING CODE"] = df["TRADING CODE"].astype(str).str.upper().str.strip()
    for col in COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=["DATE"]).drop_duplicates(subset=["DATE", "TRADING CODE"])
    return df[COLUMNS].sort_values(["DATE", "TRADING CODE"]).reset_index(drop=True)


def to_frame(norm: pd.DataFrame) -> pd.DataFrame:
    """Keep ordinary shares with a real closing price (DSE reports 0 on days with no trades)."""
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(norm["DATE"]),
            "code": norm["TRADING CODE"],
            "close": norm["CLOSEP*"],
            "volume": norm["VOLUME"],
        }
    ).dropna(subset=["date", "code", "close"])
    out = out[out["close"] > 0]
    bad = out["code"].isin(EXCLUDED_CODES) | out["code"].str.contains(EXCLUDE_PATTERN, regex=True, na=False)
    bad |= out["code"].str.fullmatch(r"TB\d+Y\d+", na=False)
    return out[~bad].sort_values(["code", "date"]).reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_dse(today_iso: str) -> pd.DataFrame:
    from bdshare import get_historical_data

    end = date.fromisoformat(today_iso)
    start = end - timedelta(days=LOOKBACK_DAYS)
    frame = to_frame(normalize(get_historical_data(start.isoformat(), end.isoformat())))
    if frame.empty:
        raise ValueError("The exchange returned no price rows for this period.")
    return frame


def demo_frame() -> pd.DataFrame:
    """Synthetic random-walk prices so the app can be explored when DSE is unreachable."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=110, freq="C", weekmask=WEEKMASK)
    starts = {"BRACBANK": 41, "GP": 285, "SQURPHARMA": 210, "BATBC": 390, "ROBI": 26, "RENATA": 760,
              "CITYBANK": 24, "ISLAMIBANK": 33, "LHB": 54, "WALTONHIL": 880}
    rows = []
    for code, p0 in starts.items():
        price = p0 * np.exp(np.cumsum(rng.normal(0.0003, 0.013, len(days))))
        vol = rng.lognormal(12, 0.5, len(days))
        rows.append(pd.DataFrame({"date": days, "code": code, "close": price.round(1), "volume": vol.round()}))
    return pd.concat(rows, ignore_index=True)


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
        # network, DSE downtime, library change
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
# Model
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=10)
def train_lstm(code: str, last_date: str, prices: tuple):
