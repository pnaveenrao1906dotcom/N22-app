import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import data_delta as dd
import indicators as ind

MAXB = 3000  # चार्ट को भेजे जाने वाले अधिकतम बार (तेज़ी के लिए)


def n2_icon():
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.new("RGB", (128, 128), "#2962ff")
        f = ImageFont.load_default(size=64)
        ImageDraw.Draw(im).text((64, 64), "N2", fill="white", font=f, anchor="mm")
        return im
    except Exception:
        return "📈"


st.set_page_config(page_title="N2", page_icon=n2_icon(), layout="wide")

TF = {"1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "25m": 25, "30m": 30,
      "35m": 35, "1h": 60, "75m": 75, "77m": 77, "2h": 120, "4h": 240,
      "1d": "D", "1w": "W", "1M": "M", "1Y": "Y"}
CTYPES = ["Candlestick", "Bar", "Line", "Step line", "Heikin Ashi"]
DELTA_PRESETS = {"BTC": "BTCUSD", "ETH": "ETHUSD", "Gold (XAUT)": "XAUTUSD"}
DELTA_NATIVE = [360, 240, 120, 60, 30, 15, 5, 3, 1]
YF_NATIVE = [60, 30, 15, 5, 1]
RES = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h",
       120: "2h", 240: "4h", 360: "6h"}
YF_PERIOD = {1: 7, 5: 60, 15: 60, 30: 60, 60: 730}


def pick(v, natives):
    for n in natives:
        if v % n == 0:
            return n
    return 1


# ---------------- data (सब कैश में) ----------------
def clean_idx(d):
    return d[~d.index.duplicated(keep="last")].sort_index()


def yf_candles(sym, v, days):
    import yfinance as yf
    if isinstance(v, int):
        n = pick(v, YF_NATIVE)
        iv = "60m" if n == 60 else f"{n}m"
        d = yf.download(sym, interval=iv, period=f"{min(days, YF_PERIOD[n])}d",
                        progress=False, auto_adjust=False)
    else:
        n = None
        d = yf.download(sym, interval="1d", period="10y", progress=False,
                        auto_adjust=False).tail(days)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.columns = [c.lower() for c in d.columns]
    d.index = pd.to_datetime(d.index, utc=True)
    if "volume" not in d:
        d["volume"] = 0.0
    return d[["open", "high", "low", "close", "volume"]].astype(float), n


def resample(df, v, n):
    if df.empty or v == "D" or (isinstance(v, int) and v == n):
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if isinstance(v, int):
        mins = df.index.hour * 60 + df.index.minute
        key = [df.index.normalize(), mins // v]
        out = df.groupby(key).agg(agg)
        ts = df.index.to_series().groupby(key).first()
        out.index = pd.DatetimeIndex(ts.values, tz="UTC")
        return out.sort_index().dropna(subset=["close"])
    rule = {"W": "W-SUN", "M": "MS", "Y": "YS"}[v]
    return df.resample(rule).agg(agg).dropna(subset=["close"])


@st.cache_data(ttl=60, show_spinner=False)
def get_frame(exch, sym, tf, days, host):
    v = TF[tf]
    if exch == "delta":
        if isinstance(v, int):
            n = pick(v, DELTA_NATIVE)
            raw = dd.candles(sym, RES[n], days, host)
        else:
            n = None
            raw = dd.candles(sym, "1d", days, host)
    else:
        raw, n = yf_candles(sym, v, days)
    if raw.empty:
        return raw
    return clean_idx(resample(clean_idx(raw), v, n))


@st.cache_data(ttl=60, show_spinner=False)
def delta_chain(und, host):
    return dd.option_chain(und, host)


# ---------------- sidebar ----------------
st.sidebar.title("N2")
srcname = st.sidebar.radio("Source", ["Delta", "EURUSD"], horizontal=True)
exch = "delta" if srcname == "Delta" else "yf"
host = "India"
legs, chain_view = [], None

if srcname == "Delta":
    host = st.sidebar.radio("Delta server", ["India", "Global"], horizontal=True)
    mode = st.sidebar.radio("Type", ["Perp/Future", "Option", "Strangle"], horizontal=True)
    if mode == "Perp/Future":
        pr = st.sidebar.selectbox("Instrument", list(DELTA_PRESETS))
        legs = [st.sidebar.text_input("Symbol", DELTA_PRESETS[pr], key=f"ds_{pr}")]
    else:
        und = st.sidebar.selectbox("Options underlying", ["BTC", "ETH", "XAUT"])
        try:
            ch = delta_chain(und, host)
        except Exception as e:
            ch = pd.DataFrame()
            st.sidebar.error(f"ऑप्शन चेन एरर: {e}")
        if ch.empty:
            st.sidebar.warning("ऑप्शन चेन खाली है। ऊपर Delta server India/Global बदलकर देखें")
        else:
            ex = st.sidebar.selectbox("Expiry", sorted(ch.expiry.unique()))
            c = ch[ch.expiry == ex]
            chain_view = c
            calls, puts = c[c.type == "CALL"], c[c.type == "PUT"]
            if mode == "Option":
                t = st.sidebar.radio("Call/Put", ["CALL", "PUT"], horizontal=True)
                d = calls if t == "CALL" else puts
                k = st.sidebar.selectbox("Strike", sorted(d.strike.unique()))
                legs = [d[d.strike == k].symbol.iloc[0]]
            else:
                kc = st.sidebar.selectbox("Call strike", sorted(calls.strike.unique()))
                kp = st.sidebar.selectbox("Put strike", sorted(puts.strike.unique()))
                legs = [calls[calls.strike == kc].symbol.iloc[0],
                        puts[puts.strike == kp].symbol.iloc[0]]
else:
    legs = ["EURUSD=X"]

extras_txt = st.sidebar.text_area(
    "Extra symbols (हर लाइन में एक)", placeholder="delta:BTCUSD\nyf:EURUSD=X",
    help="Spread, Sum और Relative Strength के लिए")
auto = st.sidebar.toggle("Auto refresh (60s)", False)

# ---------------- top controls ----------------
c1, c2, c3 = st.columns([2, 2, 2])
tf = c1.selectbox("Timeframe", list(TF), index=list(TF).index("5m"))
ctype = c2.selectbox("Chart", CTYPES)
intraday = isinstance(TF[tf], int)
days = c3.slider("Days of data", 1, 30, 5) if intraday else c3.slider("Days of data", 30, 3650, 730)

if not legs:
    st.info("साइडबार में इंस्ट्रूमेंट चुनें")
    st.stop()

# ---------------- load ----------------
try:
    frames = [get_frame(exch, s, tf, days, host) for s in legs]
    if len(frames) == 2:
        idx = frames[0].index.intersection(frames[1].index)
        df = frames[0].loc[idx] + frames[1].loc[idx]
    else:
        df = frames[0]
except Exception as e:
    st.error(f"डेटा नहीं मिला: {e}")
    st.stop()
if df.empty:
    st.warning("इस सिंबल/रेंज का डेटा खाली है")
    st.stop()

symbols = {"MAIN": df}
for line in extras_txt.splitlines():
    if ":" not in line:
        continue
    ex_, sym_ = line.strip().split(":", 1)
    ex_ = {"delta": "delta", "yf": "yf"}.get(ex_.lower())
    if ex_:
        try:
            symbols[line.strip()] = get_frame(ex_, sym_, tf, days, host)
        except Exception as e:
            st.warning(f"{line}: {e}")

comp_name = None
if len(symbols) > 1:
    comp_name = st.sidebar.selectbox("Relative Strength comparison",
                                     [k for k in symbols if k != "MAIN"])
ctx = {"symbols": symbols, "comp": symbols[comp_name]["close"] if comp_name else None}

# ---------------- indicators UI ----------------
active = {}
with c1.popover("fx  Indicators"):
    for name, spec in ind.REG.items():
        if st.checkbox(name, key=f"on_{name}"):
            with st.container(border=True):
                p = {}
                for k, v in spec["params"].items():
                    key = f"{name}_{k}"
                    if k.startswith("sym"):
                        opts = ["—"] + list(symbols)
                        p[k] = st.selectbox(k, opts, index=opts.index(v) if v in opts else 0, key=key)
                    elif k in ind.CHOICES:
                        opts = ind.CHOICES[k]
                        p[k] = st.selectbox(k, opts, index=opts.index(v), key=key)
                    elif isinstance(v, bool):
                        p[k] = st.checkbox(k, v, key=key)
                    elif isinstance(v, int):
                        p[k] = st.number_input(k, value=v, step=1, key=key)
                    elif isinstance(v, float):
                        p[k] = st.number_input(k, value=v, step=0.1, format="%.3f", key=key)
                    else:
                        p[k] = st.text_input(k, v, key=key)
                p["colors"] = [st.color_picker(f"colour {i + 1}", col, key=f"{name}_c{i}")
                               for i, col in enumerate(spec["colors"])]
            active[name] = p


# ---------------- chart payload ----------------
def tsec(idx):
    t = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    return ((t - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64")


def pts(times, s, colors=None):
    v = s.values.astype(float)
    out = []
    for i in np.flatnonzero(np.isfinite(v)):
        d = {"time": int(times[i]), "value": float(v[i])}
        if colors is not None:
            d["color"] = colors[i]
        out.append(d)
    return out


def heikin(d):
    c = (d.open + d.high + d.low + d.close) / 4
    o = np.zeros(len(d))
    if len(d):
        o[0] = (d.open.iloc[0] + d.close.iloc[0]) / 2
    for i in range(1, len(d)):
        o[i] = (o[i - 1] + c.iloc[i - 1]) / 2
    o = pd.Series(o, index=d.index)
    return pd.DataFrame({"open": o, "high": pd.concat([d.high, o, c], axis=1).max(axis=1),
                         "low": pd.concat([d.low, o, c], axis=1).min(axis=1), "close": c},
                        index=d.index)


view = df.tail(MAXB)                      # इंडिकेटर पूरे डेटा पर, चार्ट को सिर्फ़ आख़िरी बार
times = tsec(view.index)
plot_df = heikin(df).reindex(view.index) if ctype == "Heikin Ashi" else view
payload = {"times": [int(t) for t in times],
           "type": {"Candlestick": "candle", "Heikin Ashi": "candle", "Bar": "bar",
                    "Line": "line", "Step line": "step"}[ctype]}
if payload["type"] in ("candle", "bar"):
    payload["ohlc"] = [{"time": int(t), "open": float(o), "high": float(h),
                        "low": float(l), "close": float(c)}
                       for t, o, h, l, c in zip(times, plot_df.open, plot_df.high,
                                                plot_df.low, plot_df.close)
                       if np.isfinite([o, h, l, c]).all()]
else:
    payload["line"] = [{"time": int(t), "value": float(c)}
                       for t, c in zip(times, view.close) if np.isfinite(c)]

series, hl_main, pane_i = [], [], 0
for name, p in active.items():
    try:
        lines = ind.REG[name]["fn"](df, p, ctx)
    except Exception as e:
        st.warning(f"{name}: {e}")
        continue
    if not lines and name == "Relative Strength":
        st.warning("Relative Strength के लिए Extra symbols में एक सिंबल डालें")
    if any(l["pane"] == "sub" for l in lines):
        pane_i += 1
    for l in lines:
        try:
            if l.get("data") is None:
                hl_main += l.get("hlines", [])
                continue
            s = l["data"].reindex(view.index)
            cols = None
            if l.get("colors_pt") is not None:
                cols = pd.Series(l["colors_pt"], index=df.index).reindex(view.index).values
            series.append({"name": l["name"], "kind": l.get("kind", "line"), "color": l["color"],
                           "width": l.get("width", 1), "pane": pane_i if l["pane"] == "sub" else 0,
                           "data": pts(times, s, cols), "hlines": l.get("hlines", [])})
        except Exception as e:
            st.warning(f"{name}/{l.get('name')}: {e}")
payload["ind"], payload["hlines"] = series, hl_main

# ---------------- tabs ----------------
tab1, tab2 = st.tabs(["Chart", "Option chain"])
with tab1:
    html = Path(__file__).with_name("chart.html").read_text(encoding="utf-8")
    html = html.replace("__PAYLOAD__", json.dumps(payload, default=float))
    components.html(html, height=520 + 170 * pane_i, scrolling=False)

with tab2:
    if chain_view is None:
        st.info("Option या Strangle मोड चुनें")
    else:
        c = chain_view
        cc, pp = c[c.type == "CALL"].set_index("strike"), c[c.type == "PUT"].set_index("strike")
        cols = ["ltp", "oi", "volume", "iv", "delta", "gamma", "theta", "vega"]
        tbl = cc[cols].add_prefix("C_").join(pp[cols].add_prefix("P_"), how="outer").sort_index()
        a, b = st.columns(2)
        a.metric("PCR (OI)", round(pp.oi.sum() / max(cc.oi.sum(), 1e-9), 3))
        b.metric("PCR (Volume)", round(pp.volume.sum() / max(cc.volume.sum(), 1e-9), 3))
        st.dataframe(tbl, use_container_width=True)

if auto:
    time.sleep(60)
    st.rerun()
