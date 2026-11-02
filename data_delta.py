import re
import time
import pandas as pd
import requests

HOSTS = {"India": "https://api.india.delta.exchange/v2",
         "Global": "https://api.delta.exchange/v2"}
SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
       "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "1d": 86400}
S = requests.Session()
S.headers.update({"User-Agent": "n2-app", "Accept": "application/json"})


def _get(host, path, params=None):
    r = S.get(HOSTS[host] + path, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Delta {host} HTTP {r.status_code}: {r.text[:200]}")
    j = r.json()
    if j.get("success") is False:
        raise RuntimeError(f"Delta {host}: {j.get('error')}")
    return j.get("result", [])


def candles(symbol, res="5m", days=5, host="India"):
    end = int(time.time())
    start = end - days * 86400
    step = SEC[res] * 1999
    out, t = [], start
    while t < end:
        t2 = min(t + step, end)
        out += _get(host, "/history/candles",
                    {"symbol": symbol, "resolution": res, "start": t, "end": t2})
        t = t2
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df.drop_duplicates("time").sort_values("time")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)


def option_chain(underlying, host="India"):
    rows = _get(host, "/tickers", {"contract_types": "call_options,put_options",
                                   "underlying_asset_symbols": underlying})
    recs = []
    for t in rows:
        m = re.match(r"([CP])-([A-Z0-9]+)-(\d+(?:\.\d+)?)-(\d{6})$", t.get("symbol", ""))
        if not m:
            continue
        g = t.get("greeks") or {}
        q = t.get("quotes") or {}
        recs.append({
            "symbol": t["symbol"],
            "type": "CALL" if m.group(1) == "C" else "PUT",
            "strike": float(m.group(3)),
            "expiry": pd.to_datetime(m.group(4), format="%d%m%y").date(),
            "ltp": t.get("close"), "oi": t.get("oi"), "volume": t.get("volume"),
            "iv": q.get("mark_iv"), "delta": g.get("delta"), "gamma": g.get("gamma"),
            "theta": g.get("theta"), "vega": g.get("vega"),
        })
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    num = ["ltp", "oi", "volume", "iv", "delta", "gamma", "theta", "vega"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df
