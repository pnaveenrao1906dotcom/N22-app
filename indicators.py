import numpy as np
import pandas as pd

SRC_BASE = ["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"]
DERIVED = ["OBV", "RSI(14)", "ADX(14)", "BBW(20)"]
SOURCES = SRC_BASE + DERIVED
CHOICES = {"ma_type": ["SMA", "EMA"], "sg2": ["+", "-"], "sg3": ["+", "-"], "src": SOURCES}


# ---------- basic maths ----------
def ma(s, n, kind="SMA"):
    n = int(n)
    return s.ewm(span=n, adjust=False).mean() if kind == "EMA" else s.rolling(n).mean()


def rsi(s, n=14):
    d = s.diff()
    ru = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    rd = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))


def obv(df):
    return (np.sign(df.close.diff()).fillna(0) * df.volume).cumsum()


def true_range(df):
    pc = df.close.shift()
    return pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up, dn = df.high.diff(), -df.low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def bbw(s, n=20, m=2.0):
    return 2 * m * s.rolling(int(n)).std(ddof=0) / s.rolling(int(n)).mean() * 100


def kalman(s, q, r):
    v = s.values.astype(float)
    est = next((z for z in v if not np.isnan(z)), np.nan)
    p, out = 1.0, np.full(len(v), np.nan)
    for i, z in enumerate(v):
        if not np.isnan(z):
            p += q
            k = p / (p + r)
            est = est + k * (z - est)
            p = (1 - k) * p
        out[i] = est
    return pd.Series(out, index=s.index)


def get_src(df, name):
    if name == "hl2":
        return (df.high + df.low) / 2
    if name == "hlc3":
        return (df.high + df.low + df.close) / 3
    if name == "ohlc4":
        return (df.open + df.high + df.low + df.close) / 4
    if name in ("close", "open", "high", "low"):
        return df[name]
    if name == "OBV":
        return obv(df)
    if name == "RSI(14)":
        return rsi(df.close, 14)
    if name == "ADX(14)":
        return adx(df, 14)[0]
    if name == "BBW(20)":
        return bbw(df.close, 20, 2.0)
    return df.close


def L(name, s, color, pane="over", **kw):
    return dict(name=name, data=s, color=color, pane=pane, **kw)


# ---------- indicators ----------
def f_ma(df, p, ctx):
    s = get_src(df, p["src"])
    out = [L(f"MA{p['length']}", ma(s, p["length"], p["ma_type"]), p["colors"][0], width=2)]
    if int(p["length2"]) > 0:
        out.append(L(f"MA{p['length2']}", ma(s, p["length2"], p["ma_type"]), p["colors"][1], width=2))
    return out


def f_vwap(df, p, ctx):
    pv = (df.high + df.low + df.close) / 3 * df.volume
    day = df.index.date
    num, den = pv.groupby(day).cumsum(), df.volume.groupby(day).cumsum()
    return [L("VWAP", num / den.replace(0, np.nan), p["colors"][0], width=2)]


def f_vp(df, p, ctx):
    px = (df.high + df.low + df.close) / 3
    hist, edges = np.histogram(px, bins=int(p["rows"]), weights=df.volume)
    mid = (edges[:-1] + edges[1:]) / 2
    i = int(hist.argmax())
    lo = hi = i
    acc, tot = hist[i], hist.sum()
    while acc < tot * p["va_pct"] / 100 and (lo > 0 or hi < len(hist) - 1):
        a = hist[lo - 1] if lo > 0 else -1
        b = hist[hi + 1] if hi < len(hist) - 1 else -1
        if a >= b:
            lo -= 1
            acc += hist[lo]
        else:
            hi += 1
            acc += hist[hi]
    c = p["colors"]
    return [dict(name="VP", data=None, pane="over", hlines=[
        dict(price=float(mid[i]), color=c[0], title="POC"),
        dict(price=float(mid[hi]), color=c[1], title="VAH"),
        dict(price=float(mid[lo]), color=c[2], title="VAL")])]


def f_vol(df, p, ctx):
    cols = np.where(df.close >= df.open, p["colors"][0], p["colors"][1])
    return [L("Volume", df.volume, p["colors"][0], "sub", kind="hist", colors_pt=list(cols))]


def f_obv(df, p, ctx):
    o, c, out = obv(df), p["colors"], []
    if p["show_obv"]:
        out.append(L("OBV", o, c[0], "sub"))
    if p["show_ma1"]:
        out.append(L(f"OBV MA{p['ma1']}", ma(o, p["ma1"], p["ma_type"]), c[1], "sub"))
    if p["show_ma2"]:
        out.append(L(f"OBV MA{p['ma2']}", ma(o, p["ma2"], p["ma_type"]), c[2], "sub"))
    return out


def f_rsi(df, p, ctx):
    r, c, out = rsi(get_src(df, p["src"]), int(p["length"])), p["colors"], []
    out.append(L("RSI", r, c[0], "sub", hlines=[dict(price=30, color="#888", title=""),
                                                  dict(price=70, color="#888", title="")]))
    if p["show_ma1"]:
        out.append(L(f"RSI MA{p['ma1']}", ma(r, p["ma1"], p["ma_type"]), c[1], "sub"))
    if p["show_ma2"]:
        out.append(L(f"RSI MA{p['ma2']}", ma(r, p["ma2"], p["ma_type"]), c[2], "sub"))
    return out


def f_rs(df, p, ctx):
    comp = ctx.get("comp")
    if comp is None:
        return []
    rs = df.close / comp.reindex(df.index, method="ffill") * 100
    c = p["colors"]
    out = [L("Rel.Strength", rs, c[0], "sub")]
    if p["show_ma1"]:
        out.append(L(f"RS MA{p['ma1']}", ma(rs, p["ma1"], p["ma_type"]), c[1], "sub"))
    if p["show_ma2"]:
        out.append(L(f"RS MA{p['ma2']}", ma(rs, p["ma2"], p["ma_type"]), c[2], "sub"))
    return out


def f_kalman(df, p, ctx):
    k = kalman(get_src(df, p["src"]), p["q"], p["r"])
    col = np.where(k.diff() >= 0, p["colors"][0], p["colors"][1])
    return [L("Kalman", k, p["colors"][0], width=2, colors_pt=list(col))]


def f_gmma(df, p, ctx):
    s, out = get_src(df, p["src"]), []
    for key, ci in (("short", 0), ("long", 1)):
        for n in [int(x) for x in str(p[key]).split(",") if x.strip()]:
            out.append(L(f"G{n}", ma(s, n, "EMA"), p["colors"][ci]))
    return out


def f_adx(df, p, ctx):
    a, pdi, mdi = adx(df, int(p["length"]))
    c = p["colors"]
    out = [L("ADX", a, c[0], "sub", width=2)]
    if p["show_di"]:
        out += [L("+DI", pdi, c[1], "sub"), L("-DI", mdi, c[2], "sub")]
    if p["show_bands"]:
        lb = int(p["lookback"])
        out += [L("ADX High", a.rolling(lb, min_periods=1).max(), c[3], "sub"),
                L("ADX Low", a.rolling(lb, min_periods=1).min(), c[4], "sub")]
    return out


def f_bbw(df, p, ctx):
    w = bbw(get_src(df, p["src"]), p["length"], p["mult"])
    c = p["colors"]
    out = [L("BBW", w, c[0], "sub", width=2)]
    if p["show_bands"]:
        lb = int(p["lookback"])
        out += [L("Highest Expansion", w.rolling(lb, min_periods=1).max(), c[1], "sub"),
                L("Lowest Contraction", w.rolling(lb, min_periods=1).min(), c[2], "sub")]
    return out


def _band(name, basis, up, lo, c):
    return [L(f"{name} Up", up, c[0]), L(f"{name} Mid", basis, c[1]), L(f"{name} Low", lo, c[2])]


def f_pct(df, p, ctx):
    b = ma(get_src(df, p["src"]), p["length"], p["ma_type"])
    return _band("%B", b, b * (1 + p["pct"] / 100), b * (1 - p["pct"] / 100), p["colors"])


def f_atrb(df, p, ctx):
    b = ma(get_src(df, p["src"]), p["length"], p["ma_type"])
    a = atr(df, int(p["atr_len"])) * p["mult"]
    return _band("ATR", b, b + a, b - a, p["colors"])


def f_hlb(df, p, ctx):
    hi, lo = df.high.rolling(int(p["length"])).max(), df.low.rolling(int(p["length"])).min()
    return _band("HL", (hi + lo) / 2, hi, lo, p["colors"])


def _sym(df, ctx, name):
    return ctx["symbols"][name]["close"].reindex(df.index, method="ffill")


def f_spread(df, p, ctx):
    if "—" in (p["sym_1"], p["sym_2"]):
        return []
    return [L("S1-S2", _sym(df, ctx, p["sym_1"]) - _sym(df, ctx, p["sym_2"]), p["colors"][0], "sub", width=2)]


def f_sum3(df, p, ctx):
    if p["sym_1"] == "—":
        return []
    tot = _sym(df, ctx, p["sym_1"])
    for sg, key in ((p["sg2"], "sym_2"), (p["sg3"], "sym_3")):
        if p[key] != "—":
            tot = tot + (1 if sg == "+" else -1) * _sym(df, ctx, p[key])
    return [L("S1±S2±S3", tot, p["colors"][0], "sub", width=2)]


REG = {
    "MA": dict(fn=f_ma, params=dict(src="close", length=20, length2=50, ma_type="EMA"),
               colors=["#f0b90b", "#e040fb"]),
    "VWAP": dict(fn=f_vwap, params={}, colors=["#2962ff"]),
    "Volume Profile": dict(fn=f_vp, params=dict(rows=40, va_pct=70),
                           colors=["#ff9800", "#9c27b0", "#9c27b0"]),
    "Volume": dict(fn=f_vol, params={}, colors=["#26a69a", "#ef5350"]),
    "OBV": dict(fn=f_obv, params=dict(show_obv=True, show_ma1=True, ma1=20, show_ma2=True, ma2=50,
                                      ma_type="SMA"), colors=["#2962ff", "#f0b90b", "#e040fb"]),
    "RSI": dict(fn=f_rsi, params=dict(src="close", length=14, ma_type="SMA", show_ma1=True, ma1=14,
                                      show_ma2=False, ma2=28), colors=["#7e57c2", "#f0b90b", "#e040fb"]),
    "Relative Strength": dict(fn=f_rs, params=dict(ma_type="SMA", show_ma1=True, ma1=20,
                                                    show_ma2=True, ma2=50),
                              colors=["#2962ff", "#f0b90b", "#e040fb"]),
    "Kalman Filter": dict(fn=f_kalman, params=dict(src="close", q=0.05, r=1.0),
                          colors=["#26a69a", "#ef5350"]),
    "GMMA": dict(fn=f_gmma, params=dict(src="close", short="3,5,8,10,12,15", long="30,35,40,45,50,60"),
                 colors=["#26a69a", "#ef5350"]),
    "ADX": dict(fn=f_adx, params=dict(length=14, show_di=False, show_bands=True, lookback=125),
                colors=["#2962ff", "#26a69a", "#ef5350", "#b71c1c", "#00897b"]),
    "BB Width": dict(fn=f_bbw, params=dict(src="close", length=20, mult=2.0, show_bands=True, lookback=125),
                     colors=["#2962ff", "#b71c1c", "#00897b"]),
    "Percent Bands": dict(fn=f_pct, params=dict(src="close", length=20, pct=2.0, ma_type="SMA"),
                          colors=["#26a69a", "#9e9e9e", "#ef5350"]),
    "ATR Bands": dict(fn=f_atrb, params=dict(src="close", length=20, atr_len=14, mult=2.0, ma_type="SMA"),
                      colors=["#26a69a", "#9e9e9e", "#ef5350"]),
    "High-Low Bands": dict(fn=f_hlb, params=dict(length=20), colors=["#26a69a", "#9e9e9e", "#ef5350"]),
    "Symbol1 - Symbol2": dict(fn=f_spread, params=dict(sym_1="MAIN", sym_2="MAIN"), colors=["#ff9800"]),
    "Sum of 3 symbols": dict(fn=f_sum3, params=dict(sym_1="MAIN", sg2="+", sym_2="—", sg3="+", sym_3="—"),
                             colors=["#ff9800"]),
}
