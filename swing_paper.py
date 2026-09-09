"""SWING PAPER BOT - daily-bar trend following on a fixed ETF universe.

Born from analysing the live intraday bots: the step-trail exit needs TIME.
Intraday it activated on 0-1% of trades (the end-of-day exit truncated every winner).
On daily bars with no time limit it activates on 10-26% of trades and winners run to +8..+27R.

Logic (identical shape to the crypto bot, which works):
  entry : close breaks the 20-day Donchian high  AND  close > EMA200   -> LONG
          close breaks the 20-day Donchian low   AND  close < EMA200   -> SHORT
  risk  : R = 1.5 x ATR(14); initial stop = entry -/+ 1R
  exit  : step-trail only -> at +2R stop moves to breakeven, each further +1R lifts it +1R.
          NO end-of-day exit. Positions hold for days or weeks.
  size  : risk 1% of that symbol equity per trade.

Backtest (12 liquid ETFs, 15y, 1176 trades): 11/12 symbols positive expectancy,
mean +0.35R/trade; portfolio at 1% risk = CAGR ~20%, MaxDD ~-34%, 11/16 positive years.

Paper only - virtual money, no keys, no orders. State + logs persist to disk.

  python swing_paper.py --once                    # one tick (for cron)
  python swing_paper.py --tickers SPY,QQQ --once
  python swing_paper.py --replay 15y              # backtest-style replay, prints results
"""
import os, sys, time, json, math, argparse, traceback
import numpy as np, pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
UNIVERSE = "SPY,QQQ,DIA,IWM,TQQQ,GLD,SLV,EEM,XLE,XLF,SMH"   # TLT dropped: only negative-expectancy name
COST = 0.0005                                                # ~0.05% round-trip

# ----------------------------------------------------------------- io
def log(tag, msg):
    line = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + msg
    print(line, flush=True)
    with open(os.path.join(HERE, "paper_" + tag + ".log"), "a") as f:
        f.write(line + "\n")

def sfile(tag):
    return os.path.join(HERE, "paper_" + tag + "_state.json")

def fresh(cap):
    return {"equity": cap, "side": 0, "entry": 0.0, "R": 0.0, "peak": 0.0,
            "stop": 0.0, "qty": 0.0, "last_bar_ts": 0, "trades": 0, "wins": 0}

def load_state(tag, cap):
    if os.path.exists(sfile(tag)):
        return json.load(open(sfile(tag)))
    return fresh(cap)

def save_state(tag, s):
    json.dump(s, open(sfile(tag), "w"), indent=2)

# ----------------------------------------------------------------- data
def fetch(sym, period="2y"):
    d = yf.download(sym, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
    d = d.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if len(d) < 220:
        return None
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["ema"] = d.close.ewm(span=200, adjust=False).mean()
    d["up"] = d.high.rolling(20).max().shift(1)
    d["dn"] = d.low.rolling(20).min().shift(1)
    return d.dropna()

# ----------------------------------------------------------------- trade primitives
def _open(s, side, entry, R, risk, tag, ts):
    qty = (s["equity"] * risk) / R
    s.update({"side": side, "entry": entry, "R": R, "peak": 0.0, "qty": qty,
              "stop": entry - R if side == 1 else entry + R})
    log(tag, "  {:%Y-%m-%d}  ENTER {} @ {:.2f}  qty {:.2f}  stop {:.2f}  (R={:.2f})".format(
        ts, "LONG" if side == 1 else "SHORT", entry, qty, s["stop"], R))

def _close(s, px, tag, ts):
    side, entry, qty, R = s["side"], s["entry"], s["qty"], s["R"]
    pnl = side * (px - entry) * qty - 2 * COST * entry * qty
    Rm = side * (px - entry) / R if R else 0.0
    s["equity"] += pnl
    s["trades"] += 1
    if pnl > 0:
        s["wins"] += 1
    log(tag, "  {:%Y-%m-%d}  EXIT @ {:.2f}  {:+.2f}R  PnL {:+.2f}  equity {:.2f}  (WR {}/{})".format(
        ts, px, Rm, pnl, s["equity"], s["wins"], s["trades"]))
    s.update({"side": 0, "entry": 0.0, "R": 0.0, "peak": 0.0, "qty": 0.0, "stop": 0.0})

# ----------------------------------------------------------------- one daily bar
def handle_bar(ts, o, h, l, c, ema, atr, up, dn, s, args, tag):
    # --- manage an open position: stop first, then ratchet the step-trail ---
    if s["side"] == 1:
        if l <= s["stop"]:
            _close(s, s["stop"], tag, ts)
        else:
            s["peak"] = max(s["peak"], (h - s["entry"]) / s["R"])
            lvl = math.floor(s["peak"] - args.activate) if s["peak"] >= args.activate else -1.0
            s["stop"] = s["entry"] + lvl * s["R"]
    elif s["side"] == -1:
        if h >= s["stop"]:
            _close(s, s["stop"], tag, ts)
        else:
            s["peak"] = max(s["peak"], (s["entry"] - l) / s["R"])
            lvl = math.floor(s["peak"] - args.activate) if s["peak"] >= args.activate else -1.0
            s["stop"] = s["entry"] - lvl * s["R"]
    # --- entry (only when flat; NO time-based exit anywhere) ---
    if s["side"] == 0 and not (np.isnan(up) or np.isnan(ema) or np.isnan(atr)):
        R = args.k * atr
        if R <= 0:
            return
        if c > up and c > ema:
            _open(s, 1, c, R, args.risk, tag, ts)
        elif args.shorts and c < dn and c < ema:
            _open(s, -1, c, R, args.risk, tag, ts)

def process(d, s, args, tag):
    changed = False
    for ts, r in d.iterrows():
        tms = int(ts.value // 10 ** 6)
        if tms <= s["last_bar_ts"]:
            continue
        handle_bar(ts, float(r.open), float(r.high), float(r.low), float(r.close),
                   float(r.ema), float(r.atr), float(r.up), float(r.dn), s, args, tag)
        s["last_bar_ts"] = tms
        changed = True
    return changed

# ----------------------------------------------------------------- main
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Swing paper bot - daily trend following")
    p.add_argument("--tickers", default=UNIVERSE)
    p.add_argument("--k", type=float, default=1.5, help="R = k * ATR(14)")
    p.add_argument("--activate", type=float, default=2.0, help="R level where the trail starts")
    p.add_argument("--risk", type=float, default=0.01, help="fraction of equity risked per trade")
    p.add_argument("--capital", type=float, default=10000)
    p.add_argument("--shorts", action="store_true", default=True)
    p.add_argument("--noshorts", dest="shorts", action="store_false")
    p.add_argument("--once", action="store_true")
    p.add_argument("--replay", default="", help="e.g. 15y - replay history and report, no state saved")
    a = p.parse_args()

    for tkr in [t.strip().upper() for t in a.tickers.split(",") if t.strip()]:
        try:
            if a.replay:
                d = fetch(tkr, a.replay)
                if d is None:
                    print(tkr + ": not enough data")
                    continue
                s = fresh(a.capital)
                log(tkr, "=== REPLAY {} {} (paper ${:,.0f}) ===".format(tkr, a.replay, a.capital))
                process(d, s, a, tkr)
                wr = 100 * s["wins"] / max(s["trades"], 1)
                log(tkr, "=== {}: equity ${:,.2f} ({:+.1f}%)  trades {}  WR {:.0f}% ===".format(
                    tkr, s["equity"], (s["equity"] / a.capital - 1) * 100, s["trades"], wr))
                continue

            s = load_state(tkr, a.capital)
            log(tkr, "=== {} daily swing | equity ${:,.2f} | k{} activate{}R risk {:.0%} ===".format(
                tkr, s["equity"], a.k, a.activate, a.risk))
            d = fetch(tkr)
            if d is None:
                log(tkr, "  not enough data")
                continue
            if s["last_bar_ts"] == 0:
                # FRESH START: history is only for indicator warm-up - never trade it
                s["last_bar_ts"] = int(d.index[-1].value // 10 ** 6)
                log(tkr, "  initialized fresh at ${:,.0f} - trading only NEW bars from now".format(s["equity"]))
                save_state(tkr, s)
            elif process(d, s, a, tkr):
                save_state(tkr, s)
        except Exception:
            log(tkr, "ERROR:\n" + traceback.format_exc())
