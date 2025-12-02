import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import requests
import yfinance as yf

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from statistics import mean, pstdev

app = FastAPI()

# 允许所有源访问，方便你以后前端独立出来用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def fetch_price_metrics(symbol: str) -> Optional[Dict[str, float]]:
    """拉取某个标的（SPY/QQQ）近 1 年的收盘价，算 50/100/200 日线与乖离。"""
    try:
        df = yf.download(symbol, period="1y", auto_adjust=False, progress=False)
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None
    if df.empty or "Close" not in df:
        print(f"No data for {symbol}")
        return None
    closes = df["Close"].dropna()
    if closes.empty:
        return None
    price = float(closes.iloc[-1])

    def ma(n: int) -> float:
        if len(closes) < n:
            return float(closes.mean())
        return float(closes.tail(n).mean())

    ma50 = ma(50)
    ma100 = ma(100)
    ma200 = ma(200)

    dev50 = (price - ma50) / ma50 if ma50 else 0.0
    dev100 = (price - ma100) / ma100 if ma100 else 0.0
    dev200 = (price - ma200) / ma200 if ma200 else 0.0

    return {
        "symbol": symbol,
        "price": price,
        "ma50": ma50,
        "ma100": ma100,
        "ma200": ma200,
        "dev50": dev50,
        "dev100": dev100,
        "dev200": dev200,
    }


def fetch_last_close(symbol: str, period: str = "6mo") -> Optional[float]:
    """获取 yfinance 某个 symbol 最近的收盘价。"""
    try:
        df = yf.download(symbol, period=period, auto_adjust=False, progress=False)
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None
    if df.empty or "Close" not in df:
        return None
    closes = df["Close"].dropna()
    if closes.empty:
        return None
    return float(closes.iloc[-1])


def fetch_fred_latest(series_id: str, api_key: Optional[str]) -> Optional[float]:
    """从 FRED 拉某个时间序列最近的有效值。"""
    if not api_key:
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Error fetching FRED {series_id}: {e}")
        return None
    data = r.json()
    for obs in data.get("observations", []):
        val = obs.get("value")
        try:
            v = float(val)
            return v
        except (TypeError, ValueError):
            continue
    return None


def score_trend(m: Dict[str, float]) -> float:
    """根据价格和 50/100/200 日线位置给一个 0–100 的趋势分。"""
    score = 50.0
    price = m["price"]
    ma50 = m["ma50"]
    ma100 = m["ma100"]
    ma200 = m["ma200"]

    if price > ma50:
        score += 15
    else:
        score -= 15

    if price > ma100:
        score += 10
    else:
        score -= 10

    if price > ma200:
        score += 15
    else:
        score -= 20

    overheat = max(m["dev50"], m["dev100"], m["dev200"])
    if overheat > 0.10:
        score -= 10
    if overheat > 0.20:
        score -= 10

    return clamp(score)


def score_breadth(breadth_pct: Optional[float]) -> Optional[float]:
    """Breadth：多少成份股在 50MA 之上（比如 SPXA50R）。"""
    if breadth_pct is None:
        return None
    b = breadth_pct
    if b <= 20:
        score = 20
    elif b <= 30:
        score = 20 + (b - 20) / 10 * 20
    elif b <= 50:
        score = 40 + (b - 30) / 20 * 20
    elif b <= 80:
        score = 60 + (b - 50) / 30 * 30
    elif b <= 95:
        score = 90 - (b - 80) / 15 * 30
    else:
        score = 50
    return clamp(score)


def score_junk(spread_bp: Optional[float]) -> Optional[float]:
    """Junk OAS（bp）越高，信用风险越大。"""
    if spread_bp is None:
        return None
    s = spread_bp
    if s <= 250:
        score = 90 - (250 - s) / 150 * 20
    elif s <= 350:
        score = 90 - (s - 250) / 100 * 20
    elif s <= 500:
        score = 70 - (s - 350) / 150 * 40
    elif s <= 800:
        score = 30 - (s - 500) / 300 * 20
    else:
        score = 5
    return clamp(score)


def score_sentiment(putcall: Optional[float], vix: Optional[float]) -> Optional[float]:
    """Put/Call + VIX 综合成一个情绪分。"""
    s1 = s2 = None
    if putcall is not None:
        base = 100 - 200 * abs(putcall - 1.0)  # 以 1.0 为平衡点
        s1 = clamp(base)
    if vix is not None:
        v = vix
        if v <= 10:
            score = 40
        elif v <= 15:
            score = 40 + (v - 10) / 5 * 30
        elif v <= 25:
            score = 70 - (v - 15) / 10 * 10
        elif v <= 35:
            score = 60 - (v - 25) / 10 * 20
        else:
            score = 30
        s2 = clamp(score)
    if s1 is not None and s2 is not None:
        return (s1 + s2) / 2
    return s1 if s1 is not None else s2


def score_yield_curve(spread_bp: Optional[float]) -> Optional[float]:
    """10Y–2Y（bp），太深倒挂扣分。"""
    if spread_bp is None:
        return None
    s = spread_bp
    if s >= 100:
        score = 80 + (s - 100) / 100 * 10
    elif s >= 0:
        score = 60 + s / 100 * 20
    elif s >= -50:
        score = 50 + s / -50 * 10
    elif s >= -150:
        score = 40 + (s + 50) / -100 * 20
    else:
        score = 10
    return clamp(score)


def compute_overall(
    trend: Optional[float],
    breadth: Optional[float],
    junk: Optional[float],
    sentiment: Optional[float],
    yc: Optional[float],
):
    """按权重合成总分，并根据分歧对分数进行“收缩”，防止被某一项拖偏。"""
    factors: List[Dict[str, Any]] = []
    if trend is not None:
        factors.append({"label": "trend", "score": trend, "weight": 0.4})
    if breadth is not None:
        factors.append({"label": "breadth", "score": breadth, "weight": 0.2})
    if junk is not None:
        factors.append({"label": "junk", "score": junk, "weight": 0.15})
    if sentiment is not None:
        factors.append({"label": "sentiment", "score": sentiment, "weight": 0.1})
    if yc is not None:
        factors.append({"label": "yc", "score": yc, "weight": 0.15})

    if not factors:
        return None, None, 0

    total_weight = sum(f["weight"] for f in factors)
    raw = sum(f["score"] * f["weight"] for f in factors) / total_weight
    scores_arr = [f["score"] for f in factors]
    sdev = pstdev(scores_arr) if len(scores_arr) > 1 else 0.0

    shrink = 1.0
    if sdev > 25:
        shrink = 0.6
    elif sdev > 15:
        shrink = 0.8

    final = 50 + (raw - 50) * shrink
    return clamp(final), sdev, len(factors)


def judge_risk_flag(
    junk_spread_bp: Optional[float], vix: Optional[float], yc_bp: Optional[float]
):
    """额外给一个“风险状态”标签。"""
    level = "unknown"
    text = "风险状态：数据不足"
    color = "#4b5563"

    high_risk = (
        (junk_spread_bp is not None and junk_spread_bp >= 500)
        or (vix is not None and vix >= 30)
        or (yc_bp is not None and yc_bp <= -100)
    )
    med_risk = (
        (junk_spread_bp is not None and junk_spread_bp >= 400)
        or (vix is not None and vix >= 25)
        or (yc_bp is not None and yc_bp <= -50)
    )
    low_risk = (
        (junk_spread_bp is not None and junk_spread_bp <= 350)
        and (vix is not None and vix <= 22)
        and (yc_bp is not None and yc_bp >= 0)
    )

    if high_risk:
        level = "high"
        text = "风险状态：偏高（信用 / 波动 / 曲线至少一项处于高风险区，需控制仓位与杠杆）"
        color = "#f97373"
    elif med_risk:
        level = "medium"
        text = "风险状态：中等偏上（宏观或情绪有一定紧张，适合保持防守意识）"
        color = "#fbbf24"
    elif low_risk:
        level = "low"
        text = "风险状态：相对温和（信用与波动尚在可控区间，但仍需防黑天鹅）"
        color = "#22c55e"
    else:
        level = "mixed"
        text = "风险状态：混合（部分指标偏紧，部分正常，需结合价格与仓位综合判断）"
        color = "#6b7280"

    return {"level": level, "text": text, "color": color}


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/metrics")
async def get_metrics():
    fred_key = os.getenv("FRED_API_KEY")

    # 价格 & 均线：SPY / QQQ
    spy = fetch_price_metrics("SPY")
    qqq = fetch_price_metrics("QQQ")

    # 情绪相关：VIX / Put-Call / Breadth
    vix = fetch_last_close("^VIX")
    put_call = fetch_last_close("^CPC")  # CBOE Total Put/Call
    breadth_pct = fetch_last_close("^SPXA50R")  # S&P 500 above 50MA, 如果数据源可用

    # FRED：Junk OAS & 10Y/2Y
    junk_oas_pct = fetch_fred_latest("BAMLH0A0HYM2", fred_key) if fred_key else None
    junk_oas_bp = junk_oas_pct * 100 if junk_oas_pct is not None else None

    y10 = fetch_fred_latest("DGS10", fred_key) if fred_key else None
    y2 = fetch_fred_latest("DGS2", fred_key) if fred_key else None
    yc_bp = (y10 - y2) * 100 if (y10 is not None and y2 is not None) else None

    # 趋势分
    trend_scores = []
    if spy is not None:
        trend_scores.append(score_trend(spy))
    if qqq is not None:
        trend_scores.append(score_trend(qqq))
    trend_score = mean(trend_scores) if trend_scores else None

    breadth_score = score_breadth(breadth_pct)
    junk_score = score_junk(junk_oas_bp)
    sentiment_score = score_sentiment(put_call, vix)
    yc_score = score_yield_curve(yc_bp)

    overall_score, sdev, count = compute_overall(
        trend_score, breadth_score, junk_score, sentiment_score, yc_score
    )

    risk = judge_risk_flag(junk_oas_bp, vix, yc_bp)

    return JSONResponse(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw": {
                "spy": spy,
                "qqq": qqq,
                "vix": vix,
                "put_call": put_call,
                "breadth_pct": breadth_pct,
                "junk_oas_pct": junk_oas_pct,
                "junk_oas_bp": junk_oas_bp,
                "y10": y10,
                "y2": y2,
                "yc_spread_bp": yc_bp,
            },
            "scores": {
                "trend": trend_score,
                "breadth": breadth_score,
                "junk": junk_score,
                "sentiment": sentiment_score,
                "yc": yc_score,
                "overall": overall_score,
                "std": sdev,
                "count": count,
            },
            "risk_flag": risk,
        }
    )
