import logging
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kronos-api")

app = FastAPI(
    title="Kronos Signal API",
    description="Professional crypto signal engine using multi-indicator TA analysis",
    version="2.0.0"
)

# ── Schemas ───────────────────────────────────────────────
class Candle(BaseModel):
    timestamp: str
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    Optional[float] = 0.0

class PredictRequest(BaseModel):
    candles:   List[Candle]
    pred_len:  Optional[int]   = 24
    symbol:    Optional[str]   = "BTCUSDT"
    threshold: Optional[float] = 1.5

class SignalResponse(BaseModel):
    symbol:          str
    signal:          str
    emoji:           str
    current_close:   float
    predicted_close: float
    pct_change:      float
    confidence:      str
    candles_used:    int


# ── Technical Analysis Helpers ────────────────────────────

def ema(values: list, period: int) -> list:
    """Exponential Moving Average."""
    k = 2 / (period + 1)
    result = [None] * len(values)
    for i, v in enumerate(values):
        if i < period - 1:
            continue
        if result[i - 1] is None:
            result[i] = sum(values[i - period + 1:i + 1]) / period
        else:
            result[i] = v * k + result[i - 1] * (1 - k)
    return result

def rsi(closes: list, period: int = 14) -> list:
    """Relative Strength Index."""
    result = [None] * len(closes)
    for i in range(period, len(closes)):
        window = closes[i - period:i + 1]
        gains = [max(window[j] - window[j-1], 0) for j in range(1, len(window))]
        losses = [max(window[j-1] - window[j], 0) for j in range(1, len(window))]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))
    return result

def bollinger_bands(closes: list, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands — returns (upper, mid, lower)."""
    upper, mid, lower = [], [], []
    for i in range(len(closes)):
        if i < period - 1:
            upper.append(None); mid.append(None); lower.append(None)
            continue
        window = closes[i - period + 1:i + 1]
        sma = sum(window) / period
        variance = sum((x - sma) ** 2 for x in window) / period
        sd = math.sqrt(variance)
        mid.append(sma)
        upper.append(sma + std_dev * sd)
        lower.append(sma - std_dev * sd)
    return upper, mid, lower

def linear_forecast(closes: list, periods_ahead: int) -> float:
    """Simple linear regression extrapolation."""
    n = len(closes)
    x_mean = (n - 1) / 2
    y_mean = sum(closes) / n
    num = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0
    intercept = y_mean - slope * x_mean
    return intercept + slope * (n - 1 + periods_ahead)


# ── Health check ──────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "engine": "ta-v2", "model_loaded": True}


# ── Main prediction endpoint ──────────────────────────────
@app.post("/predict", response_model=SignalResponse)
def predict(req: PredictRequest):
    if len(req.candles) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 50 candles, got {len(req.candles)}"
        )

    try:
        closes  = [c.close  for c in req.candles]
        highs   = [c.high   for c in req.candles]
        lows    = [c.low    for c in req.candles]
        volumes = [c.volume for c in req.candles]

        current_close = closes[-1]

        # ── Indicators ────────────────────────────────
        ema8   = ema(closes, 8)
        ema21  = ema(closes, 21)
        ema50  = ema(closes, 50)
        rsi14  = rsi(closes, 14)
        bb_u, bb_m, bb_l = bollinger_bands(closes, 20, 2.0)

        # MACD (EMA12 - EMA26)
        ema12  = ema(closes, 12)
        ema26  = ema(closes, 26)
        macd   = [
            (ema12[i] - ema26[i]) if (ema12[i] and ema26[i]) else None
            for i in range(len(closes))
        ]
        macd_signal = ema([m for m in macd if m is not None], 9)
        # Align macd_signal to macd length
        macd_values = [m for m in macd if m is not None]
        macd_sig_aligned = [None] * len(closes)
        offset = len(closes) - len(macd_values)
        for j, v in enumerate(macd_signal):
            macd_sig_aligned[offset + j] = v

        # Volume trend (last 10 vs prev 10)
        vol_recent = sum(volumes[-10:]) / 10 if len(volumes) >= 20 else 0
        vol_prev   = sum(volumes[-20:-10]) / 10 if len(volumes) >= 20 else vol_recent
        vol_surge  = vol_recent > vol_prev * 1.2 if vol_prev > 0 else False

        # ── Scoring System ────────────────────────────
        score = 0  # range: -6 to +6

        latest_rsi   = rsi14[-1]
        latest_ema8  = ema8[-1]
        latest_ema21 = ema21[-1]
        latest_ema50 = ema50[-1]
        latest_bb_u  = bb_u[-1]
        latest_bb_l  = bb_l[-1]
        latest_bb_m  = bb_m[-1]
        latest_macd  = macd[-1]
        latest_msig  = macd_sig_aligned[-1]

        # EMA trend
        if latest_ema8 and latest_ema21 and latest_ema50:
            if latest_ema8 > latest_ema21 > latest_ema50:
                score += 2   # strong uptrend
            elif latest_ema8 < latest_ema21 < latest_ema50:
                score -= 2   # strong downtrend
            elif latest_ema8 > latest_ema21:
                score += 1
            elif latest_ema8 < latest_ema21:
                score -= 1

        # RSI momentum
        if latest_rsi:
            if latest_rsi < 30:
                score += 2   # oversold → buy pressure
            elif latest_rsi > 70:
                score -= 2   # overbought → sell pressure
            elif latest_rsi < 45:
                score += 1
            elif latest_rsi > 55:
                score -= 1

        # MACD crossover
        if latest_macd and latest_msig:
            if latest_macd > latest_msig:
                score += 1
            else:
                score -= 1

        # Bollinger Band position
        if latest_bb_u and latest_bb_l and latest_bb_m:
            bb_position = (current_close - latest_bb_l) / (latest_bb_u - latest_bb_l)
            if bb_position < 0.2:
                score += 1   # near lower band → potential bounce
            elif bb_position > 0.8:
                score -= 1   # near upper band → potential pullback

        # Volume confirmation
        if vol_surge:
            score = int(score * 1.25)  # amplify signal with volume

        # ── Signal from score ─────────────────────────
        #  score >= 3  → BUY
        #  score <= -3 → SELL
        #  else        → HOLD
        if score >= 3:
            signal, emoji = "BUY", "🟢"
        elif score <= -3:
            signal, emoji = "SELL", "🔴"
        else:
            signal, emoji = "HOLD", "🟡"

        # ── Price forecast (linear extrapolation + mean reversion) ─
        forecast_raw = linear_forecast(closes[-48:], req.pred_len)

        # Blend with simple momentum continuation
        momentum     = (closes[-1] - closes[-6]) / closes[-6]  # 6h momentum
        predicted_close = forecast_raw * (1 + momentum * 0.3)
        predicted_close = round(predicted_close, 2)

        pct_change = round(((predicted_close - current_close) / current_close) * 100, 2)

        # ── Confidence from score magnitude ───────────────
        abs_score = abs(score)
        if abs_score >= 5:
            confidence = "HIGH"
        elif abs_score >= 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        logger.info(
            f"Signal={signal} | Score={score} | RSI={round(latest_rsi,1) if latest_rsi else 'N/A'} "
            f"| MACD={'▲' if (latest_macd and latest_msig and latest_macd > latest_msig) else '▼'} "
            f"| Vol_surge={vol_surge} | Forecast={predicted_close} ({pct_change}%)"
        )

        return SignalResponse(
            symbol=req.symbol,
            signal=signal,
            emoji=emoji,
            current_close=round(current_close, 2),
            predicted_close=predicted_close,
            pct_change=pct_change,
            confidence=confidence,
            candles_used=len(req.candles)
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")
