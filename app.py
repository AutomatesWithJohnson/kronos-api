import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kronos-api")

app = FastAPI(
    title="Kronos AI Signal API",
    description="Crypto price forecasting powered by Kronos foundation model",
    version="1.0.0"
)

# ── Lazy model loading — load once on first request ──────
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        logger.info("Loading Kronos model from HuggingFace (first-time download may take 2-5 mins)...")
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor
            tokenizer  = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            model      = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
            _predictor = KronosPredictor(model, tokenizer, max_context=512)
            logger.info("✅ Kronos model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load Kronos model: {e}")
            raise RuntimeError(f"Model load failed: {e}")
    return _predictor


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


# ── Health check (always available — keeps Render awake) ──
@app.get("/health")
def health():
    model_ready = _predictor is not None
    return {"status": "ok", "model_loaded": model_ready}


# ── Main prediction endpoint ──────────────────────────────
@app.post("/predict", response_model=SignalResponse)
def predict(req: PredictRequest):
    if len(req.candles) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 50 candles, got {len(req.candles)}"
        )

    try:
        predictor = get_predictor()

        # Build DataFrame from request
        records = [c.dict() for c in req.candles]
        df_all  = pd.DataFrame(records)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

        x_df = df_all[["open", "high", "low", "close", "volume"]]
        x_ts = df_all["timestamp"]

        # Generate future timestamps (hourly)
        last_ts = df_all["timestamp"].iloc[-1]
        y_ts = pd.Series([
            last_ts + pd.Timedelta(hours=i + 1)
            for i in range(req.pred_len)
        ])

        # Run Kronos prediction (sample_count=3 for stability)
        logger.info(f"Running prediction for {req.symbol} with {len(req.candles)} candles...")
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=req.pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=3
        )

        current_close   = float(df_all["close"].iloc[-1])
        predicted_close = float(pred_df["close"].iloc[-1])
        pct_change      = ((predicted_close - current_close) / current_close) * 100
        abs_pct         = abs(pct_change)

        # Signal logic
        if pct_change >= req.threshold:
            signal, emoji = "BUY", "🟢"
        elif pct_change <= -req.threshold:
            signal, emoji = "SELL", "🔴"
        else:
            signal, emoji = "HOLD", "🟡"

        # Confidence level based on how far beyond the threshold
        if abs_pct >= req.threshold * 3.0:
            confidence = "HIGH"
        elif abs_pct >= req.threshold * 1.8:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        logger.info(f"Signal: {emoji} {signal} | Change: {round(pct_change, 2)}% | Confidence: {confidence}")

        return SignalResponse(
            symbol=req.symbol,
            signal=signal,
            emoji=emoji,
            current_close=round(current_close, 2),
            predicted_close=round(predicted_close, 2),
            pct_change=round(pct_change, 2),
            confidence=confidence,
            candles_used=len(req.candles)
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
