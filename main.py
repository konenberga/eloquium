"""FastAPI TTS service.

POST /tts  {text, language}  -> audio/wav
GET  /health                 -> {status, device}
"""
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from model import TTSEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Holds the single shared engine instance, loaded once at startup.
state: dict[str, TTSEngine] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing TTS engine...")
    state["engine"] = TTSEngine()
    yield
    state.clear()


app = FastAPI(title="Eloquium TTS", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: Literal["en", "ru"] = "en"


@app.get("/health")
def health():
    engine = state.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    return {"status": "ok", "device": engine.device}


@app.post("/tts")
def tts(req: TTSRequest):
    engine = state.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    try:
        audio = engine.synthesize(req.text, req.language)
    except Exception as exc:  # surface synthesis failures as 500
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=audio, media_type="audio/wav")
