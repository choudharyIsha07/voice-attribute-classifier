import time
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, WebSocket, WebSocketDisconnect
from ..schemas import AnalyzeResponse, Prediction
from ..config import settings
from ..services.audio import validate_audio_file
from ..services.quality import assess_audio_quality
from ..services.inference import InferenceProvider, get_inference_provider
from ..services.streaming import StreamingBuffer

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health", summary="Health check", tags=["ops"])
async def health_check():
    """Returns 200 OK when the service is up."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /analyze — multipart audio upload
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze audio for gender and age bracket",
    tags=["inference"],
)
async def analyze_audio(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/flac/ogg/m4a). Max 10 MB."),
    contact_id: Optional[str] = Form(None, description="Optional UUID. Auto-generated if omitted."),
    inference_provider: InferenceProvider = Depends(get_inference_provider),
):
    """
    Accept a multipart audio upload and return:
    - **gender** prediction with confidence
    - **age_bracket** prediction with confidence
    - **audio_quality** flag
    - **language** best-effort ISO-639-1 code (bonus)
    - **processing_ms** end-to-end latency

    Audio is never persisted to disk. All processing is in-memory.
    """
    start_time = time.monotonic()

    # Parse contact_id — accept string from Form or None
    parsed_contact_id: uuid.UUID
    if contact_id is None:
        parsed_contact_id = uuid.uuid4()
    else:
        try:
            parsed_contact_id = uuid.UUID(str(contact_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid contact_id: must be a valid UUID")

    logger.info(
        f"Analyze request started: contact_id={parsed_contact_id}, "
        f"filename={audio.filename}, content_type={audio.content_type}"
    )

    # 1. Validate and parse audio (in memory — no disk writes)
    try:
        samples, sample_rate = await validate_audio_file(audio, settings.max_upload_size)
    except HTTPException as e:
        logger.warning(f"Validation failure: contact_id={parsed_contact_id}, detail={e.detail}")
        raise
    except Exception as e:
        logger.error(f"Internal error reading file: contact_id={parsed_contact_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # 2. Assess audio quality
    quality = assess_audio_quality(samples, sample_rate)

    # 3. Run inference
    gender_pred, gender_conf, age_pred, age_conf, language = \
        inference_provider.infer_attributes(samples, sample_rate)

    processing_ms = int((time.monotonic() - start_time) * 1000)

    response = AnalyzeResponse(
        contact_id=parsed_contact_id,
        gender=Prediction(prediction=gender_pred, confidence=gender_conf),
        age_bracket=Prediction(prediction=age_pred, confidence=age_conf),
        processing_ms=processing_ms,
        audio_quality=quality,
        language=language,
    )

    logger.info(
        f"Analyze complete: contact_id={parsed_contact_id}, duration_ms={processing_ms}, "
        f"quality={quality}, gender={gender_pred}({gender_conf:.2f}), "
        f"age={age_pred}({age_conf:.2f}), lang={language}"
    )
    return response


# ---------------------------------------------------------------------------
# WebSocket /ws/analyze — real-time streaming (bonus)
# ---------------------------------------------------------------------------

@router.websocket("/ws/analyze")
async def ws_analyze(
    websocket: WebSocket,
    inference_provider: InferenceProvider = Depends(get_inference_provider),
):
    """
    Real-time WebSocket endpoint for streaming audio inference.

    Protocol
    --------
    1. Client connects.
    2. Client sends audio in binary frames (raw 16-bit LE PCM, 16 kHz mono).
       Alternatively, send JSON text frame {"type": "end"} to signal end of stream.
    3. Server emits progressive JSON results every ~2 seconds of accumulated audio.
    4. On stream end (or "end" frame), server sends a final result with is_final=true.

    Example result frame::

        {
          "chunk_index": 1,
          "gender": {"prediction": "male", "confidence": 0.81},
          "age_bracket": {"prediction": "31-45", "confidence": 0.54},
          "audio_quality": "good",
          "language": "en",
          "is_final": false
        }
    """
    await websocket.accept()
    ws_id = str(uuid.uuid4())[:8]
    logger.info(f"WebSocket connected: ws_id={ws_id}")

    buffer = StreamingBuffer(inference_provider)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.receive":
                # Binary frame: raw PCM bytes
                if message.get("bytes"):
                    result = buffer.push(message["bytes"])
                    if result:
                        await websocket.send_json(result)

                # Text frame: control message
                elif message.get("text"):
                    import json
                    try:
                        ctrl = json.loads(message["text"])
                        if ctrl.get("type") == "end":
                            final_result = buffer.finalize()
                            await websocket.send_json(final_result)
                            await websocket.close()
                            break
                    except Exception:
                        pass  # Ignore malformed control frames

            elif message["type"] == "websocket.disconnect":
                # Client disconnected — emit final result
                final_result = buffer.finalize()
                try:
                    await websocket.send_json(final_result)
                except Exception:
                    pass
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: ws_id={ws_id}")
    except Exception as exc:
        logger.error(f"WebSocket error: ws_id={ws_id}, error={exc}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        logger.info(f"WebSocket session ended: ws_id={ws_id}")
