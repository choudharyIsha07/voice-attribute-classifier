import time
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from ..schemas import AnalyzeResponse, Prediction
from ..config import settings
from ..services.audio import validate_audio_file
from ..services.quality import assess_audio_quality
from ..services.inference import InferenceProvider, get_inference_provider

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    audio: UploadFile = File(...),
    contact_id: uuid.UUID = None,
    inference_provider: InferenceProvider = Depends(get_inference_provider)
):
    start_time = time.monotonic()
    
    if contact_id is None:
        contact_id = uuid.uuid4()
        
    logger.info(f"Analyze request started: contact_id={contact_id}, filename={audio.filename}")
    
    # 1. Validate and parse audio (in memory)
    try:
        samples, sample_rate = await validate_audio_file(audio, settings.max_upload_size)
    except HTTPException as e:
        logger.warning(f"Validation failure: contact_id={contact_id}, detail={e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Internal error reading file: contact_id={contact_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
        
    # 2. Assess Quality
    quality = assess_audio_quality(samples, sample_rate)
    
    # 3. Inference
    gender_pred, gender_conf, age_pred, age_conf = inference_provider.infer_attributes(samples, sample_rate)
    
    processing_ms = int((time.monotonic() - start_time) * 1000)
    
    response = AnalyzeResponse(
        contact_id=contact_id,
        gender=Prediction(prediction=gender_pred, confidence=gender_conf),
        age_bracket=Prediction(prediction=age_pred, confidence=age_conf),
        processing_ms=processing_ms,
        audio_quality=quality
    )
    
    logger.info(f"Analyze request completed: contact_id={contact_id}, duration_ms={processing_ms}, quality={quality}")
    return response
