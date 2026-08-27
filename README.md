# Voice Attribute Classifier

## 1. Project Overview
The Voice Attribute Classifier is a production-ready backend API designed to securely process uploaded audio files, assess audio quality, and provide inference hooks for voice-based demographic predictions (gender and age bracket). This project was built as a backend engineering assignment, demonstrating clean architecture, strict memory constraints, and robust API design using FastAPI.

## 2. Architecture
The application is structured using a clean, modular design:
- **`app/main.py`**: Entry point handling FastAPI application setup and global exception handlers.
- **`app/api/routes.py`**: Presentation layer defining REST endpoints (`/health`, `/analyze`) and dependency injection.
- **`app/services/`**: Core business logic.
  - `audio.py`: In-memory audio validation and parsing using `pydub` and `numpy`.
  - `quality.py`: Objective audio quality assessment algorithms.
  - `inference.py`: Abstraction layer for demographic inference models.
- **`app/schemas.py`**: Pydantic models enforcing strict input validation and JSON serialization.
- **`app/config.py`**: Environment-based configuration powered by `pydantic-settings`.
- **`app/logging_config.py`**: Centralized structured logging setup.

## 3. Request Flow
1. **Client** submits a `multipart/form-data` request with an `audio` file.
2. **FastAPI Router** intercepts the request and injects the required `InferenceProvider`.
3. **Audio Service** reads the file directly into a byte stream, verifies constraints (size, MIME type), decodes it via `pydub`, and normalizes it into a NumPy float32 array (mono-channel) without ever writing to the disk.
4. **Quality Service** evaluates the NumPy array for silence, volume (RMS), and clipping, returning a quality string.
5. **Inference Service** (Mocked) receives the array and returns safety-first unknown predictions.
6. **Router** records the processing duration via a monotonic timer, compiles the JSON payload, logs success, and returns the response.

## 4. API Contract

### `POST /analyze`
**Request:**
- `Content-Type: multipart/form-data`
- `audio`: The audio file to analyze (e.g., .wav, .mp3, .m4a). Max size: 10MB.
- `contact_id` (Optional): A UUID string. Auto-generated if omitted.

**Response (200 OK):**
```json
{
  "contact_id": "uuid",
  "gender": {
    "prediction": "unknown",
    "confidence": 0.0
  },
  "age_bracket": {
    "prediction": "unknown",
    "confidence": 0.0
  },
  "processing_ms": 12,
  "audio_quality": "good"
}
```

### `GET /health`
**Response (200 OK):**
```json
{
  "status": "ok"
}
```

## 5. Local Setup
Requirements: Python 3.11/3.12 and `ffmpeg` installed on your host system.
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Unix/macOS: source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## 6. Docker Setup (Recommended)
The Docker environment provides an isolated, production-like runtime running securely as a non-root user.
```bash
docker compose up --build
```
The API will be available at `http://localhost:8000`.

## 7. Postman Testing Instructions
1. Open Postman and create a new **POST** request to `http://localhost:8000/analyze`.
2. Navigate to the **Body** tab and select `form-data`.
3. Create a key named `audio`. Change its type from `Text` to `File` (by hovering over the key input field).
4. Select a local audio file (e.g., a `.wav` or `.mp3` file).
5. (Optional) Create a key named `contact_id` with a valid UUID.
6. Click **Send** and verify the JSON response.

## 8. Running pytest
The project includes a robust test suite covering end-to-end API integrations, edge cases, and synthetic audio generation (to avoid committing real human audio files).
```bash
# Run tests locally
python -m pytest tests/
```

## 9. Audio Quality Methodology
The `assess_audio_quality` function evaluates the raw NumPy samples:
- **Insufficient**: The audio is shorter than 1.0 seconds, or $>90\%$ of the 20ms frames have an RMS (volume) below `0.005` (mostly silence).
- **Degraded**: The audio is shorter than 2.0 seconds, overly quiet (Overall RMS $< 0.01$), extremely loud/distorted (Overall RMS $> 0.5$), or suffers from severe clipping ($>5\%$ of samples hit the ceiling).
- **Good**: Audio falls within standard human speech parameters.

## 10. Privacy Design
Audio privacy is a first-class citizen in this architecture:
- **No Disk Persistence**: Uploaded multipart files are read directly into an `io.BytesIO` memory stream. No raw bytes or temporary WAV files are ever committed to the project directory or persistent storage.
- **No PII Logging**: Logs only track metadata such as UUIDs, filenames, processing times, and quality statuses. Raw audio data, transcripts, or sensitive inference predictions are strictly excluded from logging.

## 11. Logging & Observability
Standard Python `logging` is configured for structured output. The system logs:
- Request initiation (with `contact_id`).
- Validation failures (4xx errors).
- Internal processing errors.
- Successful completions alongside processing duration and quality metrics.

## 12. Latency Considerations
- Processing is purely CPU-bound (NumPy operations).
- In-memory decoding (bypassing disk I/O) significantly reduces standard latency overhead.
- Total processing time is measured reliably using `time.monotonic()` to prevent clock-drift issues.

## 13. Error Handling
- **400 Bad Request**: Handled deliberately for unsupported file types, empty files, or files exceeding the 10MB limit.
- **500 Internal Server Error**: A global `Exception` handler intercepts all unhandled errors. It securely logs the traceback server-side and returns a sterile `{"detail": "Internal server error"}` to the client, preventing stack-trace leaks.

## 14. Known Limitations
- Pure Python multiprocessing/async does not speed up single-request NumPy/pydub operations due to the GIL. High concurrency requires multiple worker processes.
- Memory usage scales linearly with upload size. A 10MB WAV file inflates in memory when decoded into a raw float32 NumPy array.

## 15. Inference Provider Abstraction
**Note**: This API *does not* implement real demographic inference from human voices due to ethical and privacy concerns.
Instead, inference logic is securely isolated behind an `InferenceProvider` abstract base class. FastAPI's Dependency Injection (`Depends`) dynamically injects a `MockInferenceProvider` that explicitly returns `"unknown"` and `0.0` confidence. If an approved, ethically compliant AI model is provided in the future, it can be swapped in seamlessly by changing the injected provider without touching the router logic.

## 16. Real-time WebSocket Future Work
To support real-time audio streaming (e.g., from a browser microphone):
- Create a `WebSocket` endpoint in FastAPI.
- Stream chunks of WebM/PCM audio.
- Accumulate chunks into a rolling buffer (e.g., `collections.deque`).
- Run the inference/quality engine on sliding windows (e.g., every 1 second of audio) asynchronously.

## 17. Scaling Strategy for 1,000 Concurrent Calls
If the API must handle 1,000 concurrent uploads:
1. **Web Workers**: Deploy `gunicorn` with `uvicorn` workers (e.g., `gunicorn -k uvicorn.workers.UvicornWorker -w 8`) to utilize multi-core CPUs.
2. **Horizontal Scaling**: Containerize the app and deploy it on a Kubernetes cluster with a Horizontal Pod Autoscaler (HPA) targeting CPU utilization.
3. **Decouple Processing**: Offload heavy inference tasks to a message queue (e.g., Celery/RabbitMQ). The FastAPI endpoint would immediately return a `202 Accepted` with a job ID, allowing the client to poll for results.
4. **Memory Limits**: strictly enforce the 10MB limit at an API Gateway / Ingress level (e.g., NGINX) to prevent malicious Out-Of-Memory (OOM) crashes.

## 18. Evaluation & Future Improvements
- Implement a distributed tracing system (like OpenTelemetry) for cross-service latency tracking.
- Refactor standard logging to use true JSON-structured logs (e.g., `structlog`) to enable easy ingestion by Datadog or ELK stacks.
- Swap `pydub` for a native C-binding audio library (like `soundfile`) if performance bottlenecks arise, eliminating the `ffmpeg` dependency entirely.
