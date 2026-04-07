from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_ALLOW_ORIGINS, MAX_UPLOAD_MB, MODEL_CHECKPOINT
from backend.inference import InferenceService

service = InferenceService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    lcs_status = service.get_lcs_status()
    lcs_state = "enabled" if lcs_status["enabled"] else "disabled"
    print(f"Loaded checkpoint: {MODEL_CHECKPOINT}")
    print(f"Optional LCS inference: {lcs_state}")
    print(f"LCS detail: {lcs_status['detail']}")
    yield


app = FastAPI(title="CNN-LCS Inference API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    lcs_status = service.get_lcs_status()
    return {
        "status": "ok",
        "service": "cnn-lcs-backend",
        "lcs_enabled": lcs_status["enabled"],
        "lcs_detail": lcs_status["detail"],
        "lcs_artifact_dir": lcs_status["artifact_dir"],
        "lcs_run_id": lcs_status["run_id"],
        "model_checkpoint": str(MODEL_CHECKPOINT),
    }


@app.post("/api/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    machine_type: str = Form("bearing"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was provided.")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Limit is {MAX_UPLOAD_MB}MB.",
        )

    safe_machine_type = machine_type.lower().strip()
    if safe_machine_type not in {"bearing", "gearbox", "unknown"}:
        safe_machine_type = "unknown"

    try:
        result = service.predict(content, machine_type=safe_machine_type)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error

    return result
