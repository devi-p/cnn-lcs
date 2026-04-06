# Backend (FastAPI)

This backend serves real audio anomaly inference for the cnn-lcs project.

## What it does

- Loads the existing CNN checkpoint (EfficientNet-B0) once at startup.
- Accepts `.wav` uploads at `POST /api/analyze-audio`.
- Applies project-consistent preprocessing (`16kHz`, `1s segments`, `128-bin log-mel`).
- Returns anomaly probability plus label.
- Optionally enriches inference with LCS artifacts (`lcs_model.pkl`, `scaler.pkl`, `selector.pkl`) when available.
- By default, LCS loading is gated by an approval manifest so unvalidated runs do not override CNN inference.

## Install

From project root:

```bash
pip install -r backend/requirements.txt
```

## Run

From project root:

```bash
uvicorn backend.main:app --reload
```

## Endpoint

- `GET /api/health`
- `POST /api/analyze-audio`
  - `file`: WAV file (multipart/form-data)
  - `machine_type`: `bearing` | `gearbox` | `unknown`

Example response:

```json
{
  "status": "ok",
  "anomaly_probability": 0.73,
  "label": "Anomalous",
  "machine_type": "bearing",
  "notes": "Combined CNN+LCS scoring used (cnn=0.68, lcs=0.73)."
}
```

## Environment Variables

- `MODEL_CHECKPOINT`: full path to a checkpoint file (`.pth`)
- `CHECKPOINT_DIR`: directory containing `best_model.pth` (used when `MODEL_CHECKPOINT` is unset)
- `LCS_DIR`: directory containing `lcs_model.pkl`, `scaler.pkl`, `selector.pkl`
- `LCS_APPROVED_MANIFEST`: path to approval manifest (default `LCS_DIR/approved_model.json`)
- `REQUIRE_APPROVED_LCS`: require approved manifest before loading LCS (`true` by default)
- `ANOMALY_THRESHOLD`: decision threshold (default `0.50`)
- `API_ALLOW_ORIGINS`: comma-separated CORS origins (default `*`)
- `MAX_UPLOAD_MB`: upload size cap in MB (default `25`)

Defaults:

- Local checkpoint: `outputs/checkpoints/best_model.pth`
- Colab checkpoint: `/content/drive/MyDrive/cnn-lcs/checkpoints/best_model.pth`
- Local LCS dir: `outputs/lcs`
- Colab LCS dir: `/content/drive/MyDrive/cnn-lcs/lcs`
