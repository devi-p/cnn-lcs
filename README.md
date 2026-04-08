# CNN-LCS Engine Sound Anomaly Detection

This project combines a CNN-based acoustic classifier with an LCS-based rule layer for machine sound anomaly detection. It includes:

- data preprocessing scripts (audio segmentation + mel-spectrogram generation)
- CNN training and evaluation utilities
- LCS artifacts used in inference
- a FastAPI backend for audio analysis
- a Next.js frontend for upload and visualization
- Docker deployment files for a single-port Hugging Face Space setup

## Setup Instructions

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd cnn-lcs
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
```

Mac/Linux:

```bash
source venv/bin/activate
```

Windows (PowerShell):

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install librosa soundfile scikit-learn pandas matplotlib numpy torch torchvision
```

## Run Web App Locally (Backend + Frontend)

Open two terminals from the repo root (`cnn-lcs`).

### Terminal 1: Backend (FastAPI)

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Backend endpoints:

- `http://localhost:8000/api/health`
- `http://localhost:8000/api/analyze-audio`

### Terminal 2: Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Open:

- `http://localhost:3000`

Notes:

- The frontend proxies `/api/*` to the backend at `http://localhost:8000` by default in local development.
- If you change backend port/url, set `BACKEND_URL` before running `npm run dev`.

## Data Setup

### 4. Download and extract data

Download `dev_bearing.zip` and `dev_gearbox.zip` from your project dataset source.

Place and extract them into:

- `data/raw/bearing/`
- `data/raw/gearbox/`

## Run Preprocessing Pipeline

### 5. Build clip-level split

```bash
python notebooks/01_explore_data.py
```

Generates:

- `data/dataset_split.csv` (80/20 stratified split of all 2400 clips)

### 6. Segment audio into 1-second clips

```bash
python src/preprocessing/segment_audio.py
```

Generates:

- `data/segments/` (24,000 one-second WAV segments)
- `data/segments_split.csv` (segment paths with labels and splits)

### 7. Generate Mel-Spectrogram `.npy` files

```bash
python src/preprocessing/generate_spectrograms.py
```

Uses:

- sample rate: `16000`
- mel bins: `128`
- FFT window: `1024`
- hop length: `512`
- log scale via decibel conversion

Generates:

- `data/spectrograms/` (mirrors `data/segments/` folder structure)
- `data/spectrograms_split.csv` with columns:
  - `spectrogram_path`
  - `label`
  - `split`
  - `machine_type`

## Sanity Checks

### 8. Audio sanity check

```bash
python notebooks/02_sanity_check.py
```

Generates:

- `outputs/bearing_sanity_check.png`
- `outputs/gearbox_sanity_check.png`

### 9. Spectrogram sanity check

```bash
python notebooks/03_spectrogram_sanity_check.py
```

Generates:

- `outputs/bearing_spectrogram_sanity_check.png`
- `outputs/gearbox_spectrogram_sanity_check.png`

## CNN Training

Run training (local or Colab):

```bash
PYTHONPATH=. python3 src/cnn/train.py
```

The training script:

- uses class-balanced sampling (`WeightedRandomSampler`) for training batches
- keeps class-weighted cross-entropy loss to prioritize anomaly recall
- prints the best anomaly-class threshold search result (best threshold, precision, recall, F1)
- prints a threshold `0.50` comparison for easy baseline tracking

## Expected Core Outputs

- `data/dataset_split.csv`: `2400` rows
- `data/segments_split.csv`: `24000` rows
- `data/spectrograms_split.csv`: `24000` rows
- spectrogram shape consistency: `(128, 32)` across all files

## Docker Run (Single Container)

Build and run:

```bash
docker build -t cnn-lcs-hf .
docker run --rm -p 7860:7860 cnn-lcs-hf
```

Service routes in the container:

- frontend: `http://localhost:7860/`
- backend health: `http://localhost:7860/api/health`

## Repository Map

- `backend/`: FastAPI app, config, inference pipeline
- `frontend/`: Next.js UI and components
- `src/`: training and preprocessing modules (`cnn`, `lcs`, `preprocessing`)
- `data/`: split manifests
- `outputs/`: model checkpoints and LCS artifacts used for inference
- `notebooks/`: preprocessing and sanity-check scripts
