# CNN-LCS Audio Preprocessing Pipeline

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

## Data Setup

### 4. Download and extract data

Download `dev_bearing.zip` and `dev_gearbox.zip` from the Zenodo link.

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

The training script now:

- uses class-balanced sampling (`WeightedRandomSampler`) for training batches
- keeps class-weighted cross-entropy loss to prioritize anomaly recall
- prints the best anomaly-class threshold search result (best threshold, precision, recall, F1)
- prints a threshold `0.50` comparison for easy baseline tracking

## CNN Features + LCS Safe Tuning

### Extract frozen CNN features

```bash
python src/cnn/extract_features.py
```

Default local output:

- `outputs/features/train_features.npy`
- `outputs/features/train_labels.npy`
- `outputs/features/test_features.npy`
- `outputs/features/test_labels.npy`

### Train one LCS candidate with non-regression gates

```bash
python src/lcs/train_lcs.py \
  --k 100 \
  --selector f_classif \
  --learning-iterations 200000 \
  --population-size 3000 \
  --nu 8 \
  --promote-if-pass
```

What this now does:

- tunes decision threshold on a validation split (not on test)
- evaluates test metrics at the validation-selected threshold
- checks hard gates before promotion:
  - accuracy `>= 0.9125`
  - precision `>= 0.4640`
  - recall `>= 0.3225`
  - AUC `>= 0.7347`
  - rule count `<= 3000`
- writes run artifacts under `outputs/lcs/runs/<run_id>/`
- writes `outputs/lcs/approved_model.json` only when all gates pass

### Run the medium-budget sweep (12-20 runs)

```bash
python src/lcs/sweep_lcs.py --budget 16 --promote-best
```

Sweep outputs:

- `outputs/lcs/sweeps/sweep_<timestamp>.csv` (ranked results)
- `outputs/lcs/sweeps/sweep_<timestamp>.json` (summary)
- `outputs/lcs/latest_gate_passing_run.json` (best safe run)

## Web Demo (FastAPI + Next.js)

### Run backend

From project root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

### Run frontend

From `frontend/`:

```bash
npm install
npm run dev
```

Optional frontend environment variable:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Open `http://localhost:3000`, upload a `.wav` clip, and run real anomaly inference through `POST /api/analyze-audio`.

## Expected Final Outputs

- `data/dataset_split.csv`: `2400` rows
- `data/segments_split.csv`: `24000` rows
- `data/spectrograms_split.csv`: `24000` rows
- Spectrogram shape consistency: `(128, 32)` across all files

## Handoff to Person 3

Use:

- `data/spectrograms_split.csv`

This is the training manifest for the CNN stage.
