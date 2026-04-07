import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _colab_drive_root() -> Path | None:
    drive_path = Path("/content/drive/MyDrive/cnn-lcs")
    return drive_path if drive_path.exists() else None


def _default_checkpoint_dir() -> Path:
    colab_root = _colab_drive_root()
    if colab_root:
        return colab_root / "checkpoints"
    return PROJECT_ROOT / "outputs" / "checkpoints"


def _default_lcs_dir() -> Path:
    colab_root = _colab_drive_root()
    if colab_root:
        return colab_root / "lcs"
    return PROJECT_ROOT / "outputs" / "lcs"


def _as_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(_default_checkpoint_dir())))
MODEL_CHECKPOINT = Path(
    os.getenv("MODEL_CHECKPOINT", str(CHECKPOINT_DIR / "best_model.pth"))
)
LCS_DIR = Path(os.getenv("LCS_DIR", str(_default_lcs_dir())))
LCS_APPROVED_MANIFEST = Path(
    os.getenv("LCS_APPROVED_MANIFEST", str(LCS_DIR / "approved_model.json"))
)
REQUIRE_APPROVED_LCS = _as_bool(os.getenv("REQUIRE_APPROVED_LCS", "true"), True)
ANOMALY_THRESHOLD = _as_float(os.getenv("ANOMALY_THRESHOLD", "0.50"), 0.50)
API_ALLOW_ORIGINS = os.getenv("API_ALLOW_ORIGINS", "*").split(",")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
