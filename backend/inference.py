import io
import json
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cnn.model import get_model

from backend.config import (
    ANOMALY_THRESHOLD,
    LCS_APPROVED_MANIFEST,
    LCS_DIR,
    MODEL_CHECKPOINT,
    REQUIRE_APPROVED_LCS,
)


class InferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.feature_extractor: nn.Module | None = None
        self.lcs_model = None
        self.scaler = None
        self.selector = None
        self.lcs_manifest: dict | None = None

    @property
    def has_lcs(self) -> bool:
        return self.lcs_model is not None and self.scaler is not None and self.selector is not None

    def load(self) -> None:
        if not MODEL_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found at {MODEL_CHECKPOINT}. Set MODEL_CHECKPOINT or CHECKPOINT_DIR."
            )

        model = get_model(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.feature_extractor.eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self._try_load_lcs_artifacts()

    def _try_load_lcs_artifacts(self) -> None:
        lcs_model_path = LCS_DIR / "lcs_model.pkl"
        scaler_path = LCS_DIR / "scaler.pkl"
        selector_path = LCS_DIR / "selector.pkl"

        manifest = self._load_approved_manifest()
        if REQUIRE_APPROVED_LCS and not self._is_manifest_approved(manifest):
            return

        if not (lcs_model_path.exists() and scaler_path.exists() and selector_path.exists()):
            return

        with open(lcs_model_path, "rb") as file_obj:
            self.lcs_model = pickle.load(file_obj)
        with open(scaler_path, "rb") as file_obj:
            self.scaler = pickle.load(file_obj)
        with open(selector_path, "rb") as file_obj:
            self.selector = pickle.load(file_obj)
        self.lcs_manifest = manifest

    def _load_approved_manifest(self) -> dict | None:
        if not LCS_APPROVED_MANIFEST.exists():
            return None
        try:
            with open(LCS_APPROVED_MANIFEST, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _is_manifest_approved(manifest: dict | None) -> bool:
        if not manifest:
            return False
        if manifest.get("approved") is not True:
            return False

        gates = manifest.get("gates")
        if isinstance(gates, dict):
            if gates.get("passed") is False:
                return False
            checks = gates.get("checks")
            if isinstance(checks, dict) and not all(bool(value) for value in checks.values()):
                return False
        return True

    def _decode_wav(self, file_bytes: bytes) -> tuple[np.ndarray, int]:
        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        with sf.SoundFile(io.BytesIO(file_bytes)) as sound_file:
            audio = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        if audio.size == 0:
            raise ValueError("Uploaded WAV does not contain usable audio samples.")

        return audio.astype(np.float32), sample_rate

    def _segment_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        segment_samples = sample_rate
        segments: list[np.ndarray] = []

        for start in range(0, len(audio), segment_samples):
            segment = audio[start : start + segment_samples]
            if len(segment) < segment_samples:
                if len(segment) < segment_samples // 2 and segments:
                    continue
                segment = np.pad(segment, (0, segment_samples - len(segment)))
            segments.append(segment)

        if not segments:
            raise ValueError("No valid 1-second segments could be generated from this audio.")

        return np.stack(segments, axis=0)

    def _segments_to_tensor(self, segments: np.ndarray) -> torch.Tensor:
        mel_segments = []
        for segment in segments:
            mel = librosa.feature.melspectrogram(
                y=segment,
                sr=16000,
                n_mels=128,
                n_fft=1024,
                hop_length=512,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            mel_segments.append(mel_db)

        mel_array = np.stack(mel_segments, axis=0)
        specs = torch.from_numpy(mel_array).unsqueeze(1).repeat(1, 3, 1, 1)
        return specs.float()

    def _summarize_notes(
        self,
        probability: float,
        inference_source: str,
        cnn_probability: float,
        lcs_probability: float | None,
    ) -> str:
        if probability >= 0.8:
            base = "Strong anomaly signal detected in the engine profile."
        elif probability >= 0.6:
            base = "Moderate anomaly indications found; inspection recommended."
        elif probability >= 0.4:
            base = "Borderline signal; monitor and capture another sample if possible."
        else:
            base = "Mostly normal acoustic signature based on learned patterns."

        if inference_source == "cnn_lcs" and lcs_probability is not None:
            return (
                f"{base} Combined CNN+LCS scoring used "
                f"(cnn={cnn_probability:.2f}, lcs={lcs_probability:.2f})."
            )
        return f"{base} CNN scoring used (cnn={cnn_probability:.2f})."

    def predict(self, file_bytes: bytes, machine_type: str = "unknown") -> dict:
        if self.model is None:
            raise RuntimeError("Inference service is not loaded.")

        audio, sample_rate = self._decode_wav(file_bytes)
        segments = self._segment_audio(audio, sample_rate)
        specs = self._segments_to_tensor(segments).to(self.device)

        with torch.no_grad():
            logits = self.model(specs)
            cnn_scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        cnn_probability = float(np.mean(cnn_scores))
        max_segment_probability = float(np.max(cnn_scores))
        lcs_probability = None
        inference_source = "cnn"
        anomaly_probability = cnn_probability

        if self.has_lcs and self.feature_extractor is not None:
            with torch.no_grad():
                features = self.feature_extractor(specs)
                features = features.squeeze(-1).squeeze(-1).cpu().numpy()

            scaled = self.scaler.transform(features)
            selected = self.selector.transform(scaled)
            lcs_scores = self.lcs_model.predict_proba(selected)[:, 1]
            lcs_probability = float(np.mean(lcs_scores))
            anomaly_probability = lcs_probability
            inference_source = "cnn_lcs"

        label = "Anomalous" if anomaly_probability >= ANOMALY_THRESHOLD else "Normal"
        notes = self._summarize_notes(
            probability=anomaly_probability,
            inference_source=inference_source,
            cnn_probability=cnn_probability,
            lcs_probability=lcs_probability,
        )

        response = {
            "status": "ok",
            "anomaly_probability": round(anomaly_probability, 4),
            "label": label,
            "machine_type": machine_type,
            "notes": notes,
            "segments_analyzed": int(len(segments)),
            "threshold": ANOMALY_THRESHOLD,
            "inference_source": inference_source,
            "cnn_probability": round(cnn_probability, 4),
            "max_segment_probability": round(max_segment_probability, 4),
        }

        if lcs_probability is not None:
            response["lcs_probability"] = round(lcs_probability, 4)
        if self.lcs_manifest is not None:
            response["lcs_run_id"] = self.lcs_manifest.get("run_id")

        return response
