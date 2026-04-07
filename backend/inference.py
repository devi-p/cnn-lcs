import ast
import csv
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
        self.lcs_artifact_dir: Path | None = None
        self.lcs_rules: list[dict] = []
        self.lcs_feature_aliases: dict[int, dict] = {}
        self.lcs_alias_disclaimer = (
            "Feature aliases are correlation-based proxies from offline analysis and are not definitive causal explanations."
        )
        self.lcs_status_detail: str = "LCS not loaded yet."

    @property
    def has_lcs(self) -> bool:
        return self.lcs_model is not None and self.scaler is not None and self.selector is not None

    def get_lcs_status(self) -> dict:
        return {
            "enabled": self.has_lcs,
            "detail": self.lcs_status_detail,
            "artifact_dir": str(self.lcs_artifact_dir) if self.lcs_artifact_dir else None,
            "manifest_path": str(LCS_APPROVED_MANIFEST),
            "require_approved": REQUIRE_APPROVED_LCS,
            "run_id": self.lcs_manifest.get("run_id") if self.lcs_manifest else None,
            "alias_map_loaded": bool(self.lcs_feature_aliases),
        }

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
        self.lcs_model = None
        self.scaler = None
        self.selector = None
        self.lcs_manifest = None
        self.lcs_artifact_dir = None
        self.lcs_rules = []
        self.lcs_feature_aliases = {}

        manifest = self._load_approved_manifest()
        manifest_is_approved = self._is_manifest_approved(manifest)
        if REQUIRE_APPROVED_LCS and not manifest_is_approved:
            self.lcs_status_detail = (
                f"LCS disabled: approval manifest is required at {LCS_APPROVED_MANIFEST} "
                f"and must include approved=true."
            )
            return

        candidate_dirs = self._candidate_lcs_dirs(manifest)
        selected_dir = self._find_valid_lcs_artifact_dir(candidate_dirs)
        if selected_dir is None:
            checked_paths = ", ".join(str(path) for path in candidate_dirs)
            self.lcs_status_detail = (
                "LCS disabled: could not find lcs_model.pkl, scaler.pkl, selector.pkl in "
                f"[{checked_paths}]"
            )
            return

        lcs_model_path = selected_dir / "lcs_model.pkl"
        scaler_path = selected_dir / "scaler.pkl"
        selector_path = selected_dir / "selector.pkl"

        try:
            with open(lcs_model_path, "rb") as file_obj:
                self.lcs_model = pickle.load(file_obj)
            with open(scaler_path, "rb") as file_obj:
                self.scaler = pickle.load(file_obj)
            with open(selector_path, "rb") as file_obj:
                self.selector = pickle.load(file_obj)
        except Exception as error:
            self.lcs_model = None
            self.scaler = None
            self.selector = None
            self.lcs_status_detail = f"LCS disabled: failed loading artifacts from {selected_dir}: {error}"
            return

        self.lcs_manifest = manifest if isinstance(manifest, dict) else None
        self.lcs_artifact_dir = selected_dir
        loaded_rules = self._load_lcs_rules(selected_dir)
        loaded_aliases = self._load_lcs_feature_aliases(selected_dir)
        run_id = self.lcs_manifest.get("run_id") if isinstance(self.lcs_manifest, dict) else None
        if run_id:
            self.lcs_status_detail = (
                f"LCS enabled from {selected_dir} "
                f"(run_id={run_id}, rules_loaded={loaded_rules}, aliases_loaded={loaded_aliases})."
            )
        else:
            self.lcs_status_detail = (
                f"LCS enabled from {selected_dir} "
                f"(rules_loaded={loaded_rules}, aliases_loaded={loaded_aliases})."
            )

    def _candidate_lcs_dirs(self, manifest: dict | None) -> list[Path]:
        candidates: list[Path] = []

        artifact_dir = manifest.get("artifact_dir") if isinstance(manifest, dict) else None
        if isinstance(artifact_dir, str) and artifact_dir.strip():
            candidates.append(Path(artifact_dir).expanduser())

        candidates.append(LCS_DIR)

        runs_dir = LCS_DIR / "runs"
        if runs_dir.exists() and runs_dir.is_dir():
            run_folders = sorted(
                [entry for entry in runs_dir.iterdir() if entry.is_dir()],
                key=lambda item: item.name,
                reverse=True,
            )
            candidates.extend(run_folders)

        unique_candidates: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            normalized = str(path.resolve()) if path.exists() else str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_candidates.append(path)

        return unique_candidates

    @staticmethod
    def _find_valid_lcs_artifact_dir(candidate_dirs: list[Path]) -> Path | None:
        for directory in candidate_dirs:
            if not directory.exists() or not directory.is_dir():
                continue
            required = [directory / "lcs_model.pkl", directory / "scaler.pkl", directory / "selector.pkl"]
            if all(path.exists() for path in required):
                return directory
        return None

    @staticmethod
    def _parse_literal_list(value: str | None) -> list | None:
        if value is None:
            return None
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return None
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return None

    @staticmethod
    def _to_float(value: str | None, default: float = 0.0) -> float:
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: str | None, default: int = 0) -> int:
        try:
            return int(float(value)) if value is not None else default
        except (TypeError, ValueError):
            return default

    def _load_lcs_rules(self, artifact_dir: Path) -> int:
        rules_path = artifact_dir / "rules.csv"
        if not rules_path.exists():
            self.lcs_rules = []
            return 0

        parsed_rules: list[dict] = []
        with open(rules_path, "r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for idx, row in enumerate(reader, start=1):
                condition_list = self._parse_literal_list(row.get("condition"))
                selected_indices_list = self._parse_literal_list(row.get("selected_feature_indices"))
                original_indices_list = self._parse_literal_list(row.get("original_feature_indices"))
                prediction = (row.get("prediction") or "").strip()

                if not condition_list or not selected_indices_list or not prediction:
                    continue
                if len(condition_list) != len(selected_indices_list):
                    continue

                bounds: list[tuple[float, float]] = []
                selected_indices: list[int] = []
                original_indices: list[int | None] = []

                valid_rule = True
                for part_idx, raw_bound in enumerate(condition_list):
                    if not isinstance(raw_bound, (list, tuple)) or len(raw_bound) != 2:
                        valid_rule = False
                        break

                    low = self._to_float(str(raw_bound[0]))
                    high = self._to_float(str(raw_bound[1]))
                    lower, upper = (low, high) if low <= high else (high, low)
                    bounds.append((lower, upper))

                    try:
                        selected_feature_idx = int(selected_indices_list[part_idx])
                    except Exception:
                        valid_rule = False
                        break
                    selected_indices.append(selected_feature_idx)

                    if isinstance(original_indices_list, list) and part_idx < len(original_indices_list):
                        try:
                            original_indices.append(int(original_indices_list[part_idx]))
                        except Exception:
                            original_indices.append(None)
                    else:
                        original_indices.append(None)

                if not valid_rule:
                    continue

                parsed_rules.append(
                    {
                        "rule_id": idx,
                        "prediction": prediction,
                        "accuracy": self._to_float(row.get("accuracy")),
                        "numerosity": self._to_int(row.get("numerosity"), 1),
                        "fitness": self._to_float(row.get("fitness")),
                        "rule_strength": self._to_float(row.get("fitness"))
                        * self._to_int(row.get("numerosity"), 1),
                        "condition_bounds": bounds,
                        "selected_feature_indices": selected_indices,
                        "original_feature_indices": original_indices,
                    }
                )

        self.lcs_rules = parsed_rules
        return len(parsed_rules)

    def _candidate_alias_paths(self, artifact_dir: Path) -> list[Path]:
        candidates = [
            artifact_dir / "cnn_acoustic_correlations.csv",
            artifact_dir / "interpretability" / "cnn_acoustic_correlations.csv",
            artifact_dir.parent / "interpretability" / "cnn_acoustic_correlations.csv",
            PROJECT_ROOT / "outputs" / "interpretability" / "cnn_acoustic_correlations.csv",
        ]

        unique_paths: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            normalized = str(path.resolve()) if path.exists() else str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_paths.append(path)
        return unique_paths

    def _load_lcs_feature_aliases(self, artifact_dir: Path) -> int:
        self.lcs_feature_aliases = {}
        alias_path = next((path for path in self._candidate_alias_paths(artifact_dir) if path.exists()), None)
        if alias_path is None:
            return 0

        aliases: dict[int, dict] = {}
        with open(alias_path, "r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                try:
                    feature_idx = int(float((row.get("cnn_feature_index") or "").strip()))
                except Exception:
                    continue

                alias_name = (row.get("best_acoustic_match") or "").strip()
                if not alias_name:
                    continue

                correlation = self._to_float(row.get("correlation"), 0.0)
                abs_correlation = self._to_float(row.get("abs_correlation"), abs(correlation))
                aliases[feature_idx] = {
                    "name": alias_name,
                    "correlation": correlation,
                    "abs_correlation": abs_correlation,
                }

        self.lcs_feature_aliases = aliases
        return len(aliases)

    def _extract_lcs_reasons(self, selected_features: np.ndarray, decision_label: str) -> list[dict]:
        if not self.lcs_rules:
            return []
        if selected_features.ndim != 2 or selected_features.shape[0] == 0:
            return []

        segment_count = int(selected_features.shape[0])
        feature_count = int(selected_features.shape[1])
        normalized_decision = decision_label.lower()
        candidates: list[dict] = []

        for rule in self.lcs_rules:
            selected_indices: list[int] = rule["selected_feature_indices"]
            bounds: list[tuple[float, float]] = rule["condition_bounds"]
            if len(selected_indices) != len(bounds):
                continue
            if any(index < 0 or index >= feature_count for index in selected_indices):
                continue

            match_mask = np.ones(segment_count, dtype=bool)
            for feature_idx, (low, high) in zip(selected_indices, bounds):
                values = selected_features[:, feature_idx]
                match_mask &= (values >= low) & (values <= high)
                if not np.any(match_mask):
                    break

            matched_segments = int(np.sum(match_mask))
            if matched_segments == 0:
                continue

            match_rate = matched_segments / segment_count
            rule_prediction = str(rule["prediction"])
            normalized_rule_prediction = rule_prediction.lower()
            agrees_with_decision = (
                ("anomal" in normalized_rule_prediction and "anomal" in normalized_decision)
                or ("normal" in normalized_rule_prediction and "normal" in normalized_decision)
            )

            max_terms = min(4, len(bounds))
            terms: list[str] = []
            for idx in range(max_terms):
                lower, upper = bounds[idx]
                original_indices: list[int | None] = rule["original_feature_indices"]
                if idx < len(original_indices) and original_indices[idx] is not None:
                    original_idx = original_indices[idx]
                    alias = self.lcs_feature_aliases.get(original_idx)
                    if alias:
                        feature_label = (
                            f"cnn_feature[{original_idx}] "
                            f"(~{alias['name']}, r={alias['correlation']:.2f})"
                        )
                    else:
                        feature_label = f"cnn_feature[{original_idx}]"
                else:
                    feature_label = f"selected_feature[{selected_indices[idx]}]"
                terms.append(f"{feature_label} in [{lower:.2f}, {upper:.2f}]")

            truncated_suffix = " ..." if len(bounds) > max_terms else ""
            condition_text = f"IF {' AND '.join(terms)}{truncated_suffix} THEN {rule_prediction}"

            candidates.append(
                {
                    "rule_id": int(rule["rule_id"]),
                    "then_prediction": rule_prediction,
                    "agrees_with_decision": agrees_with_decision,
                    "match_rate": round(float(match_rate), 4),
                    "matched_segments": matched_segments,
                    "total_segments": segment_count,
                    "confidence": round(float(rule["accuracy"]), 4),
                    "support": int(rule["numerosity"]),
                    "condition_text": condition_text,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["agrees_with_decision"],
                item["match_rate"],
                item["confidence"],
                item["support"],
            ),
            reverse=True,
        )

        top_reasons = candidates[:3]
        for reason in top_reasons:
            reason.pop("agrees_with_decision", None)
        return top_reasons

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

        # Process in small batches to limit peak memory on constrained hosts
        batch_size = 4
        all_cnn_scores = []
        all_features = []

        for i in range(0, specs.shape[0], batch_size):
            batch = specs[i : i + batch_size]
            with torch.no_grad():
                logits = self.model(batch)
                batch_scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_cnn_scores.append(batch_scores)

                if self.has_lcs and self.feature_extractor is not None:
                    feats = self.feature_extractor(batch)
                    feats = feats.squeeze(-1).squeeze(-1).cpu().numpy()
                    all_features.append(feats)

            del batch, logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        cnn_scores = np.concatenate(all_cnn_scores, axis=0)
        del all_cnn_scores, specs

        cnn_probability = float(np.mean(cnn_scores))
        max_segment_probability = float(np.max(cnn_scores))
        lcs_probability = None
        lcs_reasons: list[dict] = []
        inference_source = "cnn"
        anomaly_probability = cnn_probability

        if self.has_lcs and self.feature_extractor is not None and all_features:
            features = np.concatenate(all_features, axis=0)
            del all_features

            scaled = self.scaler.transform(features)
            selected = self.selector.transform(scaled)
            lcs_scores = self.lcs_model.predict_proba(selected)[:, 1]
            lcs_probability = float(np.mean(lcs_scores))
            anomaly_probability = lcs_probability
            inference_source = "cnn_lcs"

        label = "Anomalous" if anomaly_probability >= ANOMALY_THRESHOLD else "Normal"
        if inference_source == "cnn_lcs":
            lcs_reasons = self._extract_lcs_reasons(selected, decision_label=label)

        notes = self._summarize_notes(
            probability=anomaly_probability,
            inference_source=inference_source,
            cnn_probability=cnn_probability,
            lcs_probability=lcs_probability,
        )
        if lcs_reasons:
            top_reason = lcs_reasons[0]
            notes = (
                f"{notes} Top matched rule #{top_reason['rule_id']} "
                f"(match={top_reason['match_rate'] * 100:.1f}%)."
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
        if lcs_reasons:
            response["lcs_reasons"] = lcs_reasons
            response["lcs_reasoning_note"] = self.lcs_alias_disclaimer
            response["lcs_aliases_available"] = bool(self.lcs_feature_aliases)
        if self.lcs_manifest is not None:
            response["lcs_run_id"] = self.lcs_manifest.get("run_id")

        return response
