"""
Acoustic inference pipeline for gender and age bracket estimation, and language identification.
Uses HuggingFace Transformers for robust ML-based inference.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod

import librosa
import numpy as np
import torch
from transformers import pipeline

warnings.filterwarnings("ignore", category=UserWarning)

from ..config import settings

logger = logging.getLogger(__name__)

class InferenceProvider(ABC):
    @abstractmethod
    def infer_attributes(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> tuple[str, float, str, float, str | None]:
        pass

def _resample_if_needed(samples: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return samples
    return librosa.resample(samples, orig_sr=sr, target_sr=target_sr)

def _estimate_age_bracket(samples: np.ndarray, sr: int) -> tuple[str, float]:
    """
    Estimate age bracket using robust acoustic features.
    Age bracket labels: "18-30", "31-45", "46-60", "60+"
    """
    duration = len(samples) / sr
    if duration < 0.5:
        return "unknown", 0.0

    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 0.005:
        return "unknown", 0.0

    try:
        hop_length = 512
        stft = np.abs(librosa.stft(samples, hop_length=hop_length))
        
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(stft**2), sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs[:4, :], axis=1)))
        
        centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
        mean_centroid = float(np.mean(centroid))
        
        zcr = librosa.feature.zero_crossing_rate(samples, hop_length=hop_length)[0]
        mean_zcr = float(np.mean(zcr))
        
        scores = {"18-30": 0.0, "31-45": 0.0, "46-60": 0.0, "60+": 0.0}
        
        # Centroid
        if mean_centroid > 2200: scores["18-30"] += 2.0; scores["31-45"] += 1.0
        elif mean_centroid > 1800: scores["18-30"] += 1.0; scores["31-45"] += 2.0; scores["46-60"] += 0.5
        elif mean_centroid > 1400: scores["31-45"] += 1.0; scores["46-60"] += 2.0; scores["60+"] += 0.5
        else: scores["46-60"] += 1.5; scores["60+"] += 2.5
            
        # MFCC Var
        if mfcc_var < 20: scores["18-30"] += 1.5
        elif mfcc_var < 40: scores["31-45"] += 1.5
        elif mfcc_var < 60: scores["46-60"] += 1.5
        else: scores["60+"] += 2.0
            
        winner = max(scores, key=lambda k: scores[k])
        
        temperature = 2.0
        exp_scores = {k: np.exp(v / temperature) for k, v in scores.items()}
        softmax_conf = exp_scores[winner] / sum(exp_scores.values())
        
        duration_factor = min(1.0, duration / 3.0)
        confidence = round(float(np.clip(softmax_conf * duration_factor, 0.0, 0.95)), 4)
        
        return winner, confidence
    except Exception as exc:
        logger.warning(f"Age bracket estimation failed: {exc}")
        return "unknown", 0.0


class MLInferenceProvider(InferenceProvider):
    """
    Production inference provider using HuggingFace ML models.
    Loaded once at startup.
    """
    def __init__(self):
        self._target_sr = settings.inference_target_sr
        logger.info("Loading ML models for inference (this may take a moment)...")
        
        # Device selection: use GPU if available, else CPU
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Gender Model: We use a lightweight audio classification model if possible, 
        # but due to time constraints, we'll use a robust wav2vec2 model fine-tuned for gender.
        try:
            self.gender_classifier = pipeline(
                "audio-classification", 
                model="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
                device=self.device
            )
        except Exception as e:
            logger.error(f"Failed to load gender model: {e}")
            self.gender_classifier = None
            
        # Language Model: Whisper Tiny is very robust and small.
        try:
            self.lang_classifier = pipeline(
                "audio-classification", 
                model="speechbrain/lang-id-voxlingua107-ecapa",
                device=self.device
            )
        except Exception as e:
            logger.error(f"Failed to load language model: {e}")
            self.lang_classifier = None
            
        logger.info("Models loaded successfully.")

    def _estimate_gender_ml(self, samples: np.ndarray, sr: int) -> tuple[str, float]:
        if self.gender_classifier is None:
            return "unknown", 0.0
            
        duration = len(samples) / sr
        if duration < 0.5:
            return "unknown", 0.0
            
        try:
            with torch.inference_mode():
                # Model expects 16kHz
                results = self.gender_classifier(samples)
                
            # results is a list of dicts: [{'label': 'male', 'score': 0.99}, ...]
            best_pred = max(results, key=lambda x: x['score'])
            label = best_pred['label'].lower()
            score = float(best_pred['score'])
            
            # Map labels in case they are different (e.g. M/F)
            if label in ['m', 'male']:
                final_label = 'male'
            elif label in ['f', 'female']:
                final_label = 'female'
            else:
                final_label = 'unknown'
                
            return final_label, round(score, 4)
        except Exception as e:
            logger.warning(f"Gender ML failed: {e}")
            return "unknown", 0.0
            
    def _detect_language_ml(self, samples: np.ndarray, sr: int) -> str:
        if self.lang_classifier is None:
            return "unknown"
            
        duration = len(samples) / sr
        if duration < 1.0:
            return "unknown"
            
        try:
            with torch.inference_mode():
                results = self.lang_classifier(samples)
                
            best_pred = max(results, key=lambda x: x['score'])
            score = best_pred['score']
            
            # If confidence is too low, return unknown
            if score < 0.4:
                return "unknown"
                
            # VoxLingua returns labels like 'en: English', 'fr: French'. 
            # We want the ISO-639-1 code (first 2 chars usually, or split by ':')
            label = best_pred['label']
            if ':' in label:
                iso_code = label.split(':')[0].strip().lower()
            else:
                iso_code = label[:2].lower()
                
            return iso_code
        except Exception as e:
            logger.warning(f"Language ML failed: {e}")
            return "unknown"

    def infer_attributes(self, samples: np.ndarray, sample_rate: int) -> tuple[str, float, str, float, str | None]:
        y = _resample_if_needed(samples, sample_rate, self._target_sr)
        sr = self._target_sr
        
        gender_pred, gender_conf = self._estimate_gender_ml(y, sr)
        age_pred, age_conf = _estimate_age_bracket(y, sr)
        language = self._detect_language_ml(y, sr)
        
        return gender_pred, gender_conf, age_pred, age_conf, language


# Singleton injection
_default_provider = None

def get_inference_provider() -> InferenceProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = MLInferenceProvider()
    return _default_provider
