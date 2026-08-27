import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

class InferenceProvider(ABC):
    @abstractmethod
    def infer_attributes(self, samples: np.ndarray, sample_rate: int) -> Tuple[str, float, str, float]:
        """
        Returns: (gender_prediction, gender_confidence, age_prediction, age_confidence)
        """
        pass

class MockInferenceProvider(InferenceProvider):
    def infer_attributes(self, samples: np.ndarray, sample_rate: int) -> Tuple[str, float, str, float]:
        """
        Simulates inference of gender and age from audio.
        For safety and privacy, this isolated provider always returns 'unknown' with 0.0 confidence.
        """
        return "unknown", 0.0, "unknown", 0.0

default_provider = MockInferenceProvider()

def get_inference_provider() -> InferenceProvider:
    return default_provider
