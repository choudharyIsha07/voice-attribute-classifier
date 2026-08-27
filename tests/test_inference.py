import numpy as np
from app.services.inference import MockInferenceProvider

def test_mock_inference_provider():
    provider = MockInferenceProvider()
    
    # Dummy audio array
    samples = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)
    sample_rate = 16000
    
    gender_pred, gender_conf, age_pred, age_conf = provider.infer_attributes(samples, sample_rate)
    
    assert gender_pred == "unknown"
    assert gender_conf == 0.0
    assert age_pred == "unknown"
    assert age_conf == 0.0
