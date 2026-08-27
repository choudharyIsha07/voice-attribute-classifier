"""
Local Evaluation Script

Usage:
  python -m scripts.eval_local <path_to_directory_with_audio_files>

This script iterates through all supported audio files in the specified directory,
runs the inference pipeline, and prints a formatted table of the results.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.inference import get_inference_provider
from app.services.audio import ALLOWED_MIME_TYPES
from pydub import AudioSegment
import numpy as np
from app.services.quality import assess_audio_quality

def run_local_eval(directory: str):
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)
        
    print("Loading inference provider...")
    provider = get_inference_provider()
    print("Inference provider loaded.\n")
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    
    if not files:
        print(f"No supported audio files found in {directory}")
        return
        
    # Table Header
    header = f"{'filename':<25} | {'predicted_gender':<16} | {'gender_confidence':<17} | {'predicted_age':<13} | {'age_confidence':<14} | {'language':<8} | {'audio_quality':<13} | {'processing_ms':<13}"
    print(header)
    print("-" * len(header))
    
    for filename in sorted(files):
        filepath = os.path.join(directory, filename)
        
        # We simulate the exact processing done in routes.py
        try:
            start_time = time.monotonic()
            
            # 1. Decode audio
            audio: AudioSegment = AudioSegment.from_file(filepath)  # type: ignore
            if audio.channels > 1:
                audio = audio.set_channels(1)
            samples = np.array(audio.get_array_of_samples())
            if audio.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            elif audio.sample_width == 4:
                samples = samples.astype(np.float32) / 2147483648.0
            else:
                max_val = float(np.max(np.abs(samples))) if np.max(np.abs(samples)) > 0 else 1.0
                samples = samples.astype(np.float32) / max_val
            
            sample_rate = audio.frame_rate
            
            # 2. Quality
            quality = assess_audio_quality(samples, sample_rate)
            
            # 3. Inference
            g_pred, g_conf, a_pred, a_conf, lang = provider.infer_attributes(samples, sample_rate)
            
            # 4. Gate on quality
            if quality == "insufficient":
                g_pred = "unknown"
                g_conf = 0.0
                a_pred = "unknown"
                a_conf = 0.0
            elif quality == "degraded":
                g_conf = min(g_conf, 0.4)
                a_conf = min(a_conf, 0.4)
                
            processing_ms = int((time.monotonic() - start_time) * 1000)
            
            print(f"{filename[:25]:<25} | {g_pred:<16} | {g_conf:<17.2f} | {a_pred:<13} | {a_conf:<14.2f} | {str(lang):<8} | {quality:<13} | {processing_ms:<13}")
            
        except Exception as e:
            print(f"{filename[:25]:<25} | Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio files locally.")
    parser.add_argument("directory", help="Directory containing audio files to process.")
    args = parser.parse_args()
    
    run_local_eval(args.directory)
