"""
Eval harness — evaluate the acoustic inference pipeline against Mozilla Common Voice.

Usage
-----
    pip install datasets soundfile
    python -m scripts.eval_harness [--split dev] [--max-samples 200] [--language en]

What it measures
----------------
- Gender accuracy / precision / recall / F1 (binary: male vs female)
- Age bracket accuracy (4-class classification)
- Confidence calibration via Expected Calibration Error (ECE)

Dataset
-------
Mozilla Common Voice (via HuggingFace datasets).
The 'en' subset has gender and age labels for many clips.
Audio is NOT stored — clips are streamed and processed in RAM.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Lazy imports (allow script to parse --help without requiring all deps)
# ---------------------------------------------------------------------------

def _import_or_exit(pkg: str, install: str):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"[error] Missing package '{pkg}'. Install with:  pip install {install}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Age label normalisation (Common Voice uses non-standard labels)
# ---------------------------------------------------------------------------

CV_AGE_MAP = {
    "teens":    "18-30",
    "twenties": "18-30",
    "thirties": "31-45",
    "fourties": "31-45",
    "fifties":  "46-60",
    "sixties":  "60+",
    "seventies": "60+",
    "eighties": "60+",
    "nineties": "60+",
}

CV_GENDER_MAP = {
    "male":   "male",
    "female": "female",
    "other":  None,   # skip
    "":       None,
}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _accuracy(preds: List, labels: List) -> float:
    if not labels:
        return 0.0
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def _precision_recall_f1(preds: List[str], labels: List[str], pos_class: str):
    tp = sum(p == pos_class and l == pos_class for p, l in zip(preds, labels))
    fp = sum(p == pos_class and l != pos_class for p, l in zip(preds, labels))
    fn = sum(p != pos_class and l == pos_class for p, l in zip(preds, labels))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _expected_calibration_error(confs: List[float], corrects: List[bool], n_bins: int = 10) -> float:
    """ECE — lower is better. Perfect calibration = 0.0."""
    if not confs:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [lo <= c < hi for c in confs]
        if not any(mask):
            continue
        bin_confs = [c for c, m in zip(confs, mask) if m]
        bin_accs  = [a for a, m in zip(corrects, mask) if m]
        ece += len(bin_confs) / n * abs(np.mean(bin_confs) - np.mean(bin_accs))
    return ece


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(split: str = "validation", max_samples: int = 200, language: str = "en"):
    datasets = _import_or_exit("datasets", "datasets")
    soundfile = _import_or_exit("soundfile", "soundfile")
    import io

    # Lazy import of inference provider (requires librosa)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.services.inference import AcousticInferenceProvider
    provider = AcousticInferenceProvider()

    print(f"\n{'='*60}")
    print(f"  Voice Attribute Classifier — Eval Harness")
    print(f"  Dataset : Mozilla Common Voice ({language}, {split})")
    print(f"  Samples : up to {max_samples}")
    print(f"{'='*60}\n")

    print("Loading dataset (streaming=True)…")
    ds = datasets.load_dataset(
        "mozilla-foundation/common_voice_11_0",
        language,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    gender_preds, gender_labels = [], []
    gender_confs, gender_corrects = [], []

    age_preds, age_labels = [], []
    age_confs, age_corrects = [], []

    skipped = 0
    processed = 0
    total_ms = 0

    for row in ds:
        if processed >= max_samples:
            break

        # Extract ground-truth labels
        gt_gender = CV_GENDER_MAP.get(row.get("gender", ""), None)
        gt_age    = CV_AGE_MAP.get(row.get("age", ""), None)

        if gt_gender is None and gt_age is None:
            skipped += 1
            continue

        # Decode audio
        try:
            audio_dict = row["audio"]
            samples = np.array(audio_dict["array"], dtype=np.float32)
            sr = int(audio_dict["sampling_rate"])
        except Exception as e:
            skipped += 1
            continue

        # Run inference
        t0 = time.monotonic()
        try:
            g_pred, g_conf, a_pred, a_conf, _lang = provider.infer_attributes(samples, sr)
        except Exception as e:
            print(f"  [warn] Inference error: {e}")
            skipped += 1
            continue
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        total_ms += elapsed_ms
        processed += 1

        if gt_gender:
            gender_preds.append(g_pred)
            gender_labels.append(gt_gender)
            gender_confs.append(g_conf)
            gender_corrects.append(g_pred == gt_gender)

        if gt_age:
            age_preds.append(a_pred)
            age_labels.append(gt_age)
            age_confs.append(a_conf)
            age_corrects.append(a_pred == gt_age)

        if processed % 20 == 0:
            print(f"  Processed {processed}/{max_samples}…")

    # ---------------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------------
    avg_ms = total_ms / max(processed, 1)

    print(f"\n{'─'*60}")
    print(f"  Results  ({processed} samples processed, {skipped} skipped)")
    print(f"{'─'*60}")

    print(f"\n  Latency")
    print(f"    Average inference time : {avg_ms:.1f} ms/sample")

    # Gender metrics
    if gender_labels:
        g_acc = _accuracy(gender_preds, gender_labels)
        g_prec_m, g_rec_m, g_f1_m = _precision_recall_f1(gender_preds, gender_labels, "male")
        g_prec_f, g_rec_f, g_f1_f = _precision_recall_f1(gender_preds, gender_labels, "female")
        g_macro_f1 = (g_f1_m + g_f1_f) / 2
        g_ece = _expected_calibration_error(gender_confs, gender_corrects)

        print(f"\n  Gender ({len(gender_labels)} samples)")
        print(f"    Accuracy     : {g_acc:.3f}")
        print(f"    Macro F1     : {g_macro_f1:.3f}")
        print(f"    Male  P/R/F1 : {g_prec_m:.3f} / {g_rec_m:.3f} / {g_f1_m:.3f}")
        print(f"    Female P/R/F1: {g_prec_f:.3f} / {g_rec_f:.3f} / {g_f1_f:.3f}")
        print(f"    ECE (↓ better): {g_ece:.4f}")
    else:
        print("\n  Gender: no labeled samples found")

    # Age metrics
    if age_labels:
        a_acc = _accuracy(age_preds, age_labels)
        a_ece = _expected_calibration_error(age_confs, age_corrects)
        # Per-class breakdown
        brackets = ["18-30", "31-45", "46-60", "60+"]
        print(f"\n  Age Bracket ({len(age_labels)} samples)")
        print(f"    Accuracy   : {a_acc:.3f}")
        print(f"    ECE (↓ better): {a_ece:.4f}")
        print(f"    Per-class breakdown:")
        for bracket in brackets:
            prec, rec, f1 = _precision_recall_f1(age_preds, age_labels, bracket)
            count = age_labels.count(bracket)
            print(f"      {bracket:8s}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}  (n={count})")
    else:
        print("\n  Age Bracket: no labeled samples found")

    print(f"\n{'='*60}\n")

    # Return as dict for programmatic use
    return {
        "processed": processed,
        "skipped": skipped,
        "avg_inference_ms": avg_ms,
        "gender": {
            "accuracy": _accuracy(gender_preds, gender_labels) if gender_labels else None,
            "ece": _expected_calibration_error(gender_confs, gender_corrects) if gender_confs else None,
        },
        "age_bracket": {
            "accuracy": _accuracy(age_preds, age_labels) if age_labels else None,
            "ece": _expected_calibration_error(age_confs, age_corrects) if age_confs else None,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate voice attribute classifier against Mozilla Common Voice."
    )
    parser.add_argument("--split",       default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples to evaluate (default: 200)")
    parser.add_argument("--language",    default="en",         help="Common Voice language code (default: en)")
    parser.add_argument("--json",        action="store_true",  help="Output results as JSON")
    args = parser.parse_args()

    results = run_eval(split=args.split, max_samples=args.max_samples, language=args.language)

    if args.json:
        print(json.dumps(results, indent=2))
