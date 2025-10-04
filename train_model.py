#!/usr/bin/env python3
"""
Exoplanet detection model training.

- Supports fast synthetic dataset for smoke-test
- Supports (optional) ingestion from NASA catalogs (Kepler KOI, TESS TOI, K2) + light curve downloads (Lightkurve)
- Saves full model in models/exoplanet_model.keras

Quick usage (synthetic, recommended to validate pipeline):
python train_model.py --mode synthetic --epochs 3 --samples-per-class 100

Usage (NASA, can take hours and requires internet):
python train_model.py --mode nasa --epochs 20 --limit-per-class 500 --missions Kepler TESS K2

"""
from __future__ import annotations
import os
import json
import argparse
from typing import Tuple, List, Dict
import numpy as np

from config import settings
from exoplanet_detector_model import ExoplanetDetector

# Optional: heavy imports only when necessary
try:
    import tensorflow as tf
except Exception:  # pragma: no cover - environment without TF
    tf = None

# NASA ingestion (optional)
try:
    from data_ingestion import build_dataset_from_nasa
except Exception:  # pragma: no cover
    build_dataset_from_nasa = None

from sklearn.model_selection import StratifiedShuffleSplit

# Utility functions shared with the service

def _extract_aux_from_flux(flux: np.ndarray) -> np.ndarray:
    """Extract 10 auxiliary features aligned with the service.
    For synthetic training, stellar parameters are zeros.
    """
    mean = float(np.mean(flux))
    std = float(np.std(flux))
    median = float(np.median(flux))
    if std < 1e-12:
        skew = 0.0
        kurt = 0.0
        cv = 0.0
    else:
        # Avoid scipy dependencies here
        # Approximations: simple moment-based skew/kurt
        centered = flux - mean
        m2 = np.mean(centered**2) + 1e-12
        m3 = np.mean(centered**3)
        m4 = np.mean(centered**4)
        skew = float(m3 / (m2 ** 1.5))
        kurt = float(m4 / (m2 ** 2) - 3.0)
        cv = float(std / (abs(mean) + 1e-8))

    depth_estimate = float(abs(np.min(flux)))
    snr_estimate = float(depth_estimate / (std + 1e-8))

    aux = [
        0.0,  # teff
        0.0,  # logg
        0.0,  # feh
        0.0,  # radius
        0.0,  # mass
        depth_estimate,
        snr_estimate,
        cv,
        skew,
        kurt,
    ]
    return np.array(aux, dtype=np.float32)


def _build_local_view(flux: np.ndarray, local_len: int) -> np.ndarray:
    idx_min = int(np.argmin(flux)) if len(flux) > 0 else 0
    half = local_len // 2
    start = max(0, idx_min - half)
    end = start + local_len
    if end > len(flux):
        end = len(flux)
        start = max(0, end - local_len)
    local_view = flux[start:end]
    if len(local_view) < local_len:
        local_view = np.pad(local_view, (0, local_len - len(local_view)), 'constant')
    return local_view


def make_synthetic_dataset(samples_per_class: int = 100,
                           seq_len: int = None,
                           local_len: int = None,
                           rng_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Generate simple synthetic dataset with three classes.
    - CONFIRMED: pronounced transit
    - CANDIDATE: shallower/noisier transit
    - FALSE_POSITIVE: no transit (noise)
    """
    if seq_len is None:
        seq_len = settings.sequence_length
    if local_len is None:
        local_len = settings.local_view_length

    rng = np.random.default_rng(rng_seed)

    def synth_flux(kind: str) -> np.ndarray:
        x = np.linspace(0, 1, seq_len)
        noise = 0.002 * rng.standard_normal(seq_len)
        base = 1.0 + noise
        if kind == "CONFIRMED":
            # Insert Gaussian dip
            center = rng.uniform(0.3, 0.7)
            width = rng.uniform(0.01, 0.03)
            depth = rng.uniform(0.005, 0.02)
            dip = np.exp(-0.5 * ((x - center) / width) ** 2)
            return base - depth * dip
        elif kind == "CANDIDATE":
            center = rng.uniform(0.3, 0.7)
            width = rng.uniform(0.01, 0.03)
            depth = rng.uniform(0.002, 0.008)
            dip = np.exp(-0.5 * ((x - center) / width) ** 2)
            return base - depth * dip + 0.001 * rng.standard_normal(seq_len)
        else:  # FALSE_POSITIVE
            # Noise and mild variations
            return base + 0.001 * np.sin(2 * np.pi * x * rng.uniform(2, 6))

    classes = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
    Xg, Xl, Xa, y = [], [], [], []

    detector = ExoplanetDetector(sequence_length=seq_len, n_features=1, n_classes=3)

    for cls in classes:
        for _ in range(samples_per_class):
            flux = synth_flux(cls).astype(np.float32)
            # Preprocess as in the detector
            flux_p = detector.preprocess_light_curve(flux, length=seq_len)
            Xg.append(flux_p.reshape(seq_len, 1))
            Xl.append(_build_local_view(flux_p, local_len).reshape(local_len, 1))
            Xa.append(_extract_aux_from_flux(flux_p))
            y.append(cls)

    Xg = np.stack(Xg, axis=0)
    Xl = np.stack(Xl, axis=0)
    Xa = np.stack(Xa, axis=0)
    return Xg, Xl, Xa, y


def _augment_triplets(Xg: np.ndarray, Xl: np.ndarray, Xa: np.ndarray, y: List[str], noise_level: float = 0.003,
                      include_flip: bool = True, include_noise: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Consistent augmentation for global/local/aux.
    - small Gaussian noise
    - time reflection
    Recomputes Xa from Xg using _extract_aux_from_flux.
    """
    Xg_aug, Xl_aug, Xa_aug, y_aug = [Xg.copy()], [Xl.copy()], [Xa.copy()], [list(y)]

    def recalc_aux(xg_sample: np.ndarray) -> np.ndarray:
        flux = xg_sample.squeeze(-1)
        return _extract_aux_from_flux(flux)

    n = Xg.shape[0]
    if include_noise:
        g_noise = Xg + np.random.normal(0, noise_level, Xg.shape).astype(Xg.dtype)
        l_noise = Xl + np.random.normal(0, noise_level, Xl.shape).astype(Xl.dtype)
        xa_noise = np.stack([recalc_aux(g_noise[i]) for i in range(n)], axis=0)
        Xg_aug.append(g_noise); Xl_aug.append(l_noise); Xa_aug.append(xa_noise); y_aug.append(list(y))
    if include_flip:
        g_flip = np.flip(Xg, axis=1)
        l_flip = np.flip(Xl, axis=1)
        xa_flip = np.stack([recalc_aux(g_flip[i]) for i in range(n)], axis=0)
        Xg_aug.append(g_flip); Xl_aug.append(l_flip); Xa_aug.append(xa_flip); y_aug.append(list(y))

    Xg_out = np.concatenate(Xg_aug, axis=0)
    Xl_out = np.concatenate(Xl_aug, axis=0)
    Xa_out = np.concatenate(Xa_aug, axis=0)
    y_out = sum(y_aug, [])
    return Xg_out, Xl_out, Xa_out, y_out


def train_and_save(mode: str,
                   epochs: int,
                   batch_size: int,
                   samples_per_class: int,
                   output_dir: str,
                   limit_per_class: int = 200,
                   missions: List[str] = None,
                   cap_per_mission_per_class: int | None = None) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    # Build data
    if mode == "synthetic":
        Xg, Xl, Xa, y = make_synthetic_dataset(samples_per_class=samples_per_class)
    elif mode == "nasa":
        if build_dataset_from_nasa is None:
            raise RuntimeError("data_ingestion/build_dataset_from_nasa não disponível no ambiente.")
        if missions is None or len(missions) == 0:
            missions = ["Kepler", "TESS", "K2"]
        Xg, Xl, Xa, y = build_dataset_from_nasa(limit_per_class=limit_per_class, missions=missions,
                                               cap_per_mission_per_class=cap_per_mission_per_class)
    else:
        raise ValueError(f"Modo desconhecido: {mode}")

    # Stratified split
    y_arr = np.array(y)
    classes = np.unique(y_arr)
    n_samples = len(y_arr)
    # Ensure at least 1 per class in test and train sets
    base_test = int(round(n_samples * 0.2))
    test_count = max(base_test, len(classes))
    # Do not exceed n_samples - n_classes (to leave at least 1 per class in train)
    max_test = max(len(classes), n_samples - len(classes))
    test_count = min(test_count, max_test)
    if test_count <= 0 or test_count >= n_samples:
        test_count = len(classes)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_count, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(y_arr)), y_arr))

    Xg_tr, Xl_tr, Xa_tr, y_tr = Xg[train_idx], Xl[train_idx], Xa[train_idx], y_arr[train_idx].tolist()
    Xg_val, Xl_val, Xa_val, y_val = Xg[val_idx], Xl[val_idx], Xa[val_idx], y_arr[val_idx].tolist()

    # Data augmentation (train only)
    Xg_tr, Xl_tr, Xa_tr, y_tr = _augment_triplets(Xg_tr, Xl_tr, Xa_tr, y_tr,
                                                  noise_level=0.003, include_flip=True, include_noise=True)

    # Create and train model
    detector = ExoplanetDetector(
        sequence_length=settings.sequence_length,
        n_features=settings.n_features,
        n_classes=settings.n_classes,
        classes=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"],
    )
    model = detector.create_model()

    detector.train(Xg_tr, Xl_tr, Xa_tr, y_tr,
                   validation_data=(Xg_val, Xl_val, Xa_val, y_val),
                   epochs=epochs, batch_size=batch_size)

    # Validation evaluation
    try:
        report, cm, _ = detector.evaluate_model(Xg_val, Xl_val, Xa_val, y_val)
        val_accuracy = float(report.get('accuracy', 0.0))
        print(f"VALIDATION ACCURACY: {val_accuracy:.4f}")
    except Exception as e:
        print(f"WARN: Falha na avaliação de validação: {e}")
        report = None
        val_accuracy = None

    # Save only the full model (.keras)
    keras_path = settings.model_full_path
    model.save(keras_path)

    # Load smoke test
    try:
        _loaded = tf.keras.models.load_model(keras_path) if tf is not None else None
        if _loaded is not None:
            _ = _loaded.predict([Xg_val[:1], Xl_val[:1], Xa_val[:1]], verbose=0)
    except Exception as e:
        print(f"WARN: Falha no smoke test de load_model: {e}")

    meta = {
        "keras_model": keras_path,
        "classes": ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"],
        "sequence_length": settings.sequence_length,
        "local_view_length": settings.local_view_length,
        "mode": mode,
        "missions": missions if mode == "nasa" else None,
        "validation": {
            "accuracy": val_accuracy,
            "report": report,
        }
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"keras_model": keras_path}


def parse_args():
    p = argparse.ArgumentParser(description="Treino do ExoplanetDetector")
    p.add_argument("--mode", choices=["synthetic", "nasa"], default="synthetic",
                   help="Fonte dos dados: sintético rápido ou NASA (lento)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples-per-class", type=int, default=100,
                   help="Apenas para modo synthetic")
    p.add_argument("--output-dir", type=str, default=settings.model_path)
    p.add_argument("--limit-per-class", type=int, default=200,
                   help="Apenas para modo nasa: máximo por classe (total)")
    p.add_argument("--missions", nargs="*", default=["Kepler", "TESS", "K2"],
                   help="Apenas para modo nasa: lista de missões a incluir")
    p.add_argument("--cap-per-mission-per-class", type=int, default=None,
                   help="Apenas modo nasa: limite por missão por classe (para balancear catálogos enviesados)")
    return p.parse_args()


def main():
    args = parse_args()
    paths = train_and_save(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
        output_dir=args.output_dir,
        limit_per_class=args.limit_per_class,
        missions=args.missions,
        cap_per_mission_per_class=args.cap_per_mission_per_class,
    )
    print(json.dumps({"status": "ok", **paths}, indent=2))


if __name__ == "__main__":
    main()
