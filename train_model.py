#!/usr/bin/env python3
"""
Treinamento do modelo de detecção de exoplanetas.

- Suporta dataset sintético rápido para smoke-test
- Suporta (opcional) ingestão de catálogos NASA (Kepler KOI, TESS TOI, K2) + download de curvas (Lightkurve)
- Salva modelo completo em models/exoplanet_model.keras

Uso rápido (sintético, recomendado para validar pipeline):
python train_model.py --mode synthetic --epochs 3 --samples-per-class 100

Uso (NASA, pode levar horas e requer internet):
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

# Opcional: importações pesadas apenas quando necessário
try:
    import tensorflow as tf
except Exception:  # pragma: no cover - ambiente sem TF
    tf = None

# NASA ingestion (opcional)
try:
    from data_ingestion import build_dataset_from_nasa
except Exception:  # pragma: no cover
    build_dataset_from_nasa = None

# Funções utilitárias compartilhadas com o serviço

def _extract_aux_from_flux(flux: np.ndarray) -> np.ndarray:
    """Extrai 10 features auxiliares alinhadas com o serviço.
    Para treino sintético, parâmetros estelares são zeros.
    """
    mean = float(np.mean(flux))
    std = float(np.std(flux))
    median = float(np.median(flux))
    if std < 1e-12:
        skew = 0.0
        kurt = 0.0
        cv = 0.0
    else:
        # Evitar dependências de scipy aqui
        # Aproximações: skew/kurt moment-based simples
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
    """Gera dataset sintético simples com três classes.
    - CONFIRMED: trânsito pronunciado
    - CANDIDATE: trânsito mais raso/ruidoso
    - FALSE_POSITIVE: sem trânsito (ruído)
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
            # Inserir dip gaussiano
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
            # Ruído e variações leves
            return base + 0.001 * np.sin(2 * np.pi * x * rng.uniform(2, 6))

    classes = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
    Xg, Xl, Xa, y = [], [], [], []

    detector = ExoplanetDetector(sequence_length=seq_len, n_features=1, n_classes=3)

    for cls in classes:
        for _ in range(samples_per_class):
            flux = synth_flux(cls).astype(np.float32)
            # Pré-processar como no detector
            flux_p = detector.preprocess_light_curve(flux, length=seq_len)
            Xg.append(flux_p.reshape(seq_len, 1))
            Xl.append(_build_local_view(flux_p, local_len).reshape(local_len, 1))
            Xa.append(_extract_aux_from_flux(flux_p))
            y.append(cls)

    Xg = np.stack(Xg, axis=0)
    Xl = np.stack(Xl, axis=0)
    Xa = np.stack(Xa, axis=0)
    return Xg, Xl, Xa, y


def train_and_save(mode: str,
                   epochs: int,
                   batch_size: int,
                   samples_per_class: int,
                   output_dir: str,
                   limit_per_class: int = 200,
                   missions: List[str] = None) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    # Montar dados
    if mode == "synthetic":
        Xg, Xl, Xa, y = make_synthetic_dataset(samples_per_class=samples_per_class)
    elif mode == "nasa":
        if build_dataset_from_nasa is None:
            raise RuntimeError("data_ingestion/build_dataset_from_nasa não disponível no ambiente.")
        if missions is None or len(missions) == 0:
            missions = ["Kepler", "TESS", "K2"]
        Xg, Xl, Xa, y = build_dataset_from_nasa(limit_per_class=limit_per_class, missions=missions)
    else:
        raise ValueError(f"Modo desconhecido: {mode}")

    # Criar e treinar modelo
    detector = ExoplanetDetector(
        sequence_length=settings.sequence_length,
        n_features=settings.n_features,
        n_classes=settings.n_classes,
        classes=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"],
    )
    model = detector.create_model()

    detector.train(Xg, Xl, Xa, y, validation_split=0.2, epochs=epochs, batch_size=batch_size)

    # Salvar apenas o modelo completo (.keras)
    keras_path = settings.model_full_path
    model.save(keras_path)

    # Smoke test de carregamento
    _loaded = tf.keras.models.load_model(keras_path) if tf is not None else None
    if _loaded is not None:
        # Predição em um batch para validar I/O
        _ = _loaded.predict([Xg[:1], Xl[:1], Xa[:1]], verbose=0)

    meta = {
        "keras_model": keras_path,
        "classes": ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"],
        "sequence_length": settings.sequence_length,
        "local_view_length": settings.local_view_length,
        "mode": mode,
        "missions": missions if mode == "nasa" else None,
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
                   help="Apenas para modo nasa: máximo por classe")
    p.add_argument("--missions", nargs="*", default=["Kepler", "TESS", "K2"],
                   help="Apenas para modo nasa: lista de missões a incluir")
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
    )
    print(json.dumps({"status": "ok", **paths}, indent=2))


if __name__ == "__main__":
    main()
