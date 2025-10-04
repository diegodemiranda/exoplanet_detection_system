"""
Ingestão de dados das bases NASA (KOI cumulative, TOI, K2 P&C) e
construção de dataset para treino do modelo.

Requer internet e pacotes: astroquery, lightkurve.
"""
from __future__ import annotations
import os
import time
from typing import List, Dict, Tuple
import numpy as np

from config import settings
from exoplanet_detector_model import ExoplanetDetector

# Importações pesadas apenas quando usadas
try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import lightkurve as lk
except Exception:  # pragma: no cover
    lk = None


KOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=json"
TOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=json"
K2_URL  = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=json"


def _extract_aux_from_flux(flux: np.ndarray) -> np.ndarray:
    """Extrai as 10 features auxiliares compatíveis com o serviço."""
    mean = float(np.mean(flux))
    std = float(np.std(flux))
    if std < 1e-12:
        skew = 0.0
        kurt = 0.0
        cv = 0.0
    else:
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
        0.0, 0.0, 0.0, 0.0, 0.0,  # teff, logg, feh, radius, mass
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


def _safe_get(url: str, timeout: int = 60) -> List[Dict]:
    if requests is None:
        raise RuntimeError("Pacote 'requests' não disponível no ambiente")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_catalogs(sources: List[str]) -> List[Dict]:
    """Busca catálogos das fontes indicadas e harmoniza campos principais.
    sources: subset de ["Kepler", "TESS", "K2"]
    Retorna lista de registros com possíveis chaves: target_name, mission, label, ids
    """
    records: List[Dict] = []
    if "Kepler" in sources:
        try:
            data = _safe_get(KOI_URL)
            for row in data:
                disp = row.get("koi_disposition")
                if disp not in ("CANDIDATE", "CONFIRMED", "FALSE POSITIVE"):
                    continue
                kepid = row.get("kepid")
                name = row.get("kepler_name") or row.get("kepoi_name") or f"KIC {kepid}"
                records.append({
                    "target_name": name,
                    "mission": "Kepler",
                    "label": disp.replace(" ", "_"),
                    "ids": {"kepid": kepid}
                })
        except Exception:
            pass
    if "TESS" in sources:
        try:
            data = _safe_get(TOI_URL)
            for row in data:
                # Campo de disposição varia; tentar vários
                disp = row.get("tfopwg_disp") or row.get("disposition") or row.get("toi_disposition")
                if disp is None:
                    continue
                disp_up = str(disp).upper().replace(" ", "_")
                if disp_up not in ("CANDIDATE", "CONFIRMED", "FALSE_POSITIVE"):
                    # Mapear alguns comuns
                    if disp_up in ("CANDIDATE", "PC", "KP" ):
                        disp_up = "CANDIDATE"
                    elif disp_up in ("CONFIRMED", "CP"):
                        disp_up = "CONFIRMED"
                    elif disp_up in ("FALSE_POSITIVE", "FP"):
                        disp_up = "FALSE_POSITIVE"
                    else:
                        continue
                tic = row.get("tic_id") or row.get("TICID") or row.get("TIC")
                name = f"TIC {tic}" if tic else row.get("toi")
                records.append({
                    "target_name": name,
                    "mission": "TESS",
                    "label": disp_up,
                    "ids": {"tic": tic}
                })
        except Exception:
            pass
    if "K2" in sources:
        try:
            data = _safe_get(K2_URL)
            for row in data:
                disp = row.get("Disposition") or row.get("k2_disposition") or row.get("disposition")
                if disp is None:
                    continue
                disp_up = str(disp).upper().replace(" ", "_")
                if disp_up not in ("CANDIDATE", "CONFIRMED", "FALSE_POSITIVE"):
                    continue
                epic = row.get("epic_number") or row.get("epic")
                name = row.get("k2_name") or (f"EPIC {epic}" if epic else None)
                if not name:
                    continue
                records.append({
                    "target_name": name,
                    "mission": "K2",
                    "label": disp_up,
                    "ids": {"epic": epic}
                })
        except Exception:
            pass

    return records


def _download_light_curve(target: Dict, max_attempts: int = 2):
    """Baixa uma curva de luz usando lightkurve para um registro de alvo.
    Retorna array de fluxos (float32) ou None.
    """
    if lk is None:
        return None

    mission = target["mission"]
    ids = target.get("ids", {})
    try:
        if mission == "Kepler":
            kepid = ids.get("kepid")
            if kepid is None:
                return None
            sr = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler")
        elif mission == "TESS":
            tic = ids.get("tic")
            if tic is None:
                return None
            sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
        else:  # K2
            epic = ids.get("epic")
            if epic is None:
                return None
            sr = lk.search_lightcurve(f"EPIC {epic}", mission="K2")

        if len(sr) == 0:
            return None
        lc = sr.download(quality_bitmask="hard", download_dir=os.path.join(settings.model_path, "_lc"))
        if lc is None:
            return None
        # Converter para PDCSAP_FLUX se disponível
        if hasattr(lc, 'PDCSAP_FLUX'):
            flux = np.asarray(lc.PDCSAP_FLUX, dtype=np.float32)
        else:
            flux = np.asarray(lc.flux.value if hasattr(lc.flux, 'value') else lc.flux, dtype=np.float32)
        # Remover NaNs
        flux = flux[np.isfinite(flux)]
        if flux.size < 100:
            return None
        return flux
    except Exception:
        return None


def build_dataset_from_nasa(limit_per_class: int = 100,
                            missions: List[str] = None,
                            sleep_between: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Monta dataset a partir dos catálogos NASA + download Lightkurve.
    Pode ser lento e falhar para alguns alvos.
    """
    if missions is None:
        missions = ["Kepler", "TESS", "K2"]

    detector = ExoplanetDetector(
        sequence_length=settings.sequence_length,
        n_features=settings.n_features,
        n_classes=settings.n_classes,
    )

    records = fetch_catalogs(missions)
    # Embaralhar para diversidade
    rng = np.random.default_rng(123)
    rng.shuffle(records)

    Xg, Xl, Xa, y = [], [], [], []
    per_class_counter = {"CONFIRMED": 0, "CANDIDATE": 0, "FALSE_POSITIVE": 0}

    for rec in records:
        label = rec["label"]
        if per_class_counter.get(label, 0) >= limit_per_class:
            continue
        flux = _download_light_curve(rec)
        if flux is None:
            continue
        # Pré-processar alinhado ao detector
        flux_p = detector.preprocess_light_curve(flux, length=settings.sequence_length)
        Xg.append(flux_p.reshape(settings.sequence_length, 1))
        # local view
        local = _build_local_view(flux_p, settings.local_view_length)
        Xl.append(local.reshape(settings.local_view_length, 1))
        # aux features
        aux = _extract_aux_from_flux(flux_p)
        Xa.append(aux)
        y.append(label)
        per_class_counter[label] = per_class_counter.get(label, 0) + 1

        # Check se já atingiu limites
        if all(per_class_counter.get(c, 0) >= limit_per_class for c in per_class_counter):
            break
        time.sleep(sleep_between)

    if not Xg:
        raise RuntimeError("Falha ao montar dataset NASA: nenhuma curva de luz foi baixada")

    return np.stack(Xg), np.stack(Xl), np.stack(Xa), y
