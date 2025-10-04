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

try:
    from astropy.io import fits
except Exception:  # pragma: no cover
    fits = None
import re
import glob
from collections import Counter


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
        except Exception as e:
            print(f"DEBUG: Falha ao buscar catálogo Kepler: {e}")
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
        except Exception as e:
            print(f"DEBUG: Falha ao buscar catálogo TESS: {e}")
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
        except Exception as e:
            print(f"DEBUG: Falha ao buscar catálogo K2: {e}")
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
            # print(f"DEBUG: Nenhum resultado de search_lightcurve para {target.get('target_name')}")
            return None

        dl_dir = os.path.join(settings.model_path, "_lc")
        os.makedirs(dl_dir, exist_ok=True)
        # Para múltiplos segmentos, costurar
        lc = sr.download(download_dir=dl_dir)
        try:
            # Alguns retornam LightCurveCollection; preferir stitch()
            if hasattr(lc, "stitch"):
                lc = lc.stitch()
        except Exception:
            pass

        if lc is None:
            return None

        # Normalizar/remover NaNs para robustez entre missões
        try:
            lcn = lc.normalize().remove_nans()
        except Exception:
            lcn = lc

        flux = None
        # Tentar fontes de flux conhecidas
        for attr in ("PDCSAP_FLUX", "SAP_FLUX", "flux"):
            if hasattr(lcn, attr):
                val = getattr(lcn, attr)
                try:
                    arr = val.value if hasattr(val, "value") else np.asarray(val)
                except Exception:
                    continue
                arr = np.asarray(arr, dtype=np.float32)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    flux = arr
                    break

        if flux is None:
            # Último recurso: se existir lcn.flux e for indexável
            try:
                val = lcn.flux
                arr = val.value if hasattr(val, "value") else np.asarray(val)
                arr = np.asarray(arr, dtype=np.float32)
                arr = arr[np.isfinite(arr)]
                flux = arr if arr.size > 0 else None
            except Exception:
                flux = None

        if flux is None or flux.size < 100:
            return None
        return flux
    except Exception as e:
        print(f"DEBUG: Falha ao baixar/processar {target.get('target_name')}: {e}")
        return None


def _load_flux_from_local_cache(rec: Dict) -> np.ndarray | None:
    """Tenta construir a curva de luz a partir do cache local em models/_lc/mastDownload.
    Suporta Kepler inicialmente (pastas kplrXXXXXXXXX_lc_*). Retorna np.ndarray ou None.
    """
    mission = rec.get("mission")
    ids = rec.get("ids", {})
    root = os.path.join(settings.model_path, "_lc", "mastDownload")
    if mission != "Kepler":
        return None  # por ora, só Kepler
    kepid = ids.get("kepid")
    if kepid is None:
        return None
    kep9 = f"{int(kepid):09d}"
    mroot = os.path.join(root, "Kepler")
    if not os.path.isdir(mroot):
        return None
    pattern = os.path.join(mroot, f"kplr{kep9}_lc_*/")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return None
    # Coletar todos os arquivos .fits nas subpastas
    fits_files = []
    for d in dirs:
        fits_files.extend(glob.glob(os.path.join(d, "*.fits")))
    if not fits_files:
        return None
    # Ler e concatenar fluxos
    fluxes = []
    for fp in fits_files:
        try:
            if lk is not None:
                lc = lk.read(fp)
                # normalizar e limpar
                try:
                    lc = lc.normalize().remove_nans()
                except Exception:
                    pass
                cand = None
                for attr in ("PDCSAP_FLUX", "SAP_FLUX", "flux"):
                    if hasattr(lc, attr):
                        v = getattr(lc, attr)
                        arr = v.value if hasattr(v, "value") else np.asarray(v)
                        arr = np.asarray(arr, dtype=np.float32)
                        arr = arr[np.isfinite(arr)]
                        if arr.size:
                            cand = arr
                            break
                if cand is None and hasattr(lc, "flux"):
                    v = lc.flux
                    arr = v.value if hasattr(v, "value") else np.asarray(v)
                    arr = np.asarray(arr, dtype=np.float32)
                    arr = arr[np.isfinite(arr)]
                    cand = arr if arr.size else None
                if cand is not None and cand.size >= 100:
                    fluxes.append(cand)
                    continue
            # Fallback: astropy FITS direto
            if fits is not None:
                with fits.open(fp) as hdul:
                    data = None
                    for name in ("PDCSAP_FLUX", "SAP_FLUX", "FLUX"):
                        try:
                            if name in hdul[1].data.columns.names:
                                data = hdul[1].data[name]
                                break
                        except Exception:
                            continue
                    if data is not None:
                        arr = np.asarray(data, dtype=np.float32)
                        arr = arr[np.isfinite(arr)]
                        if arr.size >= 100:
                            fluxes.append(arr)
        except Exception:
            continue
    if not fluxes:
        return None
    # Concatenar segmentos
    flux = np.concatenate(fluxes)
    return flux if flux.size >= 100 else None


def _has_kep_cache(kepid: int) -> bool:
    kep9 = f"{int(kepid):09d}"
    mroot = os.path.join(settings.model_path, "_lc", "mastDownload", "Kepler")
    pattern = os.path.join(mroot, f"kplr{kep9}_lc_*")
    return len(glob.glob(pattern)) > 0


def _build_from_kepler_cache(records: List[Dict], limit_per_class: int,
                             detector: ExoplanetDetector) -> Tuple[list, list, list, list]:
    # Mapa kepid -> label
    label_map = {r["ids"].get("kepid"): r["label"] for r in records if r.get("mission") == "Kepler" and r.get("ids", {}).get("kepid") is not None}
    # Filtrar apenas os que têm cache local disponível
    kepids_with_cache = [k for k in label_map.keys() if _has_kep_cache(k)]
    if not kepids_with_cache:
        return [], [], [], []

    Xg, Xl, Xa, y = [], [], [], []
    per_class_counter = Counter()

    # Percorrer kepids e coletar até limite por classe
    for kepid in kepids_with_cache:
        label = label_map[kepid]
        if per_class_counter[label] >= limit_per_class:
            continue
        rec = {"mission": "Kepler", "ids": {"kepid": kepid}, "label": label}
        flux = _load_flux_from_local_cache(rec)
        if flux is None:
            continue
        flux_p = detector.preprocess_light_curve(flux, length=settings.sequence_length)
        Xg.append(flux_p.reshape(settings.sequence_length, 1))
        local = _build_local_view(flux_p, settings.local_view_length)
        Xl.append(local.reshape(settings.local_view_length, 1))
        Xa.append(_extract_aux_from_flux(flux_p))
        y.append(label)
        per_class_counter[label] += 1
        if all(per_class_counter[c] >= limit_per_class for c in ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]):
            break

    return Xg, Xl, Xa, y


def build_dataset_from_nasa(limit_per_class: int = 100,
                            missions: List[str] = None,
                            sleep_between: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Monta dataset a partir dos catálogos NASA + download Lightkurve.
    Pode ser lento e falhar para alguns alvos.
    """
    if missions is None:
        missions = ["Kepler", "TESS", "K2"]

    print(f"INFO: Buscando catálogos para as missões: {missions}...")
    records = fetch_catalogs(missions)
    print(f"INFO: Total de registros encontrados nos catálogos: {len(records)}")

    if not records:
        raise RuntimeError("Nenhum registro encontrado nos catálogos da NASA. Verifique a conexão ou as URLs da API.")

    # Priorizar ingestão via cache local Kepler (rápida) se disponível
    Xg, Xl, Xa, y = [], [], [], []
    per_class_counter = {"CONFIRMED": 0, "CANDIDATE": 0, "FALSE_POSITIVE": 0}

    detector = ExoplanetDetector(
        sequence_length=settings.sequence_length,
        n_features=settings.n_features,
        n_classes=settings.n_classes,
    )

    Xgc, Xlc, Xac, yc = _build_from_kepler_cache(records, limit_per_class, detector)
    if Xgc:
        Xg.extend(Xgc); Xl.extend(Xlc); Xa.extend(Xac); y.extend(yc)
        for lbl in y:
            if lbl in per_class_counter:
                per_class_counter[lbl] = y.count(lbl)
        print(f"INFO: Coletado via cache Kepler: {len(yc)} amostras")

    # Se ainda faltar amostras, tentar download remoto + fallback por item
    if not all(per_class_counter[c] >= limit_per_class for c in per_class_counter):
        # Embaralhar registros para diversidade
        rng = np.random.default_rng(123)
        rng.shuffle(records)
        for rec in records:
            label = rec["label"]
            if per_class_counter.get(label, 0) >= limit_per_class:
                continue
            flux = _download_light_curve(rec)
            if flux is None:
                # tentar cache local (já cobre Kepler, mas mantém aqui por completude)
                flux = _load_flux_from_local_cache(rec)
            if flux is None:
                continue
            flux_p = detector.preprocess_light_curve(flux, length=settings.sequence_length)
            Xg.append(flux_p.reshape(settings.sequence_length, 1))
            local = _build_local_view(flux_p, settings.local_view_length)
            Xl.append(local.reshape(settings.local_view_length, 1))
            Xa.append(_extract_aux_from_flux(flux_p))
            y.append(label)
            per_class_counter[label] = per_class_counter.get(label, 0) + 1
            if all(per_class_counter.get(c, 0) >= limit_per_class for c in per_class_counter):
                break
            time.sleep(sleep_between)

    if not Xg:
        raise RuntimeError("Falha ao montar dataset NASA: nenhuma curva de luz foi baixada")

    # Resumo
    counts = {c: y.count(c) for c in ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]}
    print(f"INFO: Dataset montado: {len(y)} amostras. Distribuição: {counts}")

    return np.stack(Xg), np.stack(Xl), np.stack(Xa), y
