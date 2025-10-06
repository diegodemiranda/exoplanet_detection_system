"""
Data ingestion from NASA archives (KOI cumulative, TOI, K2 P&C) and
construction of a dataset for model training.

Requires internet and packages: astroquery, lightkurve.
"""
from __future__ import annotations
import os
import time
from typing import List, Dict, Tuple
import numpy as np

from backend.config import settings
from backend.exoplanet_detector_model import ExoplanetDetector

# Heavy imports only when used
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
from typing import Optional


KOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=json"
TOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=json"
K2_URL  = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=json"


def _extract_aux_from_flux(flux: np.ndarray) -> np.ndarray:
    """Extract the 10 auxiliary features compatible with the service."""
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
    """Fetch catalogs from given sources and harmonize key fields.
    sources: subset of ["Kepler", "TESS", "K2"]
    Returns a list of records with possible keys: target_name, mission, label, ids
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
            print(f"DEBUG: Failed to fetch Kepler catalog: {e}")
            pass
    if "TESS" in sources:
        try:
            data = _safe_get(TOI_URL)
            for row in data:
                # Disposition field varies; try several
                disp = row.get("tfopwg_disp") or row.get("disposition") or row.get("toi_disposition")
                if disp is None:
                    continue
                disp_up = str(disp).upper().replace(" ", "_")
                if disp_up not in ("CANDIDATE", "CONFIRMED", "FALSE_POSITIVE"):
                    # Map some common aliases
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
            print(f"DEBUG: Failed to fetch TESS catalog: {e}")
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
            print(f"DEBUG: Failed to fetch K2 catalog: {e}")
            pass

    return records


def _download_light_curve(target: Dict, max_attempts: int = 2):
    """Download a light curve using lightkurve for a target record.
    Returns an array of fluxes (float32) or None.
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
            # print(f"DEBUG: No search_lightcurve results for {target.get('target_name')}")
            return None

        dl_dir = os.path.join(settings.model_path, "_lc")
        os.makedirs(dl_dir, exist_ok=True)
        # For multiple segments, stitch
        lc = sr.download(download_dir=dl_dir)
        try:
            # Some return LightCurveCollection; prefer stitch()
            if hasattr(lc, "stitch"):
                lc = lc.stitch()
        except Exception:
            pass

        if lc is None:
            return None

        # Normalize/remove NaNs for robustness across missions
        try:
            lcn = lc.normalize().remove_nans()
        except Exception:
            lcn = lc

        flux = None
        # Try known flux sources
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
            # Last resort: if lcn.flux exists and is indexable
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
        print(f"DEBUG: Failed to download/process {target.get('target_name')}: {e}")
        return None


def _load_flux_from_local_cache(rec: Dict) -> np.ndarray | None:
    """Attempt to construct the light curve from local cache at models/_lc/mastDownload.
    Supports Kepler, K2 and TESS. Returns np.ndarray or None.
    """
    mission = (rec.get("mission") or "").upper()
    ids = rec.get("ids", {})
    root = os.path.join(settings.model_path, "_lc", "mastDownload")

    def _read_fluxes_from_files(files: list) -> list:
        fluxes = []
        for fp in files:
            try:
                if lk is not None:
                    lc = lk.read(fp)
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
        return fluxes

    if mission == "KEPLER":
        kepid = ids.get("kepid")
        if kepid is None:
            return None
        kep9 = f"{int(kepid):09d}"
        mroot = os.path.join(root, "Kepler")
        if not os.path.isdir(mroot):
            return None
        pattern = os.path.join(mroot, f"kplr{kep9}_lc_*/")
        dirs = sorted(glob.glob(pattern))
        fits_files = []
        for d in dirs:
            fits_files.extend(glob.glob(os.path.join(d, "*.fits")))
        fluxes = _read_fluxes_from_files(fits_files)
        if not fluxes:
            return None
        flux = np.concatenate(fluxes)
        return flux if flux.size >= 100 else None

    if mission == "K2":
        epic = ids.get("epic")
        if epic is None:
            return None
        epic_str = f"{int(epic):09d}" if isinstance(epic, (int, np.integer)) or str(epic).isdigit() else str(epic)
        mroot = os.path.join(root, "Kepler")  # K2 also resides under 'Kepler' in lightkurve's MAST cache
        if not os.path.isdir(mroot):
            return None
        # Search for directories/files starting with 'ktwo<EPIC>'
        fits_files = glob.glob(os.path.join(mroot, f"ktwo{epic_str}_*_llc.fits"))
        if not fits_files:
            # Recursive search
            fits_files = glob.glob(os.path.join(mroot, f"**/ktwo{epic_str}*.*fits*"), recursive=True)
        fluxes = _read_fluxes_from_files(fits_files)
        if not fluxes:
            return None
        flux = np.concatenate(fluxes)
        return flux if flux.size >= 100 else None

    if mission == "TESS":
        tic = ids.get("tic")
        if tic is None:
            return None
        mroot = os.path.join(root, "TESS")
        if not os.path.isdir(mroot):
            return None
        # Search for any file containing the TIC in the name (TESS patterns vary), usually '*<tic>*_lc.fits'
        tic_str = str(int(tic)) if str(tic).isdigit() else str(tic)
        fits_files = glob.glob(os.path.join(mroot, f"**/*{tic_str}*lc.fits*"), recursive=True)
        if not fits_files:
            fits_files = glob.glob(os.path.join(mroot, f"**/*{tic_str}*.fits*"), recursive=True)
        fluxes = _read_fluxes_from_files(fits_files)
        if not fluxes:
            return None
        flux = np.concatenate(fluxes)
        return flux if flux.size >= 100 else None

    return None


def _has_kep_cache(kepid: int) -> bool:
    kep9 = f"{int(kepid):09d}"
    mroot = os.path.join(settings.model_path, "_lc", "mastDownload", "Kepler")
    pattern = os.path.join(mroot, f"kplr{kep9}_lc_*")
    return len(glob.glob(pattern)) > 0


def _build_from_kepler_cache(records: List[Dict], limit_per_class: int,
                             detector: ExoplanetDetector,
                             cap_per_mission_per_class: Optional[int] = None,
                             per_mission_class_counter: Optional[Dict[tuple, int]] = None) -> Tuple[list, list, list, list]:
    # Mapa kepid -> (label, mission)
    label_map = {r["ids"].get("kepid"): (r["label"], r.get("mission")) for r in records if r.get("mission") == "Kepler" and r.get("ids", {}).get("kepid") is not None}
    # Filtrar apenas os que têm cache local disponível
    kepids_with_cache = [k for k in label_map.keys() if _has_kep_cache(k)]
    if not kepids_with_cache:
        return [], [], [], []

    Xg, Xl, Xa, y = [], [], [], []
    per_class_counter = Counter()
    pmc = per_mission_class_counter if per_mission_class_counter is not None else {}

    # Percorrer kepids e coletar até limite por classe
    for kepid in kepids_with_cache:
        label, mission = label_map[kepid]
        if per_class_counter[label] >= limit_per_class:
            continue
        if cap_per_mission_per_class is not None:
            key = (mission, label)
            if pmc.get(key, 0) >= cap_per_mission_per_class:
                continue
        rec = {"mission": mission or "Kepler", "ids": {"kepid": kepid}, "label": label}
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
        if cap_per_mission_per_class is not None:
            key = (rec["mission"], label)
            pmc[key] = pmc.get(key, 0) + 1
        if all(per_class_counter[c] >= limit_per_class for c in ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]):
            break

    return Xg, Xl, Xa, y


def build_dataset_from_nasa(limit_per_class: int = 100,
                            missions: List[str] = None,
                            sleep_between: float = 0.2,
                            cap_per_mission_per_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build dataset from NASA catalogs + Lightkurve downloads.
    May be slow and fail for some targets.
    cap_per_mission_per_class: if set, limits per mission per class (e.g., up to N CONFIRMED from Kepler, N from TESS, etc.)
    """
    if missions is None:
        missions = ["Kepler", "TESS", "K2"]

    print(f"INFO: Fetching catalogs for missions: {missions}...")
    records = fetch_catalogs(missions)
    print(f"INFO: Total records found: {len(records)}")

    if not records:
        raise RuntimeError("No records found in NASA catalogs. Check your connection or API URLs.")

    # Prioritize ingestion via local Kepler cache (fast) if available
    Xg, Xl, Xa, y = [], [], [], []
    per_class_counter = {"CONFIRMED": 0, "CANDIDATE": 0, "FALSE_POSITIVE": 0}
    per_mission_class_counter: Dict[tuple, int] = {}

    detector = ExoplanetDetector(
        sequence_length=settings.sequence_length,
        n_features=settings.n_features,
        n_classes=settings.n_classes,
    )

    Xgc, Xlc, Xac, yc = _build_from_kepler_cache(records, limit_per_class, detector,
                                                 cap_per_mission_per_class=cap_per_mission_per_class,
                                                 per_mission_class_counter=per_mission_class_counter)
    if Xgc:
        Xg.extend(Xgc); Xl.extend(Xlc); Xa.extend(Xac); y.extend(yc)
        for lbl in y:
            if lbl in per_class_counter:
                per_class_counter[lbl] = y.count(lbl)
        # Update per mission/class counters from existing labels (assuming Kepler)
        for lbl in yc:
            key = ("Kepler", lbl)
            per_mission_class_counter[key] = per_mission_class_counter.get(key, 0) + 1
        print(f"INFO: Collected via Kepler cache: {len(yc)} samples")

    # If still missing samples, try remote download + per-item fallback
    if not all(per_class_counter[c] >= limit_per_class for c in per_class_counter):
        # Shuffle records for diversity
        rng = np.random.default_rng(123)
        rng.shuffle(records)
        for rec in records:
            label = rec["label"]
            mission = rec.get("mission") or "Unknown"
            if per_class_counter.get(label, 0) >= limit_per_class:
                continue
            if cap_per_mission_per_class is not None:
                key = (mission, label)
                if per_mission_class_counter.get(key, 0) >= cap_per_mission_per_class:
                    continue
            flux = _download_light_curve(rec)
            if flux is None:
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
            if cap_per_mission_per_class is not None:
                key = (mission, label)
                per_mission_class_counter[key] = per_mission_class_counter.get(key, 0) + 1
            if all(per_class_counter.get(c, 0) >= limit_per_class for c in per_class_counter):
                break
            time.sleep(sleep_between)

    if not Xg:
        raise RuntimeError("Failed to build NASA dataset: no light curves were downloaded")

    # Summary
    counts = {c: y.count(c) for c in ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]}
    print(f"INFO: Dataset built: {len(y)} samples. Class distribution: {counts}")

    return np.stack(Xg), np.stack(Xl), np.stack(Xa), y
