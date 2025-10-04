#!/usr/bin/env python3
"""
Heavy prefetch of light curves from MAST/Lightkurve to populate local cache,
speed up dataset assembly and reduce failures during training.

Usage:
/opt/homebrew/Caskroom/miniconda/base/envs/nasa_project/bin/python prefetch_nasa_cache.py \
--missions Kepler TESS K2 --max-per-mission 5000 --sleep 0.1

Cache will be saved at models/_lc/mastDownload/...
"""
from __future__ import annotations
import argparse
import time
import sys

from data_ingestion import fetch_catalogs, _download_light_curve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--missions", nargs="*", default=["Kepler", "TESS", "K2"],
                        help="Missões a prefetch (Kepler, TESS, K2)")
    p.add_argument("--max-per-mission", type=int, default=5000,
                        help="Máximo de alvos a tentar por missão (prefetch)")
    p.add_argument("--sleep", type=float, default=0.1,
                        help="Pausa entre downloads para reduzir rate limit")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[PREFETCH] Iniciando prefetch para missões: {args.missions}", flush=True)
    try:
        records = fetch_catalogs(args.missions)
    except Exception as e:
        print(f"[PREFETCH] Falha ao buscar catálogos: {e}", flush=True)
        sys.exit(1)

    # Group by mission
    by_mission = {m: [] for m in args.missions}
    for rec in records:
        m = rec.get("mission")
        if m in by_mission:
            by_mission[m].append(rec)

    total_ok = 0
    total_fail = 0

    for mission in args.missions:
        recs = by_mission.get(mission, [])[: args.max_per_mission]
        ok = 0
        fail = 0
        print(f"[PREFETCH] {mission}: {len(recs)} alvos para tentar...", flush=True)
        for i, rec in enumerate(recs, 1):
            flux = _download_light_curve(rec)
            if flux is not None:
                ok += 1
            else:
                fail += 1
            if i % 50 == 0:
                print(f"[PREFETCH] {mission}: {i}/{len(recs)} tentados | ok={ok} fail={fail}", flush=True)
            time.sleep(args.sleep)
        print(f"[PREFETCH] {mission}: concluído | ok={ok} fail={fail}", flush=True)
        total_ok += ok
        total_fail += fail

    print(f"[PREFETCH] Finalizado. Total ok={total_ok} fail={total_fail}", flush=True)


if __name__ == "__main__":
    main()
