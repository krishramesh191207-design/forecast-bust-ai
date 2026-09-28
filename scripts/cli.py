#!/usr/bin/env python3
"""FBEWS command line. Every pipeline stage is runnable and resumable.

    python scripts/cli.py sandbox          generate the synthetic sandbox dataset
    python scripts/cli.py download         fetch real data (needs network + credentials)
    python scripts/cli.py features         build the training table
    python scripts/cli.py labels           bust labels + analogue enrichment
    python scripts/cli.py train            model ladder, regressors, importance, model card
    python scripts/cli.py evaluate         ablation, spatial CV, extreme-event evaluation
    python scripts/cli.py infer  [cycle]   write products JSON for a cycle
    python scripts/cli.py probe            report reachability of every real data source
    python scripts/cli.py all              sandbox -> features -> labels -> train -> evaluate
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbews.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sandbox", "download", "features", "labels", "train",
                                      "evaluate", "infer", "probe", "boundaries", "all"])
    ap.add_argument("cycle", nargs="?", default=None)
    ap.add_argument("--quick", action="store_true", help="short period, for smoke tests")
    args = ap.parse_args()
    cfg = load_config()

    if args.stage in ("sandbox", "all"):
        from fbews.sandbox.generate import (generate_forecasts, generate_prev_cycles,
                                            generate_states, generate_truth, write_manifest)
        print("[1/4] states"); generate_states(cfg, args.quick)
        print("[2/4] truth fields"); generate_truth(cfg, args.quick)
        print("[3/4] forecast cycles"); generate_forecasts(cfg, args.quick)
        print("[4/4] T-24 companion cycles"); generate_prev_cycles(cfg, args.quick)
        print(json.dumps(write_manifest(cfg, args.quick)["n_cycles_written"], indent=2), "cycles")

    if args.stage == "download":
        from fbews.ingestion.clients import REGISTRY, SourceUnavailable
        raw = cfg.path("raw")
        today = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        ok, blocked = [], []
        for key in ("ecmwf_opendata", "gefs", "ibtracs", "mjo_rmm"):
            try:
                fn = REGISTRY[key]
                res = fn(raw / key, today - dt.timedelta(days=1)) if key in ("ecmwf_opendata", "gefs") \
                    else fn(raw / key)
                ok.append((key, len(res.files)))
            except SourceUnavailable as exc:
                blocked.append((key, str(exc)))
        for k, n in ok:
            print(f"OK       {k}: {n} files")
        for k, msg in blocked:
            print(f"BLOCKED  {k}\n{msg}\n")

    if args.stage in ("features", "all"):
        from fbews.features.build import build_and_save
        print("features"); build_and_save(cfg)

    if args.stage in ("labels", "all"):
        from fbews.verification.labelling import run
        print("labels + analogues"); run(cfg)

    if args.stage in ("train", "all"):
        from fbews.models.train import (feature_importance, train_ladder,
                                        train_regressors, write_model_card)
        train_ladder(cfg); train_regressors(cfg); feature_importance(cfg); write_model_card(cfg)

    if args.stage in ("evaluate", "all"):
        from fbews.evaluation.ablation import (extreme_event_evaluation, run_ablation,
                                               run_spatial_cv)
        run_ablation(cfg); run_spatial_cv(cfg); extreme_event_evaluation(cfg)

    if args.stage == "infer":
        from fbews.inference.run import InferenceEngine, save_products
        eng = InferenceEngine(cfg)
        cycle = args.cycle or eng.available_cycles()[-1]
        print("wrote", save_products(cycle, cfg, eng))

    if args.stage == "boundaries":
        print("Boundary GeoJSON is prepared by scripts/prepare_boundaries.py")

    if args.stage == "probe":
        from fbews.ingestion.sources import probe_all
        for r in probe_all():
            state = "reachable" if r["reachable"] and r["credentials_present"] else "BLOCKED"
            print(f"{state:10s} {r['name']}")
            if state == "BLOCKED":
                print(f"           REASON: {r['reason']}\n           ENABLE: {r['how_to_enable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
