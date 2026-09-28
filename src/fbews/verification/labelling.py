"""Stage: bust labelling + analogue enrichment.

Produces data/labels/labelled_table.parquet, the table the models train on.

Order matters and is enforced here:
  1. the bust definition is fitted on the TRAIN split only,
  2. labels are applied to every split,
  3. the analogue index is built from TRAIN rows only (using their verified
     outcomes), then queried for every row.

This guarantees no verifying information from validation or test periods
reaches the training features.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..analogues.engine import AnalogueEngine
from ..config import Config, load_config
from ..verification.bust import BustDefinition


def run(cfg: Config | None = None, verbose: bool = True) -> Path:
    cfg = cfg or load_config()
    table = cfg.path("features") / "training_table.parquet"
    df = pd.read_parquet(table)
    if verbose:
        print(f"  loaded {len(df):,} rows")

    train = df[df["split"] == "train"]
    if train.empty:
        raise RuntimeError("no rows in the training split - check configs/default.yaml splits")

    # 1-2: bust definition fitted on training rows only, then applied to all
    bust = BustDefinition.from_config(cfg).fit(train)
    df = bust.label(df)
    bust.save(cfg.path("models") / "bust_definition.json")
    if verbose:
        rates = df.groupby("split", observed=True)["bust_overall"].mean().round(4).to_dict()
        print(f"  bust definition fitted; overall bust rate by split: {rates}")

    # 3: analogue engine built from training rows only
    ana_cfg = cfg["analogues"]
    engine = AnalogueEngine(
        k=int(ana_cfg["n_neighbours"]),
        exclude_days=int(ana_cfg["exclude_same_cycle_window_days"]),
    )
    engine.fit(df[df["split"] == "train"])
    if verbose:
        print(f"  analogue index: {len(engine.ref_error):,} reference situations")

    chunks = []
    step = 40000
    for start in range(0, len(df), step):
        part = df.iloc[start:start + step]
        chunks.append(engine.transform(part))
        if verbose:
            print(f"  analogues: {min(start + step, len(df)):,}/{len(df):,}", flush=True)
    ana = pd.concat(chunks)
    df = pd.concat([df, ana], axis=1)
    engine.save(cfg.path("models") / "analogue_engine.joblib")

    out = cfg.path("labels") / "labelled_table.parquet"
    df.to_parquet(out, index=False)
    if verbose:
        print(f"  wrote {out} rows={len(df):,} cols={df.shape[1]}")
    return out
