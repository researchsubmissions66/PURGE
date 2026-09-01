"""
Merge the per-benchmark metadata files into data/multi_benchmark_metadata.csv.

The 'organ' vocabulary here is canonical for the whole project - PROSTATE,
BREAST, LUNG, OVARIAN, KIDNEY. scripts/fetch_tcga_metadata.py must emit the same
names, otherwise anything filtering on organ selects an empty cohort in silence.
"""

import os
import sys

import pandas as pd

BASE = "/work/hdd/bhwm/metadata/RAW_DATA_Cleaned"
DATASETS = {
    "PANDA": "/work/hdd/bhwm/PANDA/20x_224px_0px_overlap",
    "BACH": "/work/hdd/bhwm/BACH/20x_224px_0px_overlap",
    "UBC-OCEAN": "/work/hdd/bhwm/UBC-OCEAN/20x_224px_0px_overlap",
    "BRACS": "/work/hdd/bhwm/BRACS/20x_224px_0px_overlap",
}
TCGA_ROOT = "/work/hdd/bhwm/trident_features/master_benchmark/20x_224px_0px_overlap"
TCGA_METADATA = "data/metadata.csv"
OUTPUT = "data/multi_benchmark_metadata.csv"

TCGA_LABEL_TO_ORGAN = {
    "BRCA": "BREAST", "IDC": "BREAST", "ILC": "BREAST",
    "LUAD": "LUNG", "LUSC": "LUNG",
    "KIRC": "KIDNEY", "KIRP": "KIDNEY", "KICH": "KIDNEY",
}
DATASET_TO_ORGAN = {
    "PANDA": "PROSTATE",
    "BACH": "BREAST",
    "BRACS": "BREAST",
    "UBC-OCEAN": "OVARIAN",
}


def get_organ(dataset, label):
    if dataset in DATASET_TO_ORGAN:
        return DATASET_TO_ORGAN[dataset]
    if dataset == "TCGA":
        return TCGA_LABEL_TO_ORGAN.get(str(label), "UNKNOWN")
    return "UNKNOWN"


def build():
    dfs = []
    for name, root in DATASETS.items():
        meta_path = f"{BASE}/{name}/metadata.csv"
        if not os.path.exists(meta_path):
            raise SystemExit(f"Missing metadata for {name}: {meta_path}")
        df = pd.read_csv(meta_path)
        df['dataset'] = name
        df['root_dir'] = root
        dfs.append(df)

    if not os.path.exists(TCGA_METADATA):
        raise SystemExit(
            f"Missing {TCGA_METADATA}. Run scripts/fetch_tcga_metadata.py first."
        )
    tcga_df = pd.read_csv(TCGA_METADATA)
    tcga_df['dataset'] = "TCGA"
    tcga_df['root_dir'] = TCGA_ROOT
    dfs.append(tcga_df)

    master = pd.concat(dfs, ignore_index=True)
    master['organ'] = [get_organ(d, l) for d, l in zip(master['dataset'], master['label'])]

    unknown = master[master['organ'] == "UNKNOWN"]
    if len(unknown):
        combos = unknown[['dataset', 'label']].drop_duplicates().to_dict('records')
        print(f"WARNING: {len(unknown)} rows have organ=UNKNOWN: {combos}", file=sys.stderr)

    for col in ('filename', 'label', 'patient_id'):
        if master[col].isna().any():
            raise SystemExit(f"Column '{col}' has missing values; refusing to write {OUTPUT}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    master.to_csv(OUTPUT, index=False)
    print(f"Built {OUTPUT} with {len(master)} slides.")
    print("\nDataset counts:\n", master['dataset'].value_counts().to_string())
    print("\nOrgan counts:\n", master['organ'].value_counts().to_string())


if __name__ == "__main__":
    build()
