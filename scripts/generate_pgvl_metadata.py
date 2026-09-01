"""Export PURGE cohorts in the CSV layout PGVL-Gym expects."""

import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
parser.add_argument('--output_dir', default='../PGVL-Gym/metadata')
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(args.metadata)

# PGVL-Gym expects a column 'OncoTreeCode' for TCGA.
# Our label column contains LUAD, LUSC, IDC, ILC etc.
df['OncoTreeCode'] = df['label']
# Make sure we have case_id
if 'patient_id' in df.columns:
    df['case_id'] = df['patient_id']

# TCGA-BRCA
brca_df = df[(df['dataset'] == 'TCGA') & (df['organ'] == 'BREAST') & (df['label'].isin(['IDC', 'ILC']))]
brca_df.to_csv(os.path.join(output_dir, 'tcga_brca.csv'), index=False)
print(f"Generated tcga_brca.csv with {len(brca_df)} rows")

# TCGA-NSCLC (LUNG)
nsclc_df = df[(df['dataset'] == 'TCGA') & (df['organ'] == 'LUNG') & (df['label'].isin(['LUAD', 'LUSC']))]
nsclc_df.to_csv(os.path.join(output_dir, 'tcga_nsclc.csv'), index=False)
print(f"Generated tcga_nsclc.csv with {len(nsclc_df)} rows")

# UBC-OCEAN
# For UBC-OCEAN, PGVL-Gym uses 'image_id' as slide_id and case_id
ubc_df = df[df['dataset'] == 'UBC-OCEAN'].copy()
ubc_df['image_id'] = ubc_df['slide_id']
ubc_df.to_csv(os.path.join(output_dir, 'ubc_ocean.csv'), index=False)
print(f"Generated ubc_ocean.csv with {len(ubc_df)} rows")
