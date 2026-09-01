"""Inventory which feature extractions exist on scratch, by cohort and magnification."""

import argparse
import os
import pandas as pd
from collections import defaultdict

# 1. Build Metadata Map
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
parser.add_argument('--base_dir', default='/work/hdd/bhwm')
args = parser.parse_args()

df = pd.read_csv(args.metadata)

# Map filename (.h5) to cohort
# We only care about BRCA, NSCLC, RCC for TCGA. UBC-OCEAN and CAMELYON16 are physically separated.
file_to_cohort = {}
for idx, row in df.iterrows():
    if row['dataset'] == 'TCGA':
        label = row['label']
        if label in ['IDC', 'ILC']:
            cohort = 'BRCA'
        elif label in ['LUAD', 'LUSC']:
            cohort = 'NSCLC'
        elif label in ['KIRC', 'KIRP', 'KICH']:
            cohort = 'RCC'
        else:
            cohort = 'OTHER_TCGA'
        file_to_cohort[row['filename']] = cohort

# 2. Define Directories to Scan
base_dirs = {
    'TCGA': f'{args.base_dir}/trident_features/master_benchmark',
    'UBC-OCEAN': f'{args.base_dir}/UBC-OCEAN',
    'CAMELYON16': f'{args.base_dir}/CAMELYON16',
}

# structure: stats[cohort][magnification][feature_name] = count
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for root_ds, ds_path in base_dirs.items():
    if not os.path.exists(ds_path):
        continue
    
    for mag_dir in os.listdir(ds_path):
        if not mag_dir.endswith('_overlap'):
            continue
        mag = mag_dir.split('_')[0] # '5x', '10x', '20x'
        
        mag_path = os.path.join(ds_path, mag_dir)
        for feat_dir in os.listdir(mag_path):
            if not feat_dir.startswith('features_'):
                continue
            feat_name = feat_dir.replace('features_', '')
            feat_path = os.path.join(mag_path, feat_dir)
            
            if os.path.isdir(feat_path):
                # Count files
                for f in os.listdir(feat_path):
                    if f.endswith('.pt') or f.endswith('.h5'):
                        if root_ds == 'TCGA':
                            h5_name = f
                            if f.endswith('.pt'):
                                h5_name = f.replace('.pt', '.h5')
                            cohort = file_to_cohort.get(h5_name, 'OTHER_TCGA')
                        else:
                            cohort = root_ds
                        
                        stats[cohort][mag][feat_name] += 1

# 3. Print Summary Table
print(f"{'Cohort':<12} | {'Magnification':<15} | {'Feature':<20} | {'Slides'}")
print("-" * 60)

for cohort in ['BRCA', 'NSCLC', 'RCC', 'UBC-OCEAN', 'CAMELYON16']:
    if cohort not in stats:
        continue
    for mag in sorted(stats[cohort].keys()):
        for feat in sorted(stats[cohort][mag].keys()):
            count = stats[cohort][mag][feat]
            print(f"{cohort:<12} | {mag:<15} | {feat:<20} | {count}")
    print("-" * 60)

