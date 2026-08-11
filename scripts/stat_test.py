import os
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product

def load_results():
    results_dir = 'results'
    all_files = glob.glob(os.path.join(results_dir, '*.json'))
    data = []
    
    for f in all_files:
        basename = os.path.basename(f)
        try:
            with open(f, 'r') as file:
                content = json.load(file)
                val_auc = content.get('best_val_auc', 0.0)
        except:
            continue
            
        parts = basename.replace('.json', '').split('_')
        condition = parts[0]
        
        if condition == "unlearned":
            organ_removed = parts[1]
            dataset = parts[3]
            condition_str = f"-{organ_removed}"
            start_encoder_idx = 4
        else:
            dataset = parts[1]
            condition_str = "Baseline"
            start_encoder_idx = 2
        
        fold_idx = len(parts) - 1
        for i, p in enumerate(parts):
            if p.startswith('fold'):
                fold_idx = i
                break
                
        encoder = "_".join(parts[start_encoder_idx:fold_idx])
        fold = int(parts[fold_idx].replace('fold', ''))
        
        data.append({
            'dataset': dataset,
            'encoder': encoder,
            'condition': condition_str,
            'fold': fold,
            'auc': val_auc
        })
        
    return pd.DataFrame(data)

def main():
    df = load_results()
    
    datasets = sorted(df['dataset'].unique())
    encoders = sorted(df['encoder'].unique())
    organs = ['PROSTATE', 'BREAST', 'LUNG', 'OVARIAN']
    
    # Map datasets to their "matching" organ
    dataset_organ_map = {
        'PANDA': 'PROSTATE',
        'BRACS': 'BREAST',
        'BACH': 'BREAST',
        'TCGA': 'LUNG',
        'UBC-OCEAN': 'OVARIAN',
    }
    
    print("=" * 100)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("Paired t-test (2-tailed) + Wilcoxon signed-rank test across 5 folds")
    print("Comparing: Baseline AUC vs. Unlearned AUC for each (Dataset, Encoder, Organ-Removed)")
    print("=" * 100)
    
    all_rows = []
    
    for dataset in datasets:
        for encoder in encoders:
            baseline = df[(df['dataset'] == dataset) & 
                         (df['encoder'] == encoder) & 
                         (df['condition'] == 'Baseline')].sort_values('fold')
            
            if len(baseline) < 5:
                continue
                
            baseline_aucs = baseline['auc'].values
            
            for organ in organs:
                cond = f"-{organ}"
                unlearned = df[(df['dataset'] == dataset) & 
                              (df['encoder'] == encoder) & 
                              (df['condition'] == cond)].sort_values('fold')
                
                if len(unlearned) < 5:
                    continue
                    
                unlearned_aucs = unlearned['auc'].values
                
                # Paired differences
                diffs = baseline_aucs - unlearned_aucs
                mean_diff = np.mean(diffs)
                
                # Paired t-test (2-tailed)
                t_stat, p_ttest = stats.ttest_rel(baseline_aucs, unlearned_aucs)
                
                # Wilcoxon signed-rank (non-parametric, more robust with n=5)
                try:
                    w_stat, p_wilcoxon = stats.wilcoxon(baseline_aucs, unlearned_aucs)
                except ValueError:
                    # All differences are zero
                    w_stat, p_wilcoxon = 0, 1.0
                
                # Is this the "matching" organ for this dataset?
                matching = dataset_organ_map.get(dataset, '') == organ
                
                sig_t = "***" if p_ttest < 0.001 else "**" if p_ttest < 0.01 else "*" if p_ttest < 0.05 else "ns"
                sig_w = "***" if p_wilcoxon < 0.001 else "**" if p_wilcoxon < 0.01 else "*" if p_wilcoxon < 0.05 else "ns"
                
                all_rows.append({
                    'Dataset': dataset,
                    'Encoder': encoder,
                    'Organ Removed': organ,
                    'Matching': '✓' if matching else '',
                    'Baseline (mean)': f"{np.mean(baseline_aucs):.4f}",
                    'Unlearned (mean)': f"{np.mean(unlearned_aucs):.4f}",
                    'Δ AUC': f"{mean_diff:+.4f}",
                    'p (t-test)': f"{p_ttest:.4f}",
                    'Sig (t)': sig_t,
                    'p (Wilcoxon)': f"{p_wilcoxon:.4f}",
                    'Sig (W)': sig_w,
                })
    
    result_df = pd.DataFrame(all_rows)
    
    # Print full table
    print()
    print(result_df.to_string(index=False))
    
    # Summary
    print()
    print("=" * 100)
    print("SUMMARY: Only 'Matching' organ removals should show significant degradation")
    print("=" * 100)
    
    matching_rows = result_df[result_df['Matching'] == '✓']
    non_matching_rows = result_df[result_df['Matching'] == '']
    
    print(f"\nMatching organ removals (should be significant):")
    print(f"  Total comparisons: {len(matching_rows)}")
    print(f"  Significant (t-test p < 0.05): {sum(matching_rows['Sig (t)'] != 'ns')}/{len(matching_rows)}")
    print(f"  Significant (Wilcoxon p < 0.05): {sum(matching_rows['Sig (W)'] != 'ns')}/{len(matching_rows)}")
    
    print(f"\nNon-matching organ removals (should NOT be significant):")
    print(f"  Total comparisons: {len(non_matching_rows)}")
    print(f"  Significant (t-test p < 0.05): {sum(non_matching_rows['Sig (t)'] != 'ns')}/{len(non_matching_rows)}")
    print(f"  Significant (Wilcoxon p < 0.05): {sum(non_matching_rows['Sig (W)'] != 'ns')}/{len(non_matching_rows)}")

if __name__ == "__main__":
    main()
