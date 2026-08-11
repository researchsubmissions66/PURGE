import os
import json
import glob
import pandas as pd

def main():
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return
        
    all_files = glob.glob(os.path.join(results_dir, '*.json'))
    data = []
    
    for f in all_files:
        basename = os.path.basename(f)
        try:
            with open(f, 'r') as file:
                content = json.load(file)
                val_auc = content.get('best_val_auc', 0.0)
        except Exception as e:
            continue
            
        # Parse baseline_DATASET_ENCODER_foldFOLD.json
        # e.g. baseline_PANDA_features_hoptimus0_fold0.json
        parts = basename.replace('.json', '').split('_')
        
        condition = parts[0] # baseline or unlearned
        
        if condition == "unlearned":
            # unlearned_ORGAN_on_DATASET_ENCODER_fold...
            # parts = ["unlearned", "PROSTATE", "on", "PANDA", "features", "hoptimus0", "fold0"]
            organ_removed = parts[1]
            dataset = parts[3]
            condition_str = f"-{organ_removed}"
            start_encoder_idx = 4
        else:
            # baseline_PANDA_features_hoptimus0_fold0.json
            dataset = parts[1]
            condition_str = "Baseline"
            start_encoder_idx = 2
        
        # Encoder might have underscores: "features", "hoptimus0"
        # Find fold part
        fold_idx = len(parts) - 1
        for i, p in enumerate(parts):
            if p.startswith('fold'):
                fold_idx = i
                break
                
        encoder = "_".join(parts[start_encoder_idx:fold_idx])
        fold = parts[fold_idx].replace('fold', '')
        
        data.append({
            'dataset': dataset,
            'encoder': encoder,
            'condition': condition_str,
            'fold': int(fold),
            'auc': val_auc
        })
        
    if not data:
        print("No result JSON files found.")
        return
        
    df = pd.DataFrame(data)
    
    # Calculate Mean and Std AUC for each Dataset + Encoder + Condition
    grouped = df.groupby(['dataset', 'encoder', 'condition']).agg(
        mean_auc=('auc', 'mean'),
        std_auc=('auc', 'std'),
        count=('auc', 'count')
    ).reset_index()
    
    # Sort for nice display (Baseline first)
    grouped['is_baseline'] = grouped['condition'] == 'Baseline'
    grouped = grouped.sort_values(['dataset', 'encoder', 'is_baseline', 'condition'], ascending=[True, True, False, True]).drop('is_baseline', axis=1)
    
    print("## Aggregated Results\n")
    print("| Dataset | Encoder | Condition | Mean AUC ± Std |")
    print("|---------|---------|-----------|----------------|")
    
    for _, row in grouped.iterrows():
        count = int(row['count'])
        mean = row['mean_auc']
        std = row['std_auc']
        
        if count < 5:
            # If incomplete
            status = f"{mean:.4f} ± {std:.4f} (Incomplete: {count}/5)"
            if pd.isna(std):
                status = f"{mean:.4f} (Incomplete: {count}/5)"
        else:
            status = f"{mean:.4f} ± {std:.4f}"
            
        print(f"| {row['dataset']} | {row['encoder']} | {row['condition']} | {status} |")

if __name__ == "__main__":
    main()
