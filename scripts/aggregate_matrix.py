import os
import glob
import json
import numpy as np
import pandas as pd

def main():
    results = []
    
    # Only pull the baseline sweep files (which are in the format results/sweep_ORGAN_on_DATASET_fold.json)
    # Actually, we also have results/PANDA_abmil_features_hoptimus0_fold0.json etc for baseline.
    files = glob.glob("results/*.json")
    
    for f in files:
        if "ksweep" in f or "otsweep" in f or "wienersweep" in f or "modelsweep" in f:
            continue
            
        try:
            with open(f, 'r') as file:
                data = json.load(file)
        except:
            continue
            
        dataset = data.get('dataset', 'Unknown')
        auc = data.get('best_val_auc', 0.5)
        
        # Determine if this was baseline or erased
        fname = os.path.basename(f)
        if fname.startswith("unlearned_"):
            parts = fname.replace("unlearned_", "").split("_on_")
            organ_erased = parts[0]
        elif fname.startswith("baseline_"):
            organ_erased = "None (Baseline)"
        else:
            continue
            
        results.append({
            'Dataset': dataset,
            'Erased Concept': organ_erased,
            'AUC': auc
        })
        
    if not results:
        print("No results found.")
        return
        
    df = pd.DataFrame(results)
    
    # Calculate Mean AUC across folds
    pivot = df.groupby(['Dataset', 'Erased Concept'])['AUC'].mean().unstack(fill_value=np.nan)
    
    print("\n=== PURGE CROSS-EVALUATION MATRIX (AUC) ===")
    print(pivot.round(3).to_string())

if __name__ == "__main__":
    main()
