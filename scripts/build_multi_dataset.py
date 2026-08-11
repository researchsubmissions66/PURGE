import pandas as pd
import os

def get_organ(dataset, label):
    if dataset == "PANDA":
        return "PROSTATE"
    if dataset in ["BACH", "BRACS"]:
        return "BREAST"
    if dataset == "UBC-OCEAN":
        return "OVARIAN"
    if dataset == "TCGA":
        if str(label) == "BRCA":
            return "BREAST"
        if str(label) in ["LUAD", "LUSC"]:
            return "LUNG"
        if str(label) in ["KIRC", "KIRP", "KICH"]:
            return "KIDNEY"
    return "UNKNOWN"

def build():
    base = "/work/hdd/bhwm/metadata/RAW_DATA_Cleaned"
    datasets = {
        "PANDA": "/work/hdd/bhwm/PANDA/20x_224px_0px_overlap",
        "BACH": "/work/hdd/bhwm/BACH/20x_224px_0px_overlap",
        "UBC-OCEAN": "/work/hdd/bhwm/UBC-OCEAN/20x_224px_0px_overlap",
        "BRACS": "/work/hdd/bhwm/BRACS/20x_224px_0px_overlap",
    }
    
    dfs = []
    for d_name, root_path in datasets.items():
        meta_path = f"{base}/{d_name}/metadata.csv"
        df = pd.read_csv(meta_path)
        df['dataset'] = d_name
        df['root_dir'] = root_path
        dfs.append(df)
        
    # TCGA LUNG & BRCA
    tcga_df = pd.read_csv("data/metadata.csv")
    tcga_df['dataset'] = "TCGA"
    tcga_df['root_dir'] = "/work/hdd/bhwm/trident_features/master_benchmark/20x_224px_0px_overlap"
    dfs.append(tcga_df)
        
    master = pd.concat(dfs, ignore_index=True)
    
    # Map organ column
    master['organ'] = master.apply(lambda row: get_organ(row['dataset'], row['label']), axis=1)
    
    out = "data/multi_benchmark_metadata.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    master.to_csv(out, index=False)
    print(f"Built master metadata with {len(master)} slides across 5 datasets!")
    print("\nDataset Counts:")
    print(master['dataset'].value_counts())
    print("\nOrgan Counts:")
    print(master['organ'].value_counts())

if __name__ == "__main__":
    build()
