import os
import glob
import json
import requests

def query_gdc(cases):
    """Query GDC API to get project_id (LUAD vs LUSC) for a list of TCGA case IDs."""
    url = "https://api.gdc.cancer.gov/cases"
    results = {}
    
    chunk_size = 500
    for i in range(0, len(cases), chunk_size):
        chunk = cases[i:i+chunk_size]
        filters = {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": chunk
            }
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": "submitter_id,project.project_id",
            "format": "JSON",
            "size": "1000"
        }
        
        try:
            response = requests.post(url, json=params)
            response.raise_for_status()
            data = response.json()
            if "data" in data and "hits" in data["data"]:
                for hit in data["data"]["hits"]:
                    if "project" in hit and "project_id" in hit["project"]:
                        results[hit["submitter_id"]] = hit["project"]["project_id"]
        except Exception as e:
            print(f"Error querying GDC: {e}")
            
    return results

def build_metadata():
    base_dir = "/work/hdd/bhwm/trident_features/master_benchmark/20x_224px_0px_overlap/features_gpfm"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return
        
    all_files = glob.glob(os.path.join(base_dir, "*.h5")) + glob.glob(os.path.join(base_dir, "*.pt"))
    print(f"Found {len(all_files)} total feature files.")
    
    tcga_cases = []
    slide_records = []
    
    for f in all_files:
        basename = os.path.basename(f)
        slide_id = basename.split('.')[0]
        
        organ = None
        dataset = None
        patient_id = None
        
        if slide_id.startswith("TCGA"):
            parts = slide_id.split("-")
            if len(parts) >= 3:
                patient_id = "-".join(parts[0:3])
                tcga_cases.append(patient_id)
                dataset = "TCGA"
        
        slide_records.append({
            "slide_id": slide_id,
            "patient_id": patient_id,
            "filename": basename,
            "dataset": dataset
        })
        
    tcga_cases = list(set(tcga_cases))
    print(f"Querying GDC for {len(tcga_cases)} unique TCGA cases...")
    tcga_mapping = query_gdc(tcga_cases)
    
    final_records = []
    for rec in slide_records:
        pid = rec["patient_id"]
        if not pid:
            continue
            
        organ = None
        label = None
        
        if rec["dataset"] == "TCGA":
            project_id = tcga_mapping.get(pid, "UNKNOWN")
            if project_id in ["TCGA-LUAD", "TCGA-LUSC"]:
                organ = "LUNG"
                label = project_id.replace("TCGA-", "")
            elif project_id in ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"]:
                organ = "KIDNEY"
                label = project_id.replace("TCGA-", "")
                
        if organ:
            rec["organ"] = organ
            rec["label"] = label
            final_records.append(rec)
            
    import csv
    out_csv = "/u/dchanda/PURGE/data/metadata.csv"
    with open(out_csv, 'w', newline='') as f:
        if final_records:
            writer = csv.DictWriter(f, fieldnames=final_records[0].keys())
            writer.writeheader()
            writer.writerows(final_records)
    print(f"Wrote {len(final_records)} records to {out_csv}")

if __name__ == "__main__":
    build_metadata()
