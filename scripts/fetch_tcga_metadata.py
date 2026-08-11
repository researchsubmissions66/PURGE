import os
import json
import urllib.request
import pandas as pd

def main():
    feat_dir = '/work/hdd/bhwm/trident_features/master_benchmark/20x_224px_0px_overlap/features_hoptimus0/'
    print(f"Scanning {feat_dir}...")
    
    files = os.listdir(feat_dir)
    slide_ids = []
    patient_ids = []
    
    for f in files:
        if not f.endswith('.h5'):
            continue
        # Extract TCGA-XX-XXXX-XXX-XX-XXX from filename
        # TCGA-39-5028-01Z-00-DX1.7994ec22-746d-4c30-8138-e6c9bc67c71f.h5
        slide_id = f.split('.')[0]
        # Patient ID is first 12 characters: TCGA-XX-XXXX
        patient_id = slide_id[:12]
        slide_ids.append(slide_id)
        patient_ids.append(patient_id)
        
    unique_patients = list(set(patient_ids))
    print(f"Found {len(files)} total files across {len(unique_patients)} unique patients.")
    
    # Query GDC API
    print("Querying GDC API for project mappings...")
    url = 'https://api.gdc.cancer.gov/cases'
    headers = {'Content-Type': 'application/json'}
    
    # Batch query all patient IDs at once (up to 10k allowed)
    query = {
        "filters": {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": unique_patients
            }
        },
        "fields": "submitter_id,project.project_id",
        "format": "JSON",
        "size": 10000
    }
    
    req = urllib.request.Request(url, data=json.dumps(query).encode('utf-8'), headers=headers)
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read().decode('utf-8'))
    
    hits = data.get('data', {}).get('hits', [])
    print(f"GDC API returned {len(hits)} matched cases.")
    
    # Create mapping dictionary
    patient_to_project = {}
    for hit in hits:
        patient = hit.get('submitter_id')
        project = hit.get('project', {}).get('project_id')
        if patient and project:
            patient_to_project[patient] = project
            
    # Map project to organ
    organ_mapping = {
        'TCGA-LUAD': 'LUNG',
        'TCGA-LUSC': 'LUNG',
        'TCGA-BRCA': 'BRCA',
        'TCGA-KIRC': 'KIDNEY',
        'TCGA-KIRP': 'KIDNEY',
        'TCGA-KICH': 'KIDNEY'
    }
    
    records = []
    for f in files:
        if not f.endswith('.h5'):
            continue
        slide_id = f.split('.')[0]
        patient_id = slide_id[:12]
        
        project = patient_to_project.get(patient_id, 'UNKNOWN')
        organ = organ_mapping.get(project, project.replace('TCGA-', '')) # Default to project name if not mapped
        
        records.append({
            'slide_id': slide_id,
            'patient_id': patient_id,
            'filename': f,
            'dataset': 'TCGA',
            'organ': organ,
            'label': project.replace('TCGA-', '')
        })
        
    df = pd.DataFrame(records)
    out_file = 'data/metadata.csv'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    
    print(f"Successfully generated new metadata at {out_file}!")
    print("\nOrgan Counts:")
    print(df['organ'].value_counts())

if __name__ == "__main__":
    main()
