import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datasets.feature_dataset import FeatureDataset
from src.models.abmil import ABMIL
from src.models.meanmil import MeanMIL
from src.models.transmil import TransMIL
from src.unlearning.subspace import remove_subspace
from src.unlearning.ot_transport import apply_ot_unlearning
from src.unlearning.wiener_filter import apply_wiener_filter
from src.unlearning.noise import apply_gaussian_noise, apply_dropout

def train(model, dataloader, optimizer, criterion, device, unlearner_w=None, unlearn_method='svd'):
    model.train()
    total_loss = 0
    for features, label, _ in dataloader:
        features = features.squeeze(0).to(device) # batch size 1
        label = label.to(device)
        
        if unlearner_w is not None:
            if unlearn_method == 'svd':
                features = remove_subspace(features, unlearner_w)
            elif unlearn_method == 'ot':
                features = apply_ot_unlearning(features, unlearner_w['W'], unlearner_w['mu_organ'], unlearner_w['mu_global'])
            elif unlearn_method == 'wiener':
                features = apply_wiener_filter(features, unlearner_w['V'], unlearner_w['lambdas'], tau=unlearner_w['tau'])
            elif unlearn_method == 'gaussian':
                features = apply_gaussian_noise(features, sigma=unlearner_w['sigma'])
            elif unlearn_method == 'dropout':
                features = apply_dropout(features, p=unlearner_w['dropout_p'])
            
        optimizer.zero_grad()
        logits, _ = model(features)
        loss = criterion(logits.unsqueeze(0), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, unlearner_w=None, num_classes=2, unlearn_method='svd'):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for features, label, _ in dataloader:
            features = features.squeeze(0).to(device)
            if unlearner_w is not None:
                if unlearn_method == 'svd':
                    features = remove_subspace(features, unlearner_w)
                elif unlearn_method == 'ot':
                    features = apply_ot_unlearning(features, unlearner_w['W'], unlearner_w['mu_organ'], unlearner_w['mu_global'])
                elif unlearn_method == 'wiener':
                    features = apply_wiener_filter(features, unlearner_w['V'], unlearner_w['lambdas'], tau=unlearner_w['tau'])
                elif unlearn_method == 'gaussian':
                    features = apply_gaussian_noise(features, sigma=unlearner_w['sigma'])
                elif unlearn_method == 'dropout':
                    features = apply_dropout(features, p=unlearner_w['dropout_p'])
                
            logits, _ = model(features)
            probs = torch.softmax(logits, dim=0)
            
            if num_classes == 2:
                all_probs.append(probs[1].item())
            else:
                all_probs.append(probs.cpu().numpy())
            all_labels.append(label.item())
            
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            all_probs = np.array(all_probs)
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        auc = 0.5 # Default if validation split misses a class
    return auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_dir', required=True)
    parser.add_argument('--dataset', required=True, help="PANDA, BACH, BRACS, or UBC-OCEAN")
    parser.add_argument('--unlearner', type=str, default=None)
    parser.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--k', type=int, default=None, help="Number of subspace components to remove")
    parser.add_argument('--tau', type=float, default=0.1, help="Regularization parameter for Wiener filter")
    parser.add_argument('--sigma', type=float, default=1.0, help="Standard deviation for Gaussian Noise")
    parser.add_argument('--dropout_p', type=float, default=0.5, help="Probability for Feature Dropout")
    parser.add_argument('--model_type', type=str, default='abmil', choices=['abmil', 'meanmil', 'transmil'])
    parser.add_argument('--unlearn_method', type=str, default='svd', choices=['svd', 'ot', 'wiener', 'gaussian', 'dropout'])
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    df_target = df[df['dataset'] == args.dataset].copy()
    
    patients = df_target['patient_id'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(patients))
    
    train_idx, val_idx = splits[args.fold]
    train_patients = patients[train_idx]
    val_patients = patients[val_idx]
    
    train_dataset = FeatureDataset(args.metadata, args.encoder_dir, split_patients=train_patients, dataset_target=args.dataset)
    val_dataset = FeatureDataset(args.metadata, args.encoder_dir, split_patients=val_patients, dataset_target=args.dataset)
    
    num_classes = train_dataset.num_classes
    print(f"Initializing ABMIL for {args.dataset} with {num_classes} classes.")
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    unlearner_w = None
    if args.unlearn_method == 'gaussian':
        unlearner_w = {'sigma': args.sigma}
        print(f"Applying Gaussian Noise Baseline with sigma={args.sigma}")
    elif args.unlearn_method == 'dropout':
        unlearner_w = {'dropout_p': args.dropout_p}
        print(f"Applying Feature Dropout Baseline with p={args.dropout_p}")
    elif args.unlearner and os.path.exists(args.unlearner):
        if args.unlearn_method == 'ot':
            unlearner_w = torch.load(args.unlearner, map_location=device)
            print(f"Loaded OT unlearner from {args.unlearner}")
        elif args.unlearn_method == 'wiener':
            unlearner_dict = torch.load(args.unlearner, map_location=device)
            # Add tau to the dict so it can be passed easily
            unlearner_dict['tau'] = args.tau
            unlearner_w = unlearner_dict
            print(f"Loaded Wiener unlearner from {args.unlearner} with tau={args.tau}")
        else:
            unlearner_w = torch.load(args.unlearner).to(device)
            if args.k is not None:
                # Dynamically truncate the pre-computed subspace
                actual_k = min(args.k, unlearner_w.shape[1])
                unlearner_w = unlearner_w[:, :actual_k]
            print(f"Loaded SVD unlearner from {args.unlearner} with shape {unlearner_w.shape}")

    sample_feat, _, _ = train_dataset[0]
    input_dim = sample_feat.shape[1]
    
    if args.model_type == 'abmil':
        model = ABMIL(input_dim=input_dim, num_classes=num_classes).to(device)
    elif args.model_type == 'meanmil':
        model = MeanMIL(input_dim=input_dim, num_classes=num_classes).to(device)
    elif args.model_type == 'transmil':
        model = TransMIL(input_dim=input_dim, num_classes=num_classes).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_auc = 0
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, criterion, device, unlearner_w, args.unlearn_method)
        auc = evaluate(model, val_loader, device, unlearner_w, num_classes, args.unlearn_method)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}, Val AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break
            
    print(f"Finished Fold {args.fold} with Best AUC: {best_auc:.4f}")
    
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        import json
        with open(args.output_json, 'w') as f:
            json.dump({
                'fold': args.fold, 
                'best_val_auc': best_auc, 
                'num_classes': num_classes,
                'model_type': args.model_type
            }, f)

if __name__ == "__main__":
    main()
