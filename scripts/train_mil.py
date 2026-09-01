"""
Train a slide-level MIL probe, optionally through a fitted eraser.

Evaluation protocol (plan section 15): the outer fold's held-out patients are the
TEST set. Early stopping and checkpoint selection watch an inner validation split
carved out of the training patients, so the reported test AUC is not selected on
the split it is measured on.

Every AUC is macro one-vs-rest, whose chance level is 0.50 for any number of
classes. A successful erasure drives the target to ~0.50.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datasets.feature_dataset import FeatureDataset, build_label_map
from src.evaluation.metrics import macro_ovr_auc
from src.models.factory import MIL_MODELS, build_mil
from src.unlearning.apply import METHODS, build_eraser
from src.utils.splits import inner_split, patient_folds


def run_epoch(model, dataloader, optimizer, criterion, device, erase=None):
    model.train()
    total_loss = 0.0
    for features, label, _ in dataloader:
        z = features.squeeze(0).to(device)
        if erase is not None:
            with torch.no_grad():
                z = erase(z)
        optimizer.zero_grad()
        logits, _ = model(z)
        loss = criterion(logits.unsqueeze(0), label.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, device, num_classes, erase=None):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for features, label, _ in dataloader:
            z = features.squeeze(0).to(device)
            if erase is not None:
                z = erase(z)
            logits, _ = model(z)
            probs = torch.softmax(logits, dim=0)
            probs_all.append(probs[1].item() if num_classes == 2 else probs.cpu().numpy())
            labels_all.append(label.item())

    # strict=False so a degenerate split reports None with a reason rather than a
    # placeholder 0.5, which would be indistinguishable from a real chance result.
    return macro_ovr_auc(np.asarray(labels_all), np.asarray(probs_all),
                         num_classes, strict=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--encoder_dir', required=True)
    parser.add_argument('--dataset', required=True,
                        help="PANDA, BACH, BRACS, UBC-OCEAN, TCGA-LUNG or TCGA-BRCA")
    parser.add_argument('--metadata', default='data/multi_benchmark_metadata.csv')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default=None)
    parser.add_argument('--model_type', choices=MIL_MODELS, default='abmil')

    parser.add_argument('--unlearn_method', choices=METHODS, default='none')
    parser.add_argument('--unlearner', type=str, default=None,
                        help="path to the fitted eraser")
    parser.add_argument('--k', type=int, default=None,
                        help="subspace dimensions to remove (svd) / eraser rank")
    parser.add_argument('--sigma', type=float, default=1.0, help="Gaussian noise sigma")
    parser.add_argument('--dropout_p', type=float, default=0.5, help="feature dropout p")
    parser.add_argument('--no_svd_affine', action='store_true',
                        help="svd: use the plain projection instead of the "
                             "mean-preserving default (ablation only - same "
                             "erasure, far more embedding distortion)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Label map spans the whole benchmark so every split agrees on class indices.
    label_map = build_label_map(args.metadata, args.dataset)
    outer_train_patients, test_patients = patient_folds(args.metadata, args.dataset, args.fold)
    train_patients, val_patients = inner_split(outer_train_patients)

    train_ds = FeatureDataset(args.metadata, args.encoder_dir, train_patients,
                              args.dataset, label_map)
    val_ds = FeatureDataset(args.metadata, args.encoder_dir, val_patients,
                            args.dataset, label_map)
    test_ds = FeatureDataset(args.metadata, args.encoder_dir, test_patients,
                             args.dataset, label_map)

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError(
            f"Empty split for {args.dataset} "
            f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}). "
            f"Check features under {args.encoder_dir}."
        )

    num_classes = train_ds.num_classes
    input_dim = train_ds[0][0].shape[1]
    print(f"{args.dataset}: {num_classes} classes, d={input_dim} | "
          f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test slides")

    loader_kw = dict(batch_size=1, num_workers=args.num_workers,
                     prefetch_factor=2 if args.num_workers else None)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    erase, _ = build_eraser(
        args.unlearn_method, path=args.unlearner, device=device, input_dim=input_dim,
        k=args.k, sigma=args.sigma, dropout_p=args.dropout_p, rank=args.k,
        affine=not args.no_svd_affine,
    )
    if erase is None:
        print("No erasure applied (baseline).")
    else:
        print(f"Applying '{args.unlearn_method}' erasure"
              + (f" from {args.unlearner}" if args.unlearner else ""))

    model = build_mil(args.model_type, input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val, best_state, best_epoch, stale = -np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, train_loader, optimizer, criterion, device, erase)
        val_auc, note = evaluate(model, val_loader, device, num_classes, erase)
        shown = 'n/a' if val_auc is None else f"{val_auc:.4f}"
        print(f"Epoch {epoch} - loss {loss:.4f}, inner-val AUC {shown}"
              + (f" [{note}]" if note else ""), flush=True)

        if val_auc is not None and val_auc > best_val:
            best_val, best_epoch, stale = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {stale} epochs without improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("WARNING: inner validation AUC was never defined; reporting the last epoch.")

    test_auc, test_note = evaluate(model, test_loader, device, num_classes, erase)
    shown = 'n/a' if test_auc is None else f"{test_auc:.4f}"
    val_shown = 'n/a' if best_val == -np.inf else f"{best_val:.4f}"
    print(f"\nFold {args.fold} | inner-val AUC {val_shown} (epoch {best_epoch}) "
          f"| TEST AUC {shown}" + (f" [{test_note}]" if test_note else ""))
    print("Chance level for macro one-vs-rest AUC is 0.50.")

    if args.save_model_path:
        parent = os.path.dirname(args.save_model_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(model.state_dict(), args.save_model_path)

    if args.output_json:
        parent = os.path.dirname(args.output_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump({
                'dataset': args.dataset,
                'fold': args.fold,
                'test_auc': test_auc,
                'val_auc': None if best_val == -np.inf else best_val,
                'best_epoch': best_epoch,
                'num_classes': num_classes,
                'model_type': args.model_type,
                'unlearn_method': args.unlearn_method,
                'unlearner': args.unlearner,
                'k': args.k,
                'n_train': len(train_ds), 'n_val': len(val_ds), 'n_test': len(test_ds),
                'test_note': test_note,
            }, f, indent=2)


if __name__ == "__main__":
    main()
