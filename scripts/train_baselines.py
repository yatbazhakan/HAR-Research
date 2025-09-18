"""
CRITICAL: Main Training Script for HAR Baseline Models

This script is the primary entry point for training Human Activity Recognition models.
It handles the complete training pipeline including data loading, model training,
evaluation, and result visualization.

Key Features:
- Supports multiple model architectures (CNN-TCN, CNN-BiLSTM)
- Multiple cross-validation strategies (holdout, k-fold, LOSO)
- Comprehensive evaluation metrics and visualization
- Weights & Biases integration for experiment tracking
- Model calibration and uncertainty estimation

Usage:
    python scripts/train_baselines.py --dataset uci_har --model cnn_tcn --cv holdout

CRITICAL: This script expects preprocessed data in NPZ shard format.
Run preprocessing scripts first to generate the required data files.
"""

from __future__ import annotations
import argparse, json, sys, glob, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# CRITICAL: Add repository root to Python path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CRITICAL: Import core HAR modules
from har.datasets.shards import NPZShardsDataset  # Dataset loading
from har.transforms.stats_io import load_stats    # Normalization statistics
from har.models.cnn_tcn import CNN_TCN           # CNN-TCN model
from har.models.cnn_bilstm import CNN_BiLSTM     # CNN-BiLSTM model
from har.train.metrics import TemperatureScaler  # Model calibration

# CRITICAL: Optional imports with graceful fallbacks
# These libraries enhance functionality but are not strictly required

try:
    import wandb  # type: ignore
    # Weights & Biases for experiment tracking and visualization
except Exception:
    wandb = None

try:
    import torchmetrics  # type: ignore
    from torchmetrics.classification import (
        MulticlassAccuracy,      # Multi-class accuracy metric
        MulticlassF1Score,       # F1 score for multi-class
        MulticlassPrecision,     # Precision for multi-class
        MulticlassRecall,        # Recall for multi-class
        MulticlassAUROC,         # Area Under ROC Curve
        MulticlassCalibrationError,  # Calibration error metric
        MulticlassConfusionMatrix,   # Confusion matrix
    )
    # CRITICAL: torchmetrics provides efficient, GPU-accelerated metrics
except Exception:
    torchmetrics = None

try:
    from tqdm import tqdm  # type: ignore
    # Progress bars for training loops
except Exception:
    tqdm = None

try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    from sklearn.calibration import calibration_curve
    has_matplotlib = True
    # CRITICAL: Visualization libraries for plots and charts
except Exception:
    has_matplotlib = False
    plt = None
    sns = None


def collate(batch):
    xs, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0))
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            pred = p.argmax(axis=1)
            probs.append(p)
            preds.append(pred)
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    P = np.concatenate(probs)
    return {
        "acc": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "ece": expected_calibration_error(P, y_true),
    }, y_true, P


def main():
    """
    CRITICAL: Main training function for HAR baseline models
    
    This function orchestrates the entire training pipeline:
    1. Parse command-line arguments
    2. Set up device and optimization settings
    3. Load and preprocess data
    4. Create model and training components
    5. Train the model
    6. Evaluate and visualize results
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # CRITICAL: Command-line argument parsing
    # These arguments control all aspects of the training process
    ap = argparse.ArgumentParser()
    
    # CRITICAL: Data loading arguments
    ap.add_argument("--shards_glob", type=str, required=True, 
                   help="Glob pattern for NPZ shard files (e.g., 'data/*.npz')")
    ap.add_argument("--fold_json", type=str, default="", 
                   help="LOSO JSON file path; if empty, use --cv mode")
    ap.add_argument("--stats", type=str, required=True,
                   help="Path to normalization statistics JSON file")
    
    # CRITICAL: Model configuration
    ap.add_argument("--model", type=str, choices=["cnn_tcn", "cnn_bilstm"], default="cnn_tcn",
                   help="Model architecture to train")
    ap.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=128,
                   help="Training batch size")
    ap.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for optimizer")
    
    # CRITICAL: Training options
    ap.add_argument("--calibrate", action="store_true",
                   help="Enable temperature scaling for model calibration")
    ap.add_argument("--amp", action="store_true", 
                   help="Enable mixed precision training on CUDA")
    ap.add_argument("--num_workers", type=int, default=4,
                   help="Number of data loading workers")
    
    # CRITICAL: Experiment tracking
    ap.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="har-baselines",
                   help="W&B project name")
    ap.add_argument("--wandb_run", type=str, default="",
                   help="W&B run name (auto-generated if empty)")
    
    # CRITICAL: Cross-validation options
    ap.add_argument("--cv", type=str, choices=["fold_json", "holdout", "kfold"], default="fold_json",
                   help="Cross-validation strategy")
    ap.add_argument("--holdout_test_ratio", type=float, default=0.2,
                   help="Test set ratio for holdout CV")
    ap.add_argument("--holdout_val_ratio", type=float, default=0.1,
                   help="Validation set ratio for holdout CV")
    ap.add_argument("--kfold_k", type=int, default=5,
                   help="Number of folds for k-fold CV")
    ap.add_argument("--kfold_idx", type=int, default=0,
                   help="Fold index to use for k-fold CV")
    
    # CRITICAL: Output options
    ap.add_argument("--plot_dir", type=str, default="artifacts/plots",
                   help="Directory to save evaluation plots")
    ap.add_argument("--class_names", type=str, default="",
                   help="Comma-separated class names for visualization")
    
    args = ap.parse_args()

    # CRITICAL: Device setup and optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # CRITICAL: Enable cuDNN optimizations for better performance
        torch.backends.cudnn.benchmark = True
        try:
            # CRITICAL: Use high precision for matrix multiplications (PyTorch 2.0+)
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"Using device: {device}")

    # CRITICAL: Load normalization statistics
    # These are computed during preprocessing and are essential for proper data scaling
    stats = load_stats(args.stats)
    fold = None
    
    # CRITICAL: Build global index over all shard files
    # This enables flexible cross-validation strategies (holdout, k-fold, LOSO)
    def build_entries(shards_glob: str):
        """
        CRITICAL: Build index of all data samples across shard files
        
        This function creates a global index that maps each sample to its location
        in the shard files. This is essential for cross-validation as it allows
        us to split data at the sample level rather than the file level.
        
        Returns:
            entries: List of (file_path, sample_index) tuples
            labels: Array of corresponding labels
        """
        files = sorted(glob.glob(shards_glob))
        entries = []
        labels = []
        for f in files:
            z = np.load(f, allow_pickle=False)
            y = z["y"].astype(int)
            n = y.shape[0]
            for j in range(n):
                entries.append((f, j))  # (file_path, sample_index)
                labels.append(int(y[j]))
        return entries, np.array(labels, dtype=np.int64)
    
    entries, labels_all = build_entries(args.shards_glob)

    train_ds = NPZShardsDataset(args.shards_glob, split="all", stats=None)
    val_ds = NPZShardsDataset(args.shards_glob, split="all", stats=None)
    test_ds = NPZShardsDataset(args.shards_glob, split="all", stats=None)

    # dataset wrapper to follow a custom index list
    def subset(ds: NPZShardsDataset, idxs: list[int]) -> Dataset:
        class _S(Dataset):
            def __init__(self, base, index):
                self.base = base
                self.index = index
            def __len__(self): return len(self.index)
            def __getitem__(self, i): return self.base[self.index[i]]
        return _S(ds, idxs)

    # Build index splits according to CV mode
    if args.cv == "fold_json":
        if not args.fold_json:
            raise SystemExit("--fold_json required when --cv=fold_json")
        fold = json.loads(Path(args.fold_json).read_text())
        idx_train = fold["train"]
        idx_val = fold["val"]
        idx_test = fold["test"]
        fold_indices = {"train": idx_train, "val": idx_val, "test": idx_test}
    elif args.cv == "holdout":
        from sklearn.model_selection import StratifiedShuffleSplit
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.holdout_test_ratio, random_state=42)
        all_idx = np.arange(len(entries))
        train_val_idx, test_idx = next(sss1.split(all_idx, labels_all))
        # val from train_val
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=args.holdout_val_ratio, random_state=43)
        tr_idx, va_idx = next(sss2.split(train_val_idx, labels_all[train_val_idx]))
        idx_train = train_val_idx[tr_idx].tolist()
        idx_val = train_val_idx[va_idx].tolist()
        idx_test = test_idx.tolist()
        fold = {"cv": "holdout", "counts": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)}}
        fold_indices = {"train": idx_train, "val": idx_val, "test": idx_test}
    else:  # kfold
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=args.kfold_k, shuffle=True, random_state=42)
        folds = list(skf.split(np.arange(len(entries)), labels_all))
        if not (0 <= args.kfold_idx < len(folds)):
            raise SystemExit(f"--kfold_idx must be in [0,{len(folds)-1}]")
        train_val_idx, test_idx = folds[args.kfold_idx]
        # take small val split from train
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=44)
        tr_idx, va_idx = next(sss.split(train_val_idx, labels_all[train_val_idx]))
        idx_train = train_val_idx[tr_idx].tolist()
        idx_val = train_val_idx[va_idx].tolist()
        idx_test = test_idx.tolist()
        fold = {"cv": "kfold", "k": args.kfold_k, "fold_idx": args.kfold_idx,
                "counts": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)}}
        fold_indices = {"train": idx_train, "val": idx_val, "test": idx_test}

    # small probe to infer channels
    x0, _ = train_ds[0]
    in_ch = x0.shape[0]
    num_classes = int(max([train_ds[i][1] for i in range(min(1000, len(train_ds)))]) + 1)

    # Parse class names if provided
    class_names = None
    if args.class_names:
        class_names = [name.strip() for name in args.class_names.split(",")]
        if len(class_names) != num_classes:
            print(f"Warning: provided {len(class_names)} class names but found {num_classes} classes")
            class_names = None
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    if args.model == "cnn_tcn":
        model = CNN_TCN(in_channels=in_ch, num_classes=num_classes)
    else:
        model = CNN_BiLSTM(in_channels=in_ch, num_classes=num_classes)
    model.to(device)

    # wandb init
    if args.wandb:
        if wandb is None:
            print("wandb not installed. Install with: pip install wandb")
        else:
            run_name = args.wandb_run or f"{Path(args.fold_json).stem}-{args.model}"
            wandb.init(project=args.wandb_project, name=run_name, config={
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "stats": str(args.stats),
                "shards_glob": args.shards_glob,
                "fold_json": args.fold_json,
                "test_subject": fold.get("test_subject", None),
                "counts": fold.get("counts", {}),
                "in_channels": in_ch,
                "num_classes": num_classes,
            })

    # Create plot directory
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # attach normalization inside the training loop
    def apply_norm(x: torch.Tensor) -> torch.Tensor:
        m = torch.from_numpy(stats.mean[None, :, None]).to(x)
        s = torch.from_numpy(stats.std[None, :, None]).to(x)
        return (x - m) / s

    pin = device.type == "cuda"
    nw = max(0, int(args.num_workers))
    tr_loader = DataLoader(subset(train_ds, fold_indices["train"]), batch_size=args.batch_size, shuffle=True, collate_fn=collate, pin_memory=pin, num_workers=nw, persistent_workers=(nw>0))
    va_loader = DataLoader(subset(val_ds, fold_indices["val"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate, pin_memory=pin, num_workers=nw, persistent_workers=(nw>0))
    te_loader = DataLoader(subset(test_ds, fold_indices["test"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate, pin_memory=pin, num_workers=nw, persistent_workers=(nw>0))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    cri = torch.nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and args.amp))
    for ep in range(1, args.epochs + 1):
        model.train()
        loop = tr_loader
        if tqdm is not None:
            loop = tqdm(tr_loader, desc=f"train ep {ep}", leave=False)
        running = 0.0
        steps = 0
        for x, y in loop:
            x = apply_norm(x).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda" and args.amp)):
                logits = model(x)
                loss = cri(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            steps += 1
            if tqdm is not None:
                loop.set_postfix({"loss": f"{running/steps:.4f}"})
        # collect logits/labels for val
        model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            vloop = va_loader
            if tqdm is not None:
                vloop = tqdm(va_loader, desc="validate", leave=False)
            for x, y in vloop:
                x = apply_norm(x).to(device, non_blocking=True)
                l = model(x)
                logits_list.append(l)
                labels_list.append(y.to(device, non_blocking=True))
        logits_va = torch.cat(logits_list, dim=0).to(device)
        labels_va = torch.cat(labels_list, dim=0).to(device)
        # torchmetrics metrics only
        if torchmetrics is None:
            raise SystemExit("torchmetrics is required for metrics; install with: pip install torchmetrics")
        acc_macro = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)(logits_va, labels_va).item()
        f1_weighted = MulticlassF1Score(num_classes=num_classes, average='weighted').to(device)(logits_va, labels_va).item()
        prec_macro = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)(logits_va, labels_va).item()
        rec_macro = MulticlassRecall(num_classes=num_classes, average='macro').to(device)(logits_va, labels_va).item()
        try:
            auroc_ovr = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)(logits_va.softmax(1), labels_va).item()
        except Exception:
            auroc_ovr = float('nan')
        cal_err = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1').to(device)(logits_va.softmax(1), labels_va).item()
        print(f"epoch {ep}: val acc_macro={acc_macro:.4f} f1_weighted={f1_weighted:.4f} ece={cal_err:.4f}")
        if args.wandb and wandb is not None:
            wandb.log({
                "epoch": ep,
                "val/acc_macro": acc_macro,
                "val/f1_weighted": f1_weighted,
                "val/precision_macro": prec_macro,
                "val/recall_macro": rec_macro,
                "val/auroc_macro": auroc_ovr,
                "val/ece": cal_err,
                "train/avg_loss": (running/steps) if steps>0 else None,
            }, step=ep)

    # final eval
    # final eval collect
    def collect(loader):
        model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in loader:
                x = apply_norm(x).to(device, non_blocking=True)
                l = model(x)
                logits_list.append(l)
                labels_list.append(y.to(device, non_blocking=True))
        logits = torch.cat(logits_list, dim=0).to(device)
        labels = torch.cat(labels_list, dim=0).to(device)
        probs = torch.softmax(logits, dim=1)
        return logits, labels, probs
    logits_va, labels_va, P_va = collect(va_loader)
    logits_te, labels_te, P_te = collect(te_loader)
    if torchmetrics is None:
        raise SystemExit("torchmetrics is required for metrics; install with: pip install torchmetrics")
    acc_macro_te = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)(logits_te, labels_te).item()
    f1_weighted_te = MulticlassF1Score(num_classes=num_classes, average='weighted').to(device)(logits_te, labels_te).item()
    prec_macro_te = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)(logits_te, labels_te).item()
    rec_macro_te = MulticlassRecall(num_classes=num_classes, average='macro').to(device)(logits_te, labels_te).item()
    try:
        auroc_ovr_te = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)(logits_te.softmax(1), labels_te).item()
    except Exception:
        auroc_ovr_te = float('nan')
    cal_err_te = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1').to(device)(logits_te.softmax(1), labels_te).item()
    print(f"test: acc_macro={acc_macro_te:.4f} f1_weighted={f1_weighted_te:.4f} ece={cal_err_te:.4f}")
    if args.wandb and wandb is not None:
        wandb.log({
            "final/test_acc_macro": acc_macro_te,
            "final/test_f1_weighted": f1_weighted_te,
            "final/test_precision_macro": prec_macro_te,
            "final/test_recall_macro": rec_macro_te,
            "final/test_auroc_macro": auroc_ovr_te,
            "final/test_ece": cal_err_te,
        })
        # extra metrics via torchmetrics
        if torchmetrics is not None:
            kappa = None
            try:
                from torchmetrics.classification import MulticlassCohenKappa
                kappa = MulticlassCohenKappa(num_classes=num_classes)(logits_te, labels_te).item()
            except Exception:
                pass
            cm = None
            try:
                cm = MulticlassConfusionMatrix(num_classes=num_classes)(logits_te, labels_te).numpy().tolist()
            except Exception:
                pass
            wandb.log({"final/test_kappa": kappa if kappa is not None else float('nan')})
            if cm is not None:
                wandb.log({"final/confusion_matrix": cm})

    # Generate plots if matplotlib is available
    if has_matplotlib:
        print("==> Generating plots...")
        # Convert to numpy for plotting
        logits_te_np = logits_te.cpu().numpy()
        labels_te_np = labels_te.cpu().numpy()
        probs_te_np = torch.softmax(logits_te, dim=1).cpu().numpy()
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)(logits_te, labels_te).cpu().numpy()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        cm_path = plot_dir / f"confusion_matrix_{args.model}_{args.cv}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. AUROC Curve (macro average)
        plt.figure(figsize=(8, 6))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_te_np == i, probs_te_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Macro average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        macro_auc = auc(all_fpr, mean_tpr)
        
        plt.plot(all_fpr, mean_tpr, label=f'Macro-average ROC (AUC = {macro_auc:.3f})')
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], alpha=0.3, label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        roc_path = plot_dir / f"roc_curves_{args.model}_{args.cv}.png"
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. AUPR Curve (macro average)
        plt.figure(figsize=(8, 6))
        precision = dict()
        recall = dict()
        pr_auc = dict()
        for i in range(num_classes):
            precision[i], recall[i], _ = precision_recall_curve(labels_te_np == i, probs_te_np[:, i])
            pr_auc[i] = auc(recall[i], precision[i])
        
        # Macro average
        all_recall = np.unique(np.concatenate([recall[i] for i in range(num_classes)]))
        mean_precision = np.zeros_like(all_recall)
        for i in range(num_classes):
            mean_precision += np.interp(all_recall, recall[i], precision[i])
        mean_precision /= num_classes
        macro_pr_auc = auc(all_recall, mean_precision)
        
        plt.plot(all_recall, mean_precision, label=f'Macro-average PR (AUC = {macro_pr_auc:.3f})')
        for i in range(num_classes):
            plt.plot(recall[i], precision[i], alpha=0.3, label=f'{class_names[i]} (AUC = {pr_auc[i]:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        pr_path = plot_dir / f"pr_curves_{args.model}_{args.cv}.png"
        plt.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Calibration Plot
        plt.figure(figsize=(10, 8))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels_te_np, probs_te_np.max(axis=1), n_bins=10
        )
        axes[0, 0].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        axes[0, 0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        axes[0, 0].set_xlabel('Mean Predicted Probability')
        axes[0, 0].set_ylabel('Fraction of Positives')
        axes[0, 0].set_title('Overall Calibration')
        axes[0, 0].legend()
        
        # Per-class calibration
        for i in range(min(3, num_classes)):  # Show first 3 classes
            row, col = (0, 1) if i == 0 else (1, i-1)
            if i < num_classes:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    labels_te_np == i, probs_te_np[:, i], n_bins=10
                )
                axes[row, col].plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{class_names[i]}")
                axes[row, col].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                axes[row, col].set_xlabel('Mean Predicted Probability')
                axes[row, col].set_ylabel('Fraction of Positives')
                axes[row, col].set_title(f'Calibration - {class_names[i]}')
                axes[row, col].legend()
        
        plt.tight_layout()
        cal_path = plot_dir / f"calibration_{args.model}_{args.cv}.png"
        plt.savefig(cal_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {plot_dir}")
        
        # Log plots to wandb
        if args.wandb and wandb is not None:
            wandb.log({
                "confusion_matrix": wandb.Image(str(cm_path)),
                "roc_curves": wandb.Image(str(roc_path)),
                "pr_curves": wandb.Image(str(pr_path)),
                "calibration": wandb.Image(str(cal_path)),
            })

    if args.calibrate:
        # fit temp on val logits reconstructed via model
        scaler = TemperatureScaler().to(logits_va)
        T = scaler.fit(logits_va, labels_va)
        print(f"Fitted temperature: {T:.3f}")

        # apply to test
        with torch.no_grad():
            P_te_cal = torch.softmax(scaler(logits_te), dim=1)
        cal_err_cal = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1').to(device)(P_te_cal, labels_te).item()
        print(f"test calibrated ECE={cal_err_cal:.4f} (was {cal_err_te:.4f})")
        if args.wandb and wandb is not None:
            wandb.log({"final/test_ece_calibrated": cal_err_cal, "calibration/T": T})

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()


