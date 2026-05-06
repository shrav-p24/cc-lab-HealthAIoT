import sys
sys.path.insert(0, '/kaggle/working')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model_utils import *
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ── Device ──────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Paths ────────────────────────────────────────────────────────
plots_dir = Path('/kaggle/working')
logs_dir  = Path('/kaggle/working/training_log.txt')
model_dir = Path('/kaggle/working/best_model.pth')

# ── Data ─────────────────────────────────────────────────────────
batch_size = 128
train_iter, val_iter, test_iter = load_data_cdc_diabetes(batch_size, device)
X_sample, y_sample = next(iter(train_iter))
print(f"Batch input shape : {X_sample.size()}")
print(f"Batch label shape : {y_sample.size()}")

# ── Model ────────────────────────────────────────────────────────
model = DiabetesClassifier(in_channels=21, out_channels=2)
model.apply(init_weights)
model.to(device)

loss_fn    = torch.nn.CrossEntropyLoss()
optimizer  = torch.optim.AdamW(
    model.parameters(),
    lr=0.00677174682597258,
    weight_decay=3.103759723739499e-05
)
early_stop = EarlyStopping(wait_epoch=150, index=True)

# ── Log setup ────────────────────────────────────────────────────
def log(lines, text):
    """Print and store a line"""
    print(text)
    lines.append(text)

log_lines = []
log(log_lines, "=" * 60)
log(log_lines, "           MLP TRAINING LOG")
log(log_lines, "=" * 60)
log(log_lines, f"Device        : {device}")
log(log_lines, f"Batch size    : {batch_size}")
log(log_lines, f"Learning rate : 0.00677174682597258")
log(log_lines, f"Weight decay  : 3.103759723739499e-05")
log(log_lines, f"Max epochs    : 200")
log(log_lines, f"Early stop    : 150 epochs patience")
log(log_lines, "=" * 60)

# ── Training loop ────────────────────────────────────────────────
losses, train_accs, val_accs = [], [], []
num_epochs = 200

log(log_lines, "\n--- EPOCH-WISE TRAINING & VALIDATION LOG ---\n")

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for Xb, yb in train_iter:
        out = model(Xb)
        l   = loss_fn(out, yb)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        losses.append(float(l))
        epoch_loss  += float(l)
        num_batches += 1

    avg_loss = epoch_loss / num_batches

    # Validation
    model.eval()
    with torch.no_grad():
        train_acc = evaluate_metric(model, train_iter, correct)
        val_acc   = evaluate_metric(model, val_iter,   correct)
        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))

    epoch_log = (
        f"Epoch {epoch+1:03d}/{num_epochs} | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"Train Acc: {float(train_acc):.4f} | "
        f"Val Acc: {float(val_acc):.4f}"
    )
    log(log_lines, epoch_log)

    # Early stopping check
    early_stop(val_acc, model, epoch+1)
    if early_stop.early_stop:
        stop_msg = (f"\n⚠ Early stopping triggered at epoch {epoch+1}."
                    f"\n  Best Val Acc : {early_stop.max_val_acc:.4f}"
                    f"\n  Best Epoch   : {early_stop.prime_epoch}")
        log(log_lines, stop_msg)
        break

    best_so_far = (f"  └─ Best val acc so far: {early_stop.max_val_acc:.4f} "
                   f"at epoch {early_stop.prime_epoch}")
    log(log_lines, best_so_far)

# ── Validation summary ───────────────────────────────────────────
log(log_lines, "\n" + "=" * 60)
log(log_lines, "           VALIDATION SUMMARY")
log(log_lines, "=" * 60)
log(log_lines, f"Best Validation Accuracy : {early_stop.max_val_acc:.4f}")
log(log_lines, f"Achieved at Epoch        : {early_stop.prime_epoch}")
log(log_lines, f"Total epochs trained     : {len(train_accs)}")

# ── Load best model and evaluate on TEST set ─────────────────────
log(log_lines, "\n" + "=" * 60)
log(log_lines, "           TEST SET EVALUATION")
log(log_lines, "=" * 60)

model.load_state_dict(torch.load(model_dir, map_location=device))
model.eval()

preds_labels, actual_labels = [], []
with torch.no_grad():
    for Xb, yb in test_iter:
        _, predicted = torch.max(model(Xb), 1)
        preds_labels.extend(predicted.cpu().numpy())
        actual_labels.extend(yb.cpu().numpy())

test_acc = evaluate_metric(model, test_iter, correct)
mlp_f1   = f1_score(actual_labels, preds_labels, average='weighted')
mlp_f1_macro = f1_score(actual_labels, preds_labels, average='macro')
report   = classification_report(actual_labels, preds_labels,
                                  target_names=['Not Diabetic', 'Diabetic'])

log(log_lines, f"Test Accuracy  : {float(test_acc):.4f}")
log(log_lines, f"F1 (weighted)  : {mlp_f1:.4f}")
log(log_lines, f"F1 (macro)     : {mlp_f1_macro:.4f}")
log(log_lines, "\nClassification Report:")
log(log_lines, report)

# ── Save full log ────────────────────────────────────────────────
with open(logs_dir, 'w') as f:
    f.write('\n'.join(log_lines))
print(f"\n✅ Full MLP log saved to {logs_dir}")