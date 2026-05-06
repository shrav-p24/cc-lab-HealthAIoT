import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

ens_log_lines = []

def log(lines, text):
    print(text)
    lines.append(text)

log(ens_log_lines, "\n" + "=" * 60)
log(ens_log_lines, "      ENSEMBLE (MLP + XGBoost) LOG")
log(ens_log_lines, "=" * 60)

# MLP probabilities
mlp_probs = []
model.eval()
with torch.no_grad():
    for Xb, yb in test_iter:
        probs = torch.nn.functional.softmax(model(Xb), dim=1)
        mlp_probs.append(probs.cpu().numpy())
mlp_flat = np.vstack(mlp_probs)

# Align sizes
n              = mlp_flat.shape[0]
X_test_aligned = X_test[:n]
y_test_aligned = y_test[:n]
xgb_probs      = xgb_model.predict_proba(X_test_aligned)

log(ens_log_lines, f"MLP test samples     : {mlp_flat.shape[0]}")
log(ens_log_lines, f"XGBoost test samples : {xgb_probs.shape[0]}")
log(ens_log_lines, f"Aligned to           : {n} samples")

# Ensemble
ensemble_probs = (mlp_flat + xgb_probs) / 2
ensemble_preds = np.argmax(ensemble_probs, axis=1)

ens_acc     = accuracy_score(y_test_aligned, ensemble_preds)
ens_f1      = f1_score(y_test_aligned, ensemble_preds, average='weighted')
ens_f1_mac  = f1_score(y_test_aligned, ensemble_preds, average='macro')
ens_report  = classification_report(y_test_aligned, ensemble_preds,
                                     target_names=['Not Diabetic', 'Diabetic'])

log(ens_log_lines, "\n" + "=" * 60)
log(ens_log_lines, "           ENSEMBLE TEST RESULTS")
log(ens_log_lines, "=" * 60)
log(ens_log_lines, f"Test Accuracy  : {ens_acc:.4f}")
log(ens_log_lines, f"F1 (weighted)  : {ens_f1:.4f}")
log(ens_log_lines, f"F1 (macro)     : {ens_f1_mac:.4f}")
log(ens_log_lines, "\nClassification Report:")
log(ens_log_lines, ens_report)

# ── Final comparison ─────────────────────────────────────────────
log(ens_log_lines, "\n" + "=" * 60)
log(ens_log_lines, "        FINAL MODEL COMPARISON SUMMARY")
log(ens_log_lines, "=" * 60)
log(ens_log_lines, f"{'Model':<12} | {'Test Acc':>10} | {'F1 Weighted':>12} | {'F1 Macro':>10}")
log(ens_log_lines, "-" * 55)
log(ens_log_lines, f"{'MLP':<12} | {float(test_acc):>10.4f} | {mlp_f1:>12.4f} | {mlp_f1_macro:>10.4f}")
log(ens_log_lines, f"{'XGBoost':<12} | {xgb_acc:>10.4f} | {xgb_f1:>12.4f} | {xgb_f1_mac:>10.4f}")
log(ens_log_lines, f"{'Ensemble':<12} | {ens_acc:>10.4f} | {ens_f1:>12.4f} | {ens_f1_mac:>10.4f}")
log(ens_log_lines, "=" * 60)

# Best model
best = max(
    [("MLP", float(test_acc)), ("XGBoost", xgb_acc), ("Ensemble", ens_acc)],
    key=lambda x: x[1]
)
log(ens_log_lines, f"\n🏆 Best Model: {best[0]} with accuracy {best[1]:.4f}")

with open(logs_dir, 'a') as f:
    f.write('\n'.join(ens_log_lines))
print(f"\n✅ Ensemble log appended to {logs_dir}")