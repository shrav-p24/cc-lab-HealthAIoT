from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

xgb_log_lines = []

def log(lines, text):
    print(text)
    lines.append(text)

log(xgb_log_lines, "\n" + "=" * 60)
log(xgb_log_lines, "           XGBOOST TRAINING LOG")
log(xgb_log_lines, "=" * 60)
log(xgb_log_lines, "Parameters:")
log(xgb_log_lines, "  n_estimators  : 300")
log(xgb_log_lines, "  max_depth     : 6")
log(xgb_log_lines, "  learning_rate : 0.05")
log(xgb_log_lines, "  subsample     : 0.8")
log(xgb_log_lines, "=" * 60)

print("Loading data...")
X_train, X_val, X_test, y_train, y_val, y_test = load_raw_data()
log(xgb_log_lines, f"Train size : {X_train.shape}")
log(xgb_log_lines, f"Val size   : {X_val.shape}")
log(xgb_log_lines, f"Test size  : {X_test.shape}")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

log(xgb_log_lines, "\n--- TRAINING PROGRESS ---")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

# ── Validation performance ───────────────────────────────────────
val_preds   = xgb_model.predict(X_val)
val_acc     = accuracy_score(y_val, val_preds)
val_f1      = f1_score(y_val, val_preds, average='weighted')

log(xgb_log_lines, "\n" + "=" * 60)
log(xgb_log_lines, "           VALIDATION RESULTS")
log(xgb_log_lines, "=" * 60)
log(xgb_log_lines, f"Validation Accuracy : {val_acc:.4f}")
log(xgb_log_lines, f"Validation F1       : {val_f1:.4f}")

# ── Test performance ─────────────────────────────────────────────
xgb_preds   = xgb_model.predict(X_test)
xgb_acc     = accuracy_score(y_test, xgb_preds)
xgb_f1      = f1_score(y_test, xgb_preds, average='weighted')
xgb_f1_mac  = f1_score(y_test, xgb_preds, average='macro')
xgb_report  = classification_report(y_test, xgb_preds,
                                     target_names=['Not Diabetic', 'Diabetic'])

log(xgb_log_lines, "\n" + "=" * 60)
log(xgb_log_lines, "           TEST SET RESULTS")
log(xgb_log_lines, "=" * 60)
log(xgb_log_lines, f"Test Accuracy  : {xgb_acc:.4f}")
log(xgb_log_lines, f"F1 (weighted)  : {xgb_f1:.4f}")
log(xgb_log_lines, f"F1 (macro)     : {xgb_f1_mac:.4f}")
log(xgb_log_lines, "\nClassification Report:")
log(xgb_log_lines, xgb_report)

joblib.dump(xgb_model, '/kaggle/working/xgb_model.pkl')
log(xgb_log_lines, "✅ XGBoost model saved to /kaggle/working/xgb_model.pkl")

# Append to main log
with open(logs_dir, 'a') as f:
    f.write('\n'.join(xgb_log_lines))
print(f"\n✅ XGBoost log appended to {logs_dir}")