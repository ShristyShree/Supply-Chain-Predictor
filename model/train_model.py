"""
model/train_model.py
--------------------
Run ONCE before launching app.py.
Saves XGBoost, RandomForest, LogisticRegression to model/*.pkl
so app.py never retrains at runtime.

Usage:
  cd supply_chain_ai
  python model/train_model.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "final_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "will_disrupt_in_next_7_days"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("Loading dataset…")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}  |  Class balance: {df[TARGET].value_counts().to_dict()}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Time-based split (no data leakage)
split = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
class_ratio = int((y_train == 0).sum() / (y_train == 1).sum())

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=class_ratio,   # handles class imbalance
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

lr = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining models…")

print("  XGBoost…", end=" ", flush=True)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("done")

print("  RandomForest…", end=" ", flush=True)
rf.fit(X_train, y_train)
print("done")

print("  LogisticRegression…", end=" ", flush=True)
lr.fit(X_train, y_train)
print("done")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(name: str, model, X_t, y_t):
    pred = model.predict(X_t)
    prob = model.predict_proba(X_t)[:, 1]
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy : {accuracy_score(y_t, pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_t, prob):.4f}")
    print(confusion_matrix(y_t, pred))
    print(classification_report(y_t, pred))
    return prob


xgb_prob = evaluate("XGBoost",           xgb, X_test, y_test)
rf_prob  = evaluate("Random Forest",     rf,  X_test, y_test)
lr_prob  = evaluate("Logistic Regr.",    lr,  X_test, y_test)

ens_prob = (xgb_prob + rf_prob + lr_prob) / 3
ens_pred = (ens_prob > 0.5).astype(int)

print(f"\n{'='*40}")
print("  ENSEMBLE")
print(f"{'='*40}")
print(f"  Accuracy : {accuracy_score(y_test, ens_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, ens_prob):.4f}")
print(classification_report(y_test, ens_pred))

# ─────────────────────────────────────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\nSaving models…")
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"), compress=3)
joblib.dump(rf,  os.path.join(MODEL_DIR, "rf_model.pkl"),  compress=3)
joblib.dump(lr,  os.path.join(MODEL_DIR, "lr_model.pkl"),  compress=3)
print(f"  Saved to {MODEL_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# SHAP GLOBAL IMPORTANCE  (saved as PNG for reference)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating SHAP importance plot…")
explainer  = shap.TreeExplainer(xgb)
shap_vals  = explainer.shap_values(X_test)

fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
plt.tight_layout()
out_path = os.path.join(MODEL_DIR, "shap_importance.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  SHAP plot saved → {out_path}")

print("\n✅ Training complete. Run: streamlit run app.py")
