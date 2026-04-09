import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "final_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "target"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)

xgb.fit(X_train, y_train)
rf.fit(X_train, y_train)

xgb_prob = xgb.predict_proba(X_test)[:, 1]
rf_prob = rf.predict_proba(X_test)[:, 1]

ensemble_prob = 0.6 * xgb_prob + 0.4 * rf_prob
ensemble_pred = (ensemble_prob > 0.4).astype(int)

print(confusion_matrix(y_test, ensemble_pred))
print(classification_report(y_test, ensemble_pred))
print("ROC-AUC:", roc_auc_score(y_test, ensemble_prob))

joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb.pkl"))
joblib.dump(rf, os.path.join(MODEL_DIR, "rf.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

explainer = shap.TreeExplainer(xgb)
shap_vals = explainer.shap_values(X_test)

shap.summary_plot(shap_vals, X_test)