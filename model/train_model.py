import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/final_dataset.csv")

# ----------------------------
# TIME-BASED SPLIT
# ----------------------------
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train.drop("will_disrupt_in_next_7_days", axis=1)
y_train = train["will_disrupt_in_next_7_days"]

X_test = test.drop("will_disrupt_in_next_7_days", axis=1)
y_test = test["will_disrupt_in_next_7_days"]

# ----------------------------
# MODELS
# ----------------------------

# XGBOOST
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=3,
    random_state=42,
    eval_metric='logloss'
)

# RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# LOGISTIC REGRESSION
lr = LogisticRegression(max_iter=2000)

# ----------------------------
# TRAIN MODELS
# ----------------------------
xgb.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# ----------------------------
# PREDICTIONS
# ----------------------------

# XGBoost
xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]

# Random Forest
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# Logistic Regression
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]

# ----------------------------
# EVALUATION
# ----------------------------

print("\n==============================")
print("🔹 XGBOOST RESULTS")
print("==============================")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_prob))
print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

print("\n==============================")
print("🔹 RANDOM FOREST RESULTS")
print("==============================")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

print("\n==============================")
print("🔹 LOGISTIC REGRESSION RESULTS")
print("==============================")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr_prob))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# ----------------------------
# ENSEMBLE (PROBABILITY BASED)
# ----------------------------

ensemble_prob = (xgb_prob + rf_prob + lr_prob) / 3
ensemble_pred = (ensemble_prob > 0.5).astype(int)

print("\n==============================")
print("🔥 ENSEMBLE RESULTS")
print("==============================")
print("Accuracy:", accuracy_score(y_test, ensemble_pred))
print("ROC-AUC:", roc_auc_score(y_test, ensemble_prob))
print(confusion_matrix(y_test, ensemble_pred))
print(classification_report(y_test, ensemble_pred))

# ----------------------------
# CLASS DISTRIBUTION
# ----------------------------
print("\nTrain distribution:\n", y_train.value_counts())
print("\nTest distribution:\n", y_test.value_counts())

# ----------------------------
# SHAP EXPLAINABILITY (XGBOOST)
# ----------------------------
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)

print("\nGenerating SHAP plots...")

# Beeswarm
shap.plots.beeswarm(shap_values)

# Bar
shap.plots.bar(shap_values)

# Waterfall (single example)
shap.plots.waterfall(shap_values[0])