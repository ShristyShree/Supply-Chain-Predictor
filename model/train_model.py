import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("../data/final_dataset.csv")

# ----------------------------
# TIME-BASED SPLIT (IMPORTANT)
# ----------------------------
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train.drop("will_disrupt_in_next_7_days", axis=1)
y_train = train["will_disrupt_in_next_7_days"]

X_test = test.drop("will_disrupt_in_next_7_days", axis=1)
y_test = test["will_disrupt_in_next_7_days"]

# ----------------------------
# MODEL (XGBOOST)
# ----------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=3,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ----------------------------
# PREDICTIONS
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------
# EVALUATION
# ----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nTrain distribution:\n", y_train.value_counts())
print("\nTest distribution:\n", y_test.value_counts())

# ----------------------------
# SHAP EXPLAINABILITY (MODERN API)
# ----------------------------

# Create explainer
explainer = shap.Explainer(model)

# Compute SHAP values
shap_values = explainer(X_test)

# ----------------------------
# 1. GLOBAL FEATURE IMPORTANCE (BEESWARM)
# ----------------------------
shap.plots.beeswarm(shap_values)

# ----------------------------
# 2. BAR PLOT (FOR PPT)
# ----------------------------
shap.plots.bar(shap_values)

# ----------------------------
# 3. SINGLE PREDICTION (WATERFALL)
# ----------------------------
shap.plots.waterfall(shap_values[0])