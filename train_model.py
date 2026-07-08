"""
NSAP Scheme Eligibility Prediction - Model Training Script
Problem Statement 34 - Edunet Foundation / IBM SkillBuild Internship

This script:
1. Loads the district-wise NSAP pension dataset
2. Engineers demographic ratio features (so the model generalizes
   instead of just memorizing raw beneficiary counts)
3. Trains a multi-class RandomForest classifier to predict schemecode
   (IGNOAPS = Old Age Pension, IGNWPS = Widow Pension, IGNDPS = Disability Pension)
4. Evaluates the model
5. Saves the trained model + label encoder + scaler for deployment
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
DATA_PATH = "DistrictwisePensiondataundertheNationalSocialAssistanceProgrammeNSAP.csv"
df = pd.read_csv(DATA_PATH)

print("Raw shape:", df.shape)
print(df['schemecode'].value_counts())

# ------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# Convert raw counts into ratios/percentages -> much better signal
# for a classifier than raw absolute numbers (which scale with
# district population and would just leak size, not eligibility pattern)
# ------------------------------------------------------------------
df["male_pct"] = df["totalmale"] / df["totalbeneficiaries"]
df["female_pct"] = df["totalfemale"] / df["totalbeneficiaries"]
df["transgender_pct"] = df["totaltransgender"] / df["totalbeneficiaries"]
df["sc_pct"] = df["totalsc"] / df["totalbeneficiaries"]
df["st_pct"] = df["totalst"] / df["totalbeneficiaries"]
df["gen_pct"] = df["totalgen"] / df["totalbeneficiaries"]
df["obc_pct"] = df["totalobc"] / df["totalbeneficiaries"]
df["aadhaar_coverage_pct"] = df["totalaadhaar"] / df["totalbeneficiaries"]
df["mobile_coverage_pct"] = df["totalmpbilenumber"] / df["totalbeneficiaries"]
df["beneficiary_density"] = df["totalbeneficiaries"] / df["totalbeneficiaries"].max()

df = df.fillna(0)

FEATURES = [
    "male_pct", "female_pct", "transgender_pct",
    "sc_pct", "st_pct", "gen_pct", "obc_pct",
    "aadhaar_coverage_pct", "mobile_coverage_pct", "beneficiary_density"
]
TARGET = "schemecode"

X = df[FEATURES]
y = df[TARGET]

# ------------------------------------------------------------------
# 3. ENCODE TARGET + SCALE FEATURES
# ------------------------------------------------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# NOTE: dataset is small (100 rows) -> stratify to keep class balance
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ------------------------------------------------------------------
# 5. TRAIN MODEL
# ------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 6. EVALUATE
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importances:\n", importances)

# ------------------------------------------------------------------
# 7. SAVE ARTIFACTS FOR DEPLOYMENT
# ------------------------------------------------------------------
joblib.dump(model, "nsap_model.pkl")
joblib.dump(scaler, "nsap_scaler.pkl")
joblib.dump(le, "nsap_label_encoder.pkl")
joblib.dump(FEATURES, "nsap_features.pkl")

print("\nSaved: nsap_model.pkl, nsap_scaler.pkl, nsap_label_encoder.pkl, nsap_features.pkl")
