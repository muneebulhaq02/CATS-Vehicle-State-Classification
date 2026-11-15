# Run:   uvicorn cats_api:app --reload
# Docs:  http://127.0.0.1:8000/docs

from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import chi2
from sklearn.svm import SVC

order = ["NORMAL", "RISK", "CRITICAL"]
mapping = {"NORMAL": 0, "RISK": 1, "CRITICAL": 2}


def add_features(d):
    d = d.copy()
    d["gyro_magnitude"] = np.sqrt(d["gyro_x"] ** 2 + d["gyro_y"] ** 2 + d["gyro_z"] ** 2)
    d["abs_accel_change"] = d["acceleration_change"].abs()
    return d


# ---------- LOAD AND CLEAN ----------
df = pd.read_csv("data/cats_telemetry.csv").drop(["record_id"], axis=1).drop_duplicates()

X = df.drop("state", axis=1).copy()
y = df["state"].map(mapping)

base_cols = list(X.columns)
base_num = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# ---------- MISSING VALUES ----------
imputer = SimpleImputer(strategy="median")
X[base_num] = imputer.fit_transform(X[base_num])

if cat_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# ---------- OUTLIERS (inside each class) ----------
for col in base_num:
    grp = X[col].groupby(y)
    Q1 = grp.transform(lambda s: s.quantile(0.25))
    Q3 = grp.transform(lambda s: s.quantile(0.75))
    IQR = Q3 - Q1
    keep = (X[col] >= Q1 - 1.5 * IQR) & (X[col] <= Q3 + 1.5 * IQR)
    X, y = X[keep], y[keep]

# ---------- NEW FEATURES ----------
X = add_features(X)
num_cols = base_num + ["gyro_magnitude", "abs_accel_change"]

# ---------- ENCODING (only if categorical columns exist) ----------
encoder = None
enc_cols = []


def encode(d):
    e = pd.DataFrame(
        encoder.transform(d[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=d.index
    )
    return pd.concat([d.drop(columns=cat_cols), e], axis=1)


if cat_cols:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X[cat_cols])
    X = encode(X)
    enc_cols = list(encoder.get_feature_names_out(cat_cols))

# ---------- FEATURE SELECTION (correlation > 0.30, chi2 for categorical) ----------
corr = pd.concat([X[num_cols], y], axis=1).corr()["state"].drop("state").abs()
features = corr[corr > 0.30].index.tolist()

if cat_cols:
    _, p_values = chi2(X[enc_cols], y)
    features += [c for c, p in zip(enc_cols, p_values) if p < 0.05]

X = X[features]

normal_mean = X[y == 0].mean()
normal_std = X[y == 0].std()

# ---------- SCALING ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- SMOTE ----------
X_res, y_res = SMOTE(random_state=42).fit_resample(X_scaled, y)

# ---------- MODEL (best from main file) ----------
model = SVC(C=0.1, probability=True, random_state=42)
model.fit(X_res, y_res)

reason_names = {
    "speed": "high speed",
    "acceleration": "abnormal acceleration",
    "acceleration_change": "sudden acceleration change",
    "abs_accel_change": "sudden acceleration change",
    "gyro_x": "strong rotation",
    "gyro_y": "strong rotation",
    "gyro_z": "strong rotation",
    "gyro_magnitude": "strong rotation",
    "impact": "strong impact",
    "battery": "low battery",
    "signal_strength": "weak signal",
}

# ---------- API ----------
app = FastAPI(title="CATS Vehicle State API")


class Telemetry(BaseModel):
    speed: Optional[float] = None
    acceleration: Optional[float] = None
    acceleration_change: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None
    impact: Optional[float] = None
    battery: Optional[float] = None
    signal_strength: Optional[float] = None


@app.get("/")
def home():
    return {"message": "CATS API is running. Send telemetry to POST /predict"}


@app.post("/predict")
def predict(data: Telemetry):
    record = pd.DataFrame([data.model_dump()]).reindex(columns=base_cols).astype(float)

    record[base_num] = imputer.transform(record[base_num])
    record = add_features(record)
    if encoder is not None:
        record = encode(record)
    record = record[features]

    probs = model.predict_proba(scaler.transform(record))[0]
    pred = int(probs.argmax())

    z = ((record.iloc[0] - normal_mean) / normal_std).abs().sort_values(ascending=False)

    reasons = []
    for feature in z.index:
        reason = reason_names.get(feature, feature)
        if z[feature] > 2 and reason not in reasons:
            reasons.append(reason)
        if len(reasons) == 3:
            break

    if pred == 0:
        reasons = ["all values look normal"]

    return {
        "prediction": order[pred],
        "confidence": round(float(probs[pred]), 2),
        "probabilities": {order[i]: round(float(probs[i]), 2) for i in range(3)},
        "reason": " + ".join(reasons),
    }