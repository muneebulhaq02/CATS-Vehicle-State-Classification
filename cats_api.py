# ==========================================================
# CATS - Crisis Alert Tech System : simple prediction API
# ==========================================================
# Run:   uvicorn cats_api:app --reload
# Docs:  http://127.0.0.1:8000/docs
#
# The model is trained when the API starts (same steps as the main
# project file), so no saved model file is needed.

from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# TRAIN THE MODEL (runs once at start)
# -----------------------------------

order = ["NORMAL", "RISK", "CRITICAL"]
mapping = {"NORMAL": 0, "RISK": 1, "CRITICAL": 2}

df = pd.read_csv("data/cats_telemetry.csv").drop(["record_id"], axis=1)

numeric_col = df.select_dtypes(include="number").columns

# missing values -> median (medians are also used for new records)
medians = df[numeric_col].median()
df[numeric_col] = df[numeric_col].fillna(medians)

df = df.drop_duplicates()

# outliers removed with IQR inside each class
for col in numeric_col:
    grp = df.groupby("state")[col]
    Q1 = grp.transform(lambda s: s.quantile(0.25))
    Q3 = grp.transform(lambda s: s.quantile(0.75))
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# derived features
df["gyro_magnitude"] = np.sqrt(df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2)
df["abs_accel_change"] = df["acceleration_change"].abs()

df["state"] = df["state"].map(mapping)

X = df.drop("state", axis=1)
y = df["state"]
features = X.columns

normal_mean = X[y == 0].mean()
normal_std = X[y == 0].std()

X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
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

# -----------------------------------
# API
# -----------------------------------

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
    record = pd.DataFrame([data.model_dump()])

    # fill missing sensor values with the training medians
    record = record.fillna(medians)

    # same derived features as in training
    record["gyro_magnitude"] = np.sqrt(
        record["gyro_x"] ** 2 + record["gyro_y"] ** 2 + record["gyro_z"] ** 2
    )
    record["abs_accel_change"] = record["acceleration_change"].abs()
    record = record[features]

    probs = model.predict_proba(scaler.transform(record))[0]
    pred = int(probs.argmax())

    # reason: features far from the average NORMAL record
    z = ((record.iloc[0] - normal_mean) / normal_std).abs().sort_values(ascending=False)

    reasons = []
    for feature in z.index:
        if z[feature] > 2 and reason_names[feature] not in reasons:
            reasons.append(reason_names[feature])
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