import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score
)


def add_features(d):
    d = d.copy()
    d["gyro_magnitude"] = np.sqrt(d["gyro_x"] ** 2 + d["gyro_y"] ** 2 + d["gyro_z"] ** 2)
    d["abs_accel_change"] = d["acceleration_change"].abs()
    return d


# ==========================================================
# PHASE 1 : LOAD AND CLEAN (before split)
# ==========================================================

df = pd.read_csv("data/cats_telemetry.csv")
df = df.drop(["record_id"], axis=1)

print(df.info())
print(df.describe())

print("\nMissing Values:")
print(df.isna().sum())

print("\nDuplicates:", df.duplicated().sum())
df = df.drop_duplicates()

order = ["NORMAL", "RISK", "CRITICAL"]
mapping = {"NORMAL": 0, "RISK": 1, "CRITICAL": 2}

# ==========================================================
# PHASE 2 : EDA (before split)
# ==========================================================

eda = add_features(df)
eda_cols = eda.select_dtypes(include="number").columns

sns.countplot(x="state", data=eda, order=order)
plt.title("Target Variable Distribution")
plt.show()
print(eda["state"].value_counts())

for col in eda_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(eda[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

    plt.figure(figsize=(6, 3))
    sns.boxplot(x="state", y=col, data=eda, order=order)
    plt.title(col)
    plt.show()

important = ["speed", "acceleration", "impact", "gyro_magnitude", "signal_strength"]
sns.pairplot(eda[important + ["state"]], hue="state", hue_order=order)
plt.show()

eda["state"] = eda["state"].map(mapping)

plt.figure(figsize=(12, 9))
sns.heatmap(eda.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(data=eda, x="speed", y="impact", hue="state")
plt.show()
sns.scatterplot(data=eda, x="gyro_magnitude", y="impact", hue="state")
plt.show()
sns.scatterplot(data=eda, x="signal_strength", y="speed", hue="state")
plt.show()
sns.scatterplot(data=eda, x="acceleration", y="abs_accel_change", hue="state")
plt.show()

# ==========================================================
# PHASE 3 : TRAIN / TEST SPLIT
# ==========================================================

X = df.drop("state", axis=1)
y = df["state"].map(mapping)
print("\nEncoding Mapping:", mapping)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_test = X_train.copy(), X_test.copy()

base_cols = list(X.columns)
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
base_num = list(num_cols)
print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# ==========================================================
# PHASE 4 : MISSING VALUES (fit on train only)
# ==========================================================

imputer = SimpleImputer(strategy="median")
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])

if cat_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

print("\nMissing after imputation:", X_train.isna().sum().sum(), X_test.isna().sum().sum())

# ==========================================================
# PHASE 5 : OUTLIERS (train only, inside each class)
# ==========================================================

print("Train rows before outlier removal:", len(X_train))

for col in num_cols:
    grp = X_train[col].groupby(y_train)
    Q1 = grp.transform(lambda s: s.quantile(0.25))
    Q3 = grp.transform(lambda s: s.quantile(0.75))
    IQR = Q3 - Q1
    keep = (X_train[col] >= Q1 - 1.5 * IQR) & (X_train[col] <= Q3 + 1.5 * IQR)
    X_train, y_train = X_train[keep], y_train[keep]

print("Train rows after outlier removal:", len(X_train))
print(y_train.value_counts())

# ==========================================================
# PHASE 6 : NEW FEATURES
# ==========================================================

X_train = add_features(X_train)
X_test = add_features(X_test)
num_cols = num_cols + ["gyro_magnitude", "abs_accel_change"]

# ==========================================================
# PHASE 7 : ENCODING (only if categorical columns exist)
# ==========================================================

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
    encoder.fit(X_train[cat_cols])
    X_train = encode(X_train)
    X_test = encode(X_test)
    enc_cols = list(encoder.get_feature_names_out(cat_cols))
else:
    print("\nNo categorical columns -> encoding and chi2 skipped")

# ==========================================================
# PHASE 8 : FEATURE SELECTION (train only)
# ==========================================================

train_df = X_train[num_cols].copy()
train_df["state"] = y_train

corr_target = train_df.corr()["state"].drop("state").abs().sort_values(ascending=False)
print("\nCorrelation with Target (train):")
print(corr_target)

plt.figure(figsize=(8, 5))
sns.barplot(x=corr_target.values, y=corr_target.index)
plt.axvline(0.30, color="red", linestyle="--")
plt.title("Correlation with Target (train data)")
plt.show()

plt.figure(figsize=(12, 9))
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (train data)")
plt.show()

threshold = 0.30
selected_features = corr_target[corr_target > threshold].index.tolist()

if cat_cols:
    chi_scores, p_values = chi2(X_train[enc_cols], y_train)
    chi_table = pd.Series(p_values, index=enc_cols)
    print("\nChi2 p-values:")
    print(chi_table)
    selected_features += chi_table[chi_table < 0.05].index.tolist()

all_features = X_train.columns.tolist()
print("\nSelected Features:", selected_features)

# ==========================================================
# PHASE 9 : CLASS IMBALANCE CHECK
# ==========================================================

class_counts = y_train.value_counts()
print("\nClass percentages (train):")
print((class_counts / len(y_train) * 100).round(2))

imbalance_ratio = class_counts.min() / class_counts.max()
print("Smallest class / biggest class =", round(imbalance_ratio, 3))

use_smote = imbalance_ratio < 0.5
print("SMOTE will be applied" if use_smote else "SMOTE not needed")

# ==========================================================
# PHASE 10 : SCALING -> SMOTE -> MODELS
# ==========================================================

models = {
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, random_state=42),
        {"max_depth": [3, 5, 8, 12]}
    ),
    "XGBoost": (
        XGBClassifier(n_estimators=200, learning_rate=0.1, eval_metric="mlogloss", random_state=42),
        {"max_depth": [3, 5, 8]}
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0]}
    ),
    "SVM": (
        SVC(probability=True, random_state=42),
        {"C": [0.1, 1, 10]}
    ),
}

results = []
fitted = {}

for set_name, feats in [("All Features", all_features), ("Selected Features", selected_features)]:

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train[feats])
    Xte = scaler.transform(X_test[feats])

    if use_smote:
        _, y_sm = SMOTE(random_state=42).fit_resample(Xtr, y_train)
        print(f"\n{set_name} - train counts after SMOTE:")
        print(y_sm.value_counts())

    print(f"\n{'=' * 60}\n{set_name.upper()}\n{'=' * 60}")

    for name, (est, grid) in models.items():
        steps = [("smote", SMOTE(random_state=42))] if use_smote else []
        pipe = Pipeline(steps + [("model", est)])
        search = GridSearchCV(
            pipe,
            {f"model__{k}": v for k, v in grid.items()},
            scoring="f1_macro", cv=5, n_jobs=-1
        )
        search.fit(Xtr, y_train)

        pred = search.predict(Xte)
        row = {
            "Features": set_name,
            "Model": name,
            "Best Params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "Train Accuracy": search.score(Xtr, y_train),
            "Test Accuracy": accuracy_score(y_test, pred),
            "Macro Precision": precision_score(y_test, pred, average="macro"),
            "Macro Recall": recall_score(y_test, pred, average="macro"),
            "Macro F1": f1_score(y_test, pred, average="macro"),
            "CRITICAL Recall": recall_score(y_test, pred, labels=[2], average=None)[0],
        }
        results.append(row)
        fitted[(set_name, name)] = {"model": search.best_estimator_, "scaler": scaler, "features": feats, "pred": pred}

        print(
            f"{name} {row['Best Params']} | "
            f"Train Acc={row['Train Accuracy']:.4f} | Test Acc={row['Test Accuracy']:.4f} | "
            f"Precision={row['Macro Precision']:.4f} | Recall={row['Macro Recall']:.4f} | "
            f"Macro F1={row['Macro F1']:.4f}"
        )

# ==========================================================
# PHASE 11 : COMPARISON
# ==========================================================

comparison = pd.DataFrame(results)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

print("\nFINAL COMPARISON")
print(comparison.drop(columns=["Best Params"]).round(4))

best_row = comparison.sort_values(["Macro F1", "CRITICAL Recall"], ascending=False).iloc[0]
best_key = (best_row["Features"], best_row["Model"])
best = fitted[best_key]

print("\nBEST MODEL:", best_key, best_row["Best Params"])
print("\nCLASSIFICATION REPORT (BEST MODEL)")
print(classification_report(y_test, best["pred"], target_names=order))

comparison["Run"] = comparison["Model"] + " - " + comparison["Features"]
comparison_long = comparison.melt(
    id_vars="Run",
    value_vars=["Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "CRITICAL Recall"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(14, 6))
sns.barplot(data=comparison_long, x="Run", y="Score", hue="Metric")
plt.title("Model Comparison")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, best["pred"])
print("\nConfusion Matrix (rows = true, columns = predicted):")
print(cm)
ConfusionMatrixDisplay(cm, display_labels=order).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_key[1]} ({best_key[0]})")
plt.show()

rf_all = fitted[("All Features", "Random Forest")]["model"].named_steps["model"]
importances = pd.Series(rf_all.feature_importances_, index=all_features).sort_values(ascending=False)
print("\nFeature Importance (Random Forest, all features):")
print(importances)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance")
plt.show()

# ==========================================================
# PHASE 12 : EXPLAINING A PREDICTION
# ==========================================================

normal_mean = X_train[y_train == 0][best["features"]].mean()
normal_std = X_train[y_train == 0][best["features"]].std()

names = {
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


def explain(record):
    feats = best["features"]
    probs = best["model"].predict_proba(best["scaler"].transform(record[feats]))[0]
    pred = probs.argmax()

    z = ((record[feats].iloc[0] - normal_mean) / normal_std).abs().sort_values(ascending=False)

    reasons = []
    for feature in z.index:
        reason = names.get(feature, feature)
        if z[feature] > 2 and reason not in reasons:
            reasons.append(reason)
        if len(reasons) == 3:
            break

    if pred == 0:
        reasons = ["all values look normal"]

    print("Prediction:", order[pred])
    print(f"Confidence: {probs[pred]:.2f}")
    print("Reason:", " + ".join(reasons))


def predict_new(raw):
    # same steps as training, using values learned from the training data
    record = pd.DataFrame([raw]).reindex(columns=base_cols)
    record[base_num] = imputer.transform(record[base_num].astype(float))
    record = add_features(record)
    if encoder is not None:
        record = encode(record)
    explain(record)


print("\n" + "=" * 60)
print("EXPLANATION OF ONE TEST PREDICTION")
print("=" * 60)

idx = np.where(best["pred"] == 2)[0][0]
explain(X_test.iloc[[idx]])

print("\n" + "=" * 60)
print("NEW TELEMETRY RECORD")
print("=" * 60)

predict_new({
    "speed": 110,
    "acceleration": -7.5,
    "acceleration_change": 9.0,
    "gyro_x": 2.1,
    "gyro_y": -1.8,
    "gyro_z": 2.5,
    "impact": 7.0,
    "battery": 40,
    "signal_strength": 25
})