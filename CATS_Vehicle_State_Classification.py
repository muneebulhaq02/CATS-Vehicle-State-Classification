import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score
)


df = pd.read_csv("data/cats_telemetry.csv")

print(df.head())



df = df.drop(["record_id"], axis=1)

print(df.info())
print(df.describe())
print(df.columns)

categ_col = df.select_dtypes(exclude="number").columns
numeric_col = df.select_dtypes(include="number").columns

print("\nMissing Values:")
print(df.isna().sum())


medians = df[numeric_col].median()
df[numeric_col] = df[numeric_col].fillna(medians)

print("\nMissing Values after filling:")
print(df.isna().sum().sum())


print("\nDuplicates:", df.duplicated().sum())

df = df.drop_duplicates()


order = ["NORMAL", "RISK", "CRITICAL"]

sns.countplot(x="state", data=df, order=order)
plt.title("Target Variable Distribution")
plt.show()

print(df["state"].value_counts())


for col in numeric_col:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()



print("Rows before outlier removal:", len(df))
print(df["state"].value_counts())

for col in numeric_col:

    plt.figure(figsize=(6, 3))
    sns.boxplot(x="state", y=col, data=df, order=order)
    plt.title(col)
    plt.show()

    grp = df.groupby("state")[col]

    Q1 = grp.transform(lambda s: s.quantile(0.25))
    Q3 = grp.transform(lambda s: s.quantile(0.75))

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[
        (df[col] >= lower_bound) &
        (df[col] <= upper_bound)
    ]

print("Rows after outlier removal:", len(df))
print(df["state"].value_counts())


df["gyro_magnitude"] = np.sqrt(df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2)
df["abs_accel_change"] = df["acceleration_change"].abs()

numeric_col = df.select_dtypes(include="number").columns



important = [
    "speed",
    "acceleration",
    "impact",
    "gyro_magnitude",
    "signal_strength"
]

sns.pairplot(df[important + ["state"]], hue="state", hue_order=order)
plt.show()



mapping = {"NORMAL": 0, "RISK": 1, "CRITICAL": 2}

df["state"] = df["state"].map(mapping)

print("\nEncoding Mapping:")
print(mapping)


corr_matrix = df.corr(numeric_only=True)

print("\nCorrelation with Target:")
print(
    corr_matrix["state"]
    .abs()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 9))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()


sns.scatterplot(data=df, x="speed", y="impact", hue="state")
plt.show()
sns.scatterplot(data=df, x="gyro_magnitude", y="impact", hue="state")
plt.show()
sns.scatterplot(data=df, x="signal_strength", y="speed", hue="state")
plt.show()
sns.scatterplot(data=df, x="acceleration", y="abs_accel_change", hue="state")
plt.show()


class_counts = df["state"].value_counts()

print("\nClass percentages:")
print((class_counts / len(df) * 100).round(2))

imbalance_ratio = class_counts.min() / class_counts.max()
print("Smallest class / biggest class =", round(imbalance_ratio, 3))

use_smote = imbalance_ratio < 0.5

if use_smote:
    print("Dataset is IMBALANCED -> SMOTE will be applied on training data")
else:
    print("Dataset is balanced -> SMOTE not needed")

# ==========================================================
# PART 1 : RANDOM FOREST USING ALL FEATURES
# ==========================================================

X = df.drop("state", axis=1)
y = df["state"]

# -----------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_raw = X_train.copy()
y_train_raw = y_train.copy()

# -----------------------------------
# SMOTE 
# -----------------------------------

if use_smote:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("\nTrain class counts after SMOTE:")
    print(y_train.value_counts())

# -----------------------------------
# SCALING
# -----------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results_all = []

print(f"\n{'='*60}")
print("ALL FEATURES - RANDOM FOREST")
print(f"{'='*60}")

for depth in [3, 5, 8, 12]:

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        random_state=42
    )

    rf.fit(X_train, y_train)

    train_acc = rf.score(X_train, y_train)
    y_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    results_all.append([depth, train_acc, test_acc, precision, recall, f1])

    print(
        f"Depth={depth} | "
        f"Train Accuracy={train_acc:.4f} | "
        f"Test Accuracy={test_acc:.4f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"Macro F1={f1:.4f}"
    )

results_all = pd.DataFrame(
    results_all,
    columns=["Depth", "Train Accuracy", "Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
)

best_all = results_all.loc[
    results_all["Macro F1"].idxmax()
]

print("\nBEST RESULT USING ALL FEATURES")
print(best_all)


model_all = RandomForestClassifier(
    n_estimators=200,
    max_depth=int(best_all["Depth"]),
    random_state=42
)
model_all.fit(X_train, y_train)
pred_all = model_all.predict(X_test)

print("\nCLASSIFICATION REPORT (ALL FEATURES)")
print(classification_report(y_test, pred_all, target_names=order))

# ==========================================================
# PART 2 : PEARSON CORRELATION FEATURE SELECTION
# ==========================================================

corr_target = (
    df.corr(numeric_only=True)["state"]
    .abs()
    .sort_values(ascending=False)
)

threshold = 0.30

selected_features = corr_target[
    corr_target > threshold
].index.tolist()

selected_features.remove("state")

print("\nSelected Features:")
print(selected_features)

X_selected = df[selected_features]
y_selected = df["state"]

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_selected, y_selected, test_size=0.2, random_state=42, stratify=y_selected
)

if use_smote:
    X_train_sel, y_train_sel = SMOTE(random_state=42).fit_resample(X_train_sel, y_train_sel)

scaler_sel = StandardScaler()

X_train_sel = scaler_sel.fit_transform(X_train_sel)
X_test_sel = scaler_sel.transform(X_test_sel)

# ==========================================================
# PART 3 : RANDOM FOREST USING SELECTED FEATURES
# ==========================================================

results_selected = []

print(f"\n{'='*60}")
print("SELECTED FEATURES - RANDOM FOREST")
print(f"{'='*60}")

for depth in [3, 5, 8, 12]:

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        random_state=42
    )

    rf.fit(X_train_sel, y_train_sel)

    train_acc = rf.score(X_train_sel, y_train_sel)
    y_pred = rf.predict(X_test_sel)
    test_acc = accuracy_score(y_test_sel, y_pred)
    precision = precision_score(y_test_sel, y_pred, average="macro")
    recall = recall_score(y_test_sel, y_pred, average="macro")
    f1 = f1_score(y_test_sel, y_pred, average="macro")

    results_selected.append([depth, train_acc, test_acc, precision, recall, f1])

    print(
        f"Depth={depth} | "
        f"Train Accuracy={train_acc:.4f} | "
        f"Test Accuracy={test_acc:.4f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"Macro F1={f1:.4f}"
    )

results_selected = pd.DataFrame(
    results_selected,
    columns=["Depth", "Train Accuracy", "Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
)

best_selected = results_selected.loc[
    results_selected["Macro F1"].idxmax()
]

print("\nBEST RESULT USING SELECTED FEATURES")
print(best_selected)

model_sel = RandomForestClassifier(
    n_estimators=200,
    max_depth=int(best_selected["Depth"]),
    random_state=42
)
model_sel.fit(X_train_sel, y_train_sel)
pred_sel = model_sel.predict(X_test_sel)

print("\nCLASSIFICATION REPORT (SELECTED FEATURES)")
print(classification_report(y_test_sel, pred_sel, target_names=order))

# ==========================================================
# FINAL COMPARISON
# ==========================================================

comparison = pd.DataFrame({
    "Dataset": ["All Features", "Selected Features"],
    "Best Depth": [best_all["Depth"], best_selected["Depth"]],
    "Test Accuracy": [
        accuracy_score(y_test, pred_all),
        accuracy_score(y_test_sel, pred_sel)
    ],
    "Macro Precision": [
        precision_score(y_test, pred_all, average="macro"),
        precision_score(y_test_sel, pred_sel, average="macro")
    ],
    "Macro Recall": [
        recall_score(y_test, pred_all, average="macro"),
        recall_score(y_test_sel, pred_sel, average="macro")
    ],
    "Macro F1": [
        f1_score(y_test, pred_all, average="macro"),
        f1_score(y_test_sel, pred_sel, average="macro")
    ],
    "CRITICAL Recall": [
        recall_score(y_test, pred_all, labels=[2], average=None)[0],
        recall_score(y_test_sel, pred_sel, labels=[2], average=None)[0]
    ]
})

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print("\nFINAL COMPARISON")
print(comparison)

# -----------------------------------
# PRECISION, RECALL, F1 FOR EACH CLASS
# -----------------------------------

per_class = []

for name, yt, yp in [
    ("All Features", y_test, pred_all),
    ("Selected Features", y_test_sel, pred_sel)
]:
    precision = precision_score(yt, yp, average=None)
    recall = recall_score(yt, yp, average=None)
    f1_each = f1_score(yt, yp, average=None)

    for i, cls in enumerate(order):
        per_class.append([name, cls, precision[i], recall[i], f1_each[i]])

per_class = pd.DataFrame(
    per_class,
    columns=["Dataset", "Class", "Precision", "Recall", "F1-score"]
)

print("\nPRECISION / RECALL / F1 FOR EACH CLASS")
print(per_class.round(4))

# -----------------------------------
# VISUAL COMPARISON
# -----------------------------------

comparison_long = comparison.melt(
    id_vars="Dataset",
    value_vars=["Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "CRITICAL Recall"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 5))
sns.barplot(data=comparison_long, x="Dataset", y="Score", hue="Metric")
plt.title("Model Comparison")
plt.ylim(0, 1)
plt.show()

# -----------------------------------
# CONFUSION MATRIX (all features model)
# -----------------------------------

cm = confusion_matrix(y_test, pred_all)
ConfusionMatrixDisplay(cm, display_labels=order).plot(cmap="Blues")
plt.title("Confusion Matrix - All Features")
plt.show()

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

importances = pd.Series(
    model_all.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importances)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance")
plt.show()

# ==========================================================
# PART 4 : EXPLAINING A PREDICTION
# ==========================================================
# Simple idea: compare the record with the average NORMAL record.
# Features that are far from normal are given as the reason.

normal_mean = X_train_raw[y_train_raw == 0].mean()
normal_std = X_train_raw[y_train_raw == 0].std()

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
    record = record.fillna(medians)
    scaled = scaler.transform(record[X.columns])
    probs = model_all.predict_proba(scaled)[0]
    pred = probs.argmax()

    z = (record[X.columns].iloc[0] - normal_mean) / normal_std
    z = z.abs().sort_values(ascending=False)

    reasons = []
    for feature in z.index:
        if z[feature] > 2 and names[feature] not in reasons:
            reasons.append(names[feature])
        if len(reasons) == 3:
            break

    if pred == 0:
        reasons = ["all values look normal"]

    print("Prediction:", order[pred])
    print(f"Confidence: {probs[pred]:.2f}")
    print("Reason:", " + ".join(reasons))


print("\n" + "=" * 60)
print("EXPLANATION OF ONE TEST PREDICTION")
print("=" * 60)

# first test record that the model predicted as CRITICAL
idx = np.where(pred_all == 2)[0][0]
one_record = X.loc[y_test.index[[idx]]]
explain(one_record)

# -----------------------------------
# TEST A NEW TELEMETRY RECORD
# -----------------------------------

new_record = pd.DataFrame([{
    "speed": 110,
    "acceleration": -7.5,
    "acceleration_change": 9.0,
    "gyro_x": 2.1,
    "gyro_y": -1.8,
    "gyro_z": 2.5,
    "impact": 7.0,
    "battery": 40,
    "signal_strength": 25
}])

new_record["gyro_magnitude"] = np.sqrt(
    new_record["gyro_x"] ** 2 + new_record["gyro_y"] ** 2 + new_record["gyro_z"] ** 2
)
new_record["abs_accel_change"] = new_record["acceleration_change"].abs()

print("\n" + "=" * 60)
print("NEW TELEMETRY RECORD")
print("=" * 60)
explain(new_record)