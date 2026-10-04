# CATS - Crisis Alert Tech System (Vehicle State Classification)

A small ML prototype that reads one vehicle telemetry record and classifies it as
**NORMAL**, **RISK** or **CRITICAL**.

This is a learning prototype on a **synthetic dataset**. It does not claim to detect
real-world accidents.

## Project files

```

CATS_Project/
├── CATS_Vehicle_State_Classification.py   # full project (EDA, preparation, model, evaluation, explanation)
├── cats_api.py                            # simple FastAPI prediction API
├── data/cats_telemetry.csv                # synthetic dataset
├── results.txt                            # full console output (metrics, reports)
├── screenshots/
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_model_comparison.png
│   ├── 04_confusion_matrix.png
│   ├── 05_feature_importance.png
│   └── 06_pairplot.png                        # key plots (6 images)
├── requirements.txt
└── README.md


```

## 1. How to install dependencies

Python 3.9 or newer is needed.

```
pip install -r requirements.txt
```

Or: `pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn fastapi uvicorn`

## 2. How to run the project

Open a terminal inside the `CATS_Project` folder (the `data` folder must be next to the script) and run:

```
python CATS_Vehicle_State_Classification.py
```

The script prints the results and opens the plots one by one. Close each plot window to continue.

What the script does, in order: load data, check and fill missing values, remove
duplicates, plot distributions, remove outliers, create new features, encode the target,
correlation analysis, split 80/20, check class balance and apply SMOTE (train data only),
scale, train Random Forest on all features and on selected features, compare (accuracy,
precision, recall, F1 overall and for each class), explain a prediction, test a new record.

## 3. Dataset used

`data/cats_telemetry.csv` is a **synthetic dataset** (3,015 rows, including 15 duplicate rows
and about 3% missing sensor values added on purpose to copy real telemetry problems).
The class values overlap on purpose, so the problem is not trivially easy.
It is not real crash data.

| Column | Meaning |
|---|---|
| record_id | row number (dropped before training) |
| speed | vehicle speed |
| acceleration | acceleration |
| acceleration_change | sudden change in acceleration |
| gyro_x, gyro_y, gyro_z | rotation / motion sensor values |
| impact | impact-related value |
| battery | device battery level (%) |
| signal_strength | signal quality (0-100) |
| state | target: NORMAL, RISK or CRITICAL |

Class split: NORMAL 69.57%, RISK 22.39%, CRITICAL 8.04% (imbalanced).

Derived features made by the script: `gyro_magnitude` (combined rotation from x, y, z) and
`abs_accel_change` (size of sudden acceleration change).

## 4. Model used

**Random Forest Classifier** (scikit-learn, 200 trees). Reasons: it handles non-linear
patterns, works well with noisy data, and gives feature importance and probabilities for
explanations.

Preparation steps:
- Missing values filled with the column median
- Duplicates removed
- Outliers removed with the IQR rule inside each class (using IQR on the whole dataset would delete almost all CRITICAL rows)
- Target encoded: NORMAL=0, RISK=1, CRITICAL=2
- 80/20 stratified train/test split
- SMOTE on the training set only, because the data is imbalanced
- StandardScaler
- Tree depth chosen from 3, 5, 8, 12 by best macro-F1 (accuracy, precision and recall are also printed for each depth)

## 5. Evaluation results

Full output is in `results.txt`. Plots are in `screenshots/`.

Results for each tree depth (macro precision, macro recall and macro F1 are averaged over the 3 classes):

All features:

| Depth | Train Acc | Test Acc | Precision | Recall | Macro F1 |
|---|---|---|---|---|---|
| 3 | 0.9880 | 0.9804 | 0.9587 | 0.9714 | 0.9647 |
| 5 | 0.9949 | 0.9857 | 0.9656 | 0.9823 | 0.9733 |
| **8** | 0.9985 | 0.9893 | 0.9729 | 0.9876 | 0.9799 |
| 12 | 0.9996 | 0.9875 | 0.9683 | 0.9831 | 0.9750 |

Selected features:

| Depth | Train Acc | Test Acc | Precision | Recall | Macro F1 |
|---|---|---|---|---|---|
| 3 | 0.9874 | 0.9804 | 0.9565 | 0.9666 | 0.9614 |
| **5** | 0.9942 | 0.9875 | 0.9665 | 0.9850 | 0.9751 |
| 8 | 0.9991 | 0.9875 | 0.9665 | 0.9850 | 0.9751 |
| 12 | 1.0000 | 0.9875 | 0.9665 | 0.9850 | 0.9751 |

The best depth (in bold) is the one with the highest macro F1. For selected features, depths 5, 8
and 12 tie, and the first one (5) is used.

Overall results (test set, 560 rows):

| Features | Best depth | Accuracy | Macro Precision | Macro Recall | Macro F1 | CRITICAL recall |
|---|---|---|---|---|---|---|
| All features | 8 | 0.989 | 0.973 | 0.988 | 0.980 | 1.00 |
| Correlation-selected (6 features) | 5 | 0.988 | 0.966 | 0.985 | 0.975 | 1.00 |

Precision, recall and F1-score for each class:

All features (11 features):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NORMAL | 1.00 | 0.99 | 1.00 | 390 |
| RISK | 0.98 | 0.97 | 0.98 | 125 |
| CRITICAL | 0.94 | 1.000 | 0.97 | 45 |

Selected features (impact, gyro_magnitude, abs_accel_change, signal_strength, speed, acceleration):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NORMAL | 1.00 | 0.99 | 1.00 | 390 |
| RISK | 0.98 | 0.960 | 0.97 | 125 |
| CRITICAL | 0.92 | 1.000 | 0.96 | 45 |

How to read this: precision is "when the model says this class, how often is it right", and
recall is "of the real cases of this class, how many did the model find". For CRITICAL, recall
is 1.00 (no crash was missed) and precision is 0.94 (some RISK cases were predicted as CRITICAL,
which are false alarms).

Confusion matrix (rows = true, columns = predicted):

| | NORMAL | RISK | CRITICAL |
|---|---|---|---|
| NORMAL | 388 | 2 | 0 |
| RISK | 1 | 121 | 3 |
| CRITICAL | 0 | 0 | 45 |

Most important features: `gyro_magnitude`, `impact`, `abs_accel_change`.

Note: results are high because the data is synthetic. Real telemetry would give lower scores.

Screenshots:
- `screenshots/01_target_distribution.png`
- `screenshots/02_correlation_heatmap.png`
- `screenshots/03_model_comparison.png`
- `screenshots/04_confusion_matrix.png`
- `screenshots/05_feature_importance.png`
- `screenshots/06_pairplot.png`

## 6. How to test a new telemetry record

At the bottom of `CATS_Vehicle_State_Classification.py`, find the `new_record` section and
change the values:

```python
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
```

Run the script again. The last output shows the prediction, confidence and a short reason:

```
Prediction: CRITICAL
Confidence: 1.00
Reason: strong rotation + strong impact + sudden acceleration change
```

Confidence is the share of trees that voted for the predicted class. It is not the real
probability of a crash. If a sensor value is missing, set it to `np.nan` (do not delete the key); it will be
filled with the median.

## 7. API / integration code
 
`cats_api.py` is a small FastAPI backend that classifies new telemetry in real time. It repeats
the same preparation steps as the main file (median fill, duplicates, outlier removal, derived
features, SMOTE, scaling) and trains a Random Forest (200 trees, depth 8) when it starts, so
**no trained model file is saved or needed**.
 
Run it from the `CATS_Project` folder (the `data` folder must be next to the file):
 
```
uvicorn cats_api:app --reload
```
 
Then open `http://127.0.0.1:8000/docs` to test it in the browser, or send a request:
 
```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"speed": 110, "acceleration": -7.5, "acceleration_change": 9.0, "gyro_x": 2.1, "gyro_y": -1.8, "gyro_z": 2.5, "impact": 7.0, "battery": 40, "signal_strength": 25}'
```
 
Example response:
 
```
{
  "prediction": "CRITICAL",
  "confidence": 1.0,
  "probabilities": {"NORMAL": 0.0, "RISK": 0.0, "CRITICAL": 1.0},
  "reason": "strong rotation + strong impact + sudden acceleration change"
}
```
 
A missing sensor value can be left out or sent as `null`; it is filled with the training median.
This is a prototype: it has no login, no data storage and no connection to a real CATS bot.
A CATS bot or vehicle device would call `POST /predict` and use the returned state, confidence
and reason to decide on an alert.   