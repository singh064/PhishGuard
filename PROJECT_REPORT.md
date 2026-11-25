# PhishGuard Project Report

**Validation date:** 2026-11-26  
**Project:** PhishGuard – Intelligent Phishing Website Detection System

## 1. Scope and implementation

PhishGuard performs offline lexical and structural analysis of URL strings.
The Streamlit interface extracts 12 numeric features, sends them to a selected
saved classifier, and displays the predicted class, a model risk score, and
feature values. It does not open or connect to submitted URLs.

Feature generation is shared between batch preparation and live scanning via
`src/feature_extraction.py`. Training and evaluation code are in
`src/train_model.py` and `src/evaluate_models.py`. The UI is `app.py`.

## 2. Dataset and labels

Inspected local files:

- `data/cleaned_urls.csv`: 507,194 rows; columns `URL`, `Label`.
- `data/features_dataset.csv`: 507,194 rows; 12 numerical feature columns and
  `Label`.
- `models/`: saved Logistic Regression, SVM, Decision Tree, and feature-column
  metadata.
- `results/model_comparison.csv`: saved held-out evaluation results.

Class counts are 392,896 `good` records (77.46%) and 114,298 `bad` records
(22.54%). Training and evaluation map `good → 0` and `bad → 1`; the loaded
models have classes `[0, 1]`, and the application interprets class 1 as
phishing. Phishing is consistently treated as the positive class.

The feature order in the extractor, feature dataset, JSON/joblib metadata, and
all three model artifacts matches exactly. Exact URL duplicate groups and
cross-split URL duplicates were not found. However, 71,842 held-out rows have
an engineered feature vector also present in training, and some such vectors
have conflicting labels. This reflects the limited distinguishing power of
the 12 features; it is not evidence that the same raw URL crossed the split.

The local project text attributes the dataset to a Kaggle “Phishing Site
URLs” source. The CSV files themselves do not establish provenance or license,
so those details should be confirmed before redistribution.

## 3. Training and evaluation method

The saved evaluation reconstructs a stratified 80/20 split with seed 42:
405,755 training records and 101,439 test records.

- Logistic Regression: StandardScaler followed by LogisticRegression.
- SVM: StandardScaler followed by CalibratedClassifierCV around LinearSVC,
  using sigmoid calibration and three folds.
- Decision Tree: maximum depth 12, with minimum split/leaf constraints.

The Logistic Regression and SVM pipelines apply their fitted scalers at live
inference; the Decision Tree is unscaled. One training-method issue remains:
the SVM scaler is fitted before CalibratedClassifierCV performs its internal
folds, so each validation fold contributes to those scaling statistics. The
held-out test data is still separate from this preprocessing fit.

## 4. Reproduced held-out results

The following metrics were reproduced by loading the saved models and
predicting on the reconstructed test split. They match
`results/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 85.58% | 92.14% | 39.39% | 55.19% | 99.02% | 0.8002 | 77,811 | 768 | 13,856 | 9,004 |
| Support Vector Machine | 85.61% | 92.66% | 39.23% | 55.13% | 99.10% | 0.7968 | 77,869 | 710 | 13,891 | 8,969 |
| Decision Tree | 87.78% | 85.08% | 55.49% | 67.17% | 97.17% | 0.8864 | 76,354 | 2,225 | 10,174 | 12,686 |

All models have false positives and false negatives. The Decision Tree has the
best recall among these saved models, but still misses 10,174 phishing records
in this test split. No perfect-detection claim is warranted.

## 5. Local prediction checks

No URL below was visited. Examples were passed only to local feature
extraction and saved model inference.

### Illustrative legitimate URLs

| Input | Logistic Regression | SVM | Decision Tree |
|---|---|---|---|
| `https://www.google.com/` | Phishing | Phishing | Legitimate |
| `https://www.wikipedia.org/` | Phishing | Phishing | Legitimate |
| `https://www.python.org/` | Phishing | Phishing | Legitimate |
| `https://www.microsoft.com/` | Phishing | Phishing | Legitimate |
| `https://example.com/` | Phishing | Phishing | Legitimate |

For the earlier Google search URL containing `cybersecurity`, all three saved
models predict phishing. The keyword counter matches substrings, so it counts
`security` inside `cybersecurity`; the query also has three special characters.

### Labeled local dataset samples

Using five seeded local `good` rows and five seeded local `bad` rows from
`cleaned_urls.csv`, features were extracted locally and compared with the
provided labels. No source URL was opened or printed.

| Model | Correct on 5 `good` rows | Correct on 5 `bad` rows |
|---|---:|---:|
| Logistic Regression | 5/5 | 1/5 |
| SVM | 5/5 | 1/5 |
| Decision Tree | 5/5 | 3/5 |

These small samples are diagnostic examples, not independent performance
estimates; the held-out confusion counts above are the evaluation results.

## 6. Risk-score and calibration validation

All three artifacts provide `predict_proba`, and the application reads the
class-1 value. That is a genuine model output, but it does not by itself prove
that a live score is a calibrated probability. Logistic Regression has no
separate calibration stage; SVM uses sigmoid calibration; Decision Tree
returns leaf class proportions. Coarse calibration checks on the overall
held-out split do not validate score reliability for URLs outside that
distribution.

The strongest warning is the HTTPS distribution: only 7 of 507,194 rows have
`is_https=1`, and all 7 are phishing. Four appear in the test split and all
four are labeled phishing. Thus there are no legitimate HTTPS examples with
which to validate live scores for Google or similar URLs. The UI now calls the
display a **Model Risk Score** and explicitly says it is not a verified,
calibrated probability.

## 7. Test record

Validation ran in Python 3.14.4 with pandas 3.1.1, NumPy 2.5.4,
scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, Streamlit 1.67.0,
and joblib 1.6.0.

| Check | Result |
|---|---|
| Feature extraction and URL validation script | Passed; normal HTTPS, bare hostname, IP URL, blank input, malformed input, unsupported scheme, and invalid port cases |
| Python syntax for app and feature files | Passed after repairing malformed app assignments |
| All three saved model artifacts load | Passed |
| Feature count and name/order alignment | Passed; 12 features match across extractor, metadata, dataset, and model artifacts |
| Held-out metrics and confusion counts | Passed; reproduced values match the saved CSV |
| Dependency consistency (`pip check`) | Passed |
| Streamlit startup | Passed; local health endpoint returned `ok` |
| Sidebar navigation | Passed; Home, URL Scanner, Model Performance, and About sections opened |
| Home quick-navigation buttons | Passed after fixing their session-state callbacks |
| Empty and malformed scanner input | Passed; both show a validation warning and do not run inference |
| Valid local URL scanner | Passed; Google root was scanned without a network request. Decision Tree: legitimate, 22.6/100; Logistic Regression: phishing, 99.3/100; SVM: phishing, 79.6/100 |
| Model Performance dashboard | Passed; CSV-backed metric table and chart rendered. Visible table rows and values matched the saved evaluation results |

## 8. Corrections and remaining limitations

The app's malformed navigation and sample-button assignments were repaired.
The Home quick-navigation buttons now update the keyed sidebar radio through
callbacks; directly changing that widget state after it had been instantiated
previously raised a Streamlit error. The scanner now rejects blank or malformed
inputs, unsupported schemes, and invalid ports before inference. Risk output
wording was changed from “probability” to “Model Risk Score.”

No model was retrained or overwritten. The following issues remain:

- The suspicious-keyword feature uses substring matching (`security` matches
  inside `cybersecurity`). Changing this feature requires regenerating training
  features and retraining to keep the saved models consistent.
- The training dataset has only seven HTTPS records, all labeled phishing.
- SVM scaling should be moved inside the estimator cloned by its calibration
  cross-validation before a future retraining run.
- Random row splitting does not establish generalization to unseen domains.
- The app is a URL-string classifier, not a browser, reputation service, or
  page-content scanner.

Recommended future work is to add representative legitimate HTTPS data,
resolve keyword tokenization, correct the SVM calibration-fold preprocessing,
and report calibration on an independent domain-aware test set before any
probability claims are made.
