# PhishGuard – Intelligent Phishing Website Detection System

PhishGuard is an offline URL-structure classifier. It extracts lexical and
structural features from URL text and uses saved scikit-learn models to predict
whether a record resembles the legitimate (`0`) or phishing (`1`) class. It
does not visit submitted URLs.

## Dataset

The local source file `data/cleaned_urls.csv` contains **507,194 labeled URL
records** with `URL` and `Label` columns:

| Label | Meaning | Records | Share |
|---|---|---:|---:|
| `good` | Legitimate (mapped to 0) | 392,896 | 77.46% |
| `bad` | Phishing (mapped to 1) | 114,298 | 22.54% |

`data/features_dataset.csv` contains the same rows as 12 numerical features
plus `Label`; the raw URL column is not used as a model input. An inspection
found no exact URL duplicates in the cleaned file. The existing project text
attributes the data to a Kaggle “Phishing Site URLs” dataset, but provenance
and licensing are not independently verifiable from the local CSV metadata.

An important data limitation: only **7** records have `is_https=1`, and all 7
are labeled phishing. The dataset contains no legitimate HTTPS records. This
causes false alarms on common legitimate HTTPS URLs for some saved models.

## Features and models

The shared offline feature extractor produces these ordered columns:

`url_length`, `domain_length`, `num_dots`, `num_hyphens`, `num_digits`,
`is_https`, `has_ip_address`, `num_suspicious_keywords`, `num_subdomains`,
`has_at_symbol`, `num_special_chars`, `has_redirection`.

The label mapping is `good → 0` and `bad → 1`; **phishing is the positive
class** throughout training, evaluation, and live predictions.

Saved models:

- Logistic Regression with `StandardScaler`.
- Linear SVM (`LinearSVC`) with `StandardScaler` and sigmoid calibration.
- Decision Tree with maximum depth 12.

All three saved models accept the same 12 ordered features. The scaler is part
of each applicable saved pipeline and is applied at inference. The SVM scaler
is currently outside its internal calibration cross-validation, so its
statistics include calibration-fold validation rows. The supplied models have
not been retrained as part of this validation.

## Held-out evaluation

The saved evaluation uses a stratified 80/20 split with `random_state=42`:
405,755 training rows and 101,439 held-out test rows. The following results
were reproduced locally from the saved model files and match
`results/model_comparison.csv`.

| Model | Accuracy | Precision (phishing) | Recall (phishing) | F1 (phishing) | Specificity | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 85.58% | 92.14% | 39.39% | 55.19% | 99.02% | 0.8002 | 77,811 | 768 | 13,856 | 9,004 |
| Support Vector Machine | 85.61% | 92.66% | 39.23% | 55.13% | 99.10% | 0.7968 | 77,869 | 710 | 13,891 | 8,969 |
| Decision Tree | 87.78% | 85.08% | 55.49% | 67.17% | 97.17% | 0.8864 | 76,354 | 2,225 | 10,174 | 12,686 |

These are dataset-split results, not a guarantee of real-world performance.
False positives and false negatives occur for every model.

## Risk display and calibration

The models expose `predict_proba`; the UI uses the class-1 output as a
**Model Risk Score** scaled from 0 to 100. It is not presented as a verified
probability. Logistic Regression does not have a separate calibration step;
the SVM uses sigmoid calibration; and Decision Tree outputs leaf class
frequencies. A coarse calibration check on the held-out split does not
establish calibration for URLs outside the dataset distribution. In
particular, the sparse HTTPS training examples prevent meaningful validation
of scores for legitimate HTTPS URLs.

For example, the saved models classify `https://www.google.com/` as phishing
with Logistic Regression (99.28 score), phishing with SVM (79.63), and
legitimate with the Decision Tree (22.56). These are illustrative model
outputs, not security assurances.

## Installation and launch

Python 3.10 or later is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address printed by Streamlit, normally
`http://localhost:8501`. URL scanning is performed on the entered text only;
the application makes no request to the submitted URL.

## Validation

The local validation covered:

- Feature extraction and URL syntax checks, including blank, malformed,
  unsupported-scheme, and invalid-port inputs.
- Loading all three saved model files and matching their feature names/order.
- Reproducing saved held-out metrics and confusion counts.
- Running the Streamlit app and exercising its navigation and scanner.
- Local-only predictions for illustrative legitimate URLs and labeled
  samples from `cleaned_urls.csv`.

No sample website was visited during these checks. See
[`PROJECT_REPORT.md`](PROJECT_REPORT.md) for the detailed test record and
limitations.

## Limitations and future work

- The feature set is lexical only; it does not inspect page content, redirects
  performed by a server, reputation, or DNS/TLS information.
- Suspicious keywords are currently counted by substring, so `security` inside
  `cybersecurity` contributes to the count.
- HTTPS coverage is severely unrepresentative, as described above.
- The low-dimensional feature vectors collide: many distinct URLs produce the
  same feature vector, sometimes with different labels.
- Evaluation uses a random row split; it does not establish performance on
  unseen domains or newer phishing campaigns.
- The SVM preprocessing should be placed inside its calibration-fold estimator
  before future model retraining.

Recommended future work is to add representative labeled legitimate HTTPS
data, improve token-boundary handling, use domain-aware evaluation, validate
calibration on independent data, and retrain/evaluate only after preserving
the current model artifacts.

## Requirements

Runtime and visualization dependencies are listed in `requirements.txt`.
The installed environment used for this validation passed `pip check`.
