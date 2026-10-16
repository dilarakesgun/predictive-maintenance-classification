# Predictive Maintenance — Machine Failure Prediction

End-to-end machine-learning project that predicts machine failures from sensor
readings, using the AI4I 2020 Predictive Maintenance dataset. The focus is a
correct, leakage-free pipeline and honest evaluation on imbalanced data.

## Overview

Given sensor measurements (temperatures, rotational speed, torque, tool wear) and
the product quality type, the goal is to predict whether a machine will fail.
Because failures are rare (~3.4% of records), the project treats this as an
**imbalanced binary classification** problem and evaluates models with
**precision / recall**, not accuracy.

## Dataset

- **AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository
  (10,000 rows, 14 columns).
- Target: `Machine failure` (0 = healthy, 1 = failure). Failure rate ≈ 3.4%.
- Link: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Pipeline

1. **Load & inspect** — no missing values; checked class balance.
2. **Clean** — dropped identifier columns (`UDI`, `Product ID`) and encoded the
   categorical `Type` (L/M/H) with one-hot encoding.
3. **Prevent data leakage** — dropped the individual failure-mode columns
   (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`). The target is derived from these, so using
   them as features would leak the answer. Only true sensor readings, known
   *before* a failure, are kept as features.
4. **Split** — stratified train / validation / test split (60 / 20 / 20) to keep
   the ~3.4% failure ratio in every partition.
5. **Scale** — `StandardScaler` fit on the **training set only**, then applied to
   validation and test.
6. **Handle imbalance** — `RandomOverSampler` applied to the **training set only**,
   so validation and test still reflect the real-world failure rate.
7. **Model & compare** — Logistic Regression (baseline), KNN, and a small neural
   network (Keras). Model selection on the validation set.
8. **Final evaluation** — the chosen model evaluated once on the held-out test set.
   
## Object-Oriented Design

The pipeline is implemented as modular, reusable classes rather than a flat script:

| Class | Responsibility |
|-------|----------------|
| `DataLoader` | Load the CSV, drop identifier/leakage columns, one-hot encode `Type` |
| `Preprocessor` | Train/validation/test split, scaling, and oversampling (fit on the training set only) |
| `BaseModel` | Shared `train` / `predict` / `evaluate` logic |
| `LogRegModel`, `KNNModel`, `NeuralNetModel` | Subclasses of `BaseModel`; the neural network overrides `train` and `predict` for Keras |

This isolates each responsibility, makes the models interchangeable through a common
interface (inheritance + polymorphism), and keeps the fitted scaler reusable on new data.
## Models & Results (failure class, recall-focused)

Validation set:

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Logistic Regression | 0.15 | 0.82 | 0.26 |
| KNN | 0.36 | 0.62 | 0.45 |
| **Neural Network** | 0.31 | **0.85** | **0.46** |

Held-out **test set** (Neural Network):

| Metric (failure class) | Value |
|------------------------|-------|
| Recall | **0.88** (caught 60 of 68 failures) |
| Precision | 0.34 |
| F1 | 0.49 |

Test performance closely matches validation, indicating the model generalizes
rather than overfits.

## Key decision

This is predictive maintenance, where a **missed failure** is far more costly than
a **false alarm**. Recall was therefore prioritized, and the neural network was
chosen for the highest failure recall (0.88 on test, missing only 8 of 68
failures). If false alarms were the dominant cost, the higher-precision KNN would
be preferable — the right model depends on the cost trade-off.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, TensorFlow/Keras,
Matplotlib, Seaborn.

## Notebooks

- **`MachineFailurePrediction_OOP.ipynb`** — the main, object-oriented pipeline (recommended)
- `MachineFailurePrediction_notes.ipynb` — earlier version with detailed step-by-step learning notes

## How to run

1. Download `ai4i2020.csv` from the UCI link above (or use the copy in this repo).
2. Open `MachineFailurePrediction_OOP.ipynb` in Google Colab or Jupyter.
3. Run the cells top to bottom.

## Limitations & next steps

- The neural network still produces many false alarms (low precision); tuning the
  decision threshold or trying class-weighted loss could improve the balance.
- Only three models were compared; tree-based models (Random Forest, XGBoost)
  are common strong baselines for tabular data and would be a natural next step.
- Hyperparameters were kept simple; a systematic search could improve results.
