# 🔧 Bearing Predictive Maintenance Using Machine Learning

> **Machine Learning + Mechanical Engineering | Bearing Fault Diagnosis | Vibration Analytics**

A complete machine-learning workflow for **bearing condition monitoring and fault classification** using vibration-derived features. The project combines mechanical engineering concepts, vibration analysis, exploratory data analysis, supervised learning, model evaluation, and engineering interpretation.

---

## 📌 Project Overview

Bearings are critical components in rotating machinery. Their failure can lead to:

- Unexpected machine downtime
- Production losses
- Increased maintenance cost
- Secondary damage to shafts, gears, and housings
- Safety and reliability issues

The goal of this project is to develop a machine-learning system capable of identifying the **condition of a bearing from vibration-related measurements**.

### Bearing conditions classified

| Class | Description |
|---|---|
| 🟢 `Normal` | Healthy bearing |
| 🔴 `Inner_Race` | Inner-race fault |
| 🟠 `Outer_Race` | Outer-race fault |
| 🟣 `Ball` | Rolling-element / ball fault |

The project compares multiple ML algorithms and analyzes which vibration features contribute most strongly to the prediction.

---

## 🎯 Objectives

1. Analyze bearing vibration-related data.
2. Perform detailed exploratory data analysis (EDA).
3. Identify relationships between vibration features and bearing condition.
4. Train multiple classification models.
5. Compare model performance using several evaluation metrics.
6. Analyze the confusion matrix to understand classification errors.
7. Examine feature importance.
8. Build a foundation for real-time predictive maintenance.
9. Provide a reusable ML pipeline for future real bearing datasets.

---

## 🧠 Machine Learning Workflow

```text
                 Bearing Condition Data
                          │
                          ▼
                 Data Cleaning & QA
                          │
                          ▼
                Exploratory Data Analysis
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
      Statistical Analysis        Visualization
            │                           │
            └─────────────┬─────────────┘
                          ▼
                  Feature Preparation
                          │
                          ▼
                 Train / Test Split
                          │
                          ▼
              ┌───────────────────────┐
              │   Multiple ML Models  │
              ├───────────────────────┤
              │ Logistic Regression   │
              │ KNN                   │
              │ SVM                   │
              │ Random Forest         │
              │ Gradient Boosting     │
              │ XGBoost               │
              └───────────┬───────────┘
                          ▼
                  Model Evaluation
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          ROC-AUC     Confusion     Feature
                      Matrix       Importance
             │            │            │
             └────────────┼────────────┘
                          ▼
                 Bearing Fault Diagnosis
```

---

# 📊 Dataset

The ML-ready dataset contains vibration-derived and operating-condition features.

### Features

| Feature | Meaning |
|---|---|
| `rms` | Root Mean Square vibration |
| `std` | Standard deviation of vibration |
| `peak` | Peak vibration amplitude |
| `peak_to_peak` | Peak-to-peak vibration amplitude |
| `kurtosis` | Measure of impulsive/non-Gaussian behavior |
| `skewness` | Distribution asymmetry |
| `crest_factor` | Peak-to-RMS ratio |
| `shape_factor` | Waveform shape characteristic |
| `rpm` | Rotational speed |
| `motor_load_hp` | Motor load |
| `sampling_rate_hz` | Sampling frequency |

### Target

```text
fault_type
```

with four classes:

```text
Normal
Inner_Race
Outer_Race
Ball
```

---

## ⚠️ Dataset Transparency

**Important:** The CSV used for the current notebook is a **synthetic convenience dataset**, created to demonstrate the complete ML workflow and project structure. It is **not the original CWRU vibration measurements**.

For a research paper, academic submission, or claim of experimental performance, replace the current CSV with a properly processed real bearing dataset such as the **Case Western Reserve University (CWRU) Bearing Dataset** and document the preprocessing and train/test split methodology.

This README intentionally does **not** present the current results as measurements from a real physical bearing experiment.

---

# 🔍 Exploratory Data Analysis

The project includes several EDA visualizations.

## 1. Feature Distributions

The distributions show clear differences between bearing conditions for several important features.

### Key observations

- `rms` differs substantially between healthy and faulty classes.
- `peak` provides strong separation between Normal, Outer Race, Inner Race, and Ball conditions.
- `kurtosis` increases noticeably for faulty conditions.
- `shape_factor` shows substantial overlap between classes.
- `crest_factor` contains extreme values/outliers and should be interpreted carefully.

These observations indicate that vibration features contain useful information for fault classification.

---

## 2. Boxplot Analysis

The boxplots show that:

### RMS

Healthy bearings have a considerably lower central RMS value, while faulty bearings generally show higher vibration energy.

### Peak

Peak vibration provides particularly strong class separation. The Ball and Inner Race conditions show higher peak values than the Normal class.

### Kurtosis

Kurtosis is useful for identifying impulsive vibration behavior associated with faults.

### Shape Factor

There is considerably more overlap between classes, suggesting that this feature alone is unlikely to be sufficient for reliable diagnosis.

---

## 3. Correlation Analysis

The correlation heatmap reveals several important relationships.

### Strong correlations

```text
RMS ↔ STD              ≈ 0.99
Peak ↔ Peak-to-Peak    ≈ 0.96
```

This indicates that some features contain highly overlapping information.

For example:

- RMS and standard deviation are strongly correlated.
- Peak and peak-to-peak amplitude are strongly correlated.

This is important from a feature-engineering perspective because highly correlated variables can provide redundant information.

### Engineering interpretation

Rather than assuming that every feature contributes independently, a future version of the project could perform:

- Feature selection
- PCA
- Recursive feature elimination
- Mutual information analysis
- SHAP-based interpretation

---

# 🤖 Models Used

The notebook evaluates multiple classification algorithms.

### 1. Logistic Regression

A strong baseline model that provides a simple and interpretable reference.

### 2. K-Nearest Neighbors

A distance-based classifier useful for evaluating whether similar vibration-feature patterns belong to the same fault class.

### 3. Support Vector Machine

An effective model for nonlinear class boundaries, particularly with scaled features.

### 4. Random Forest

An ensemble of decision trees capable of capturing nonlinear relationships and feature interactions.

### 5. Gradient Boosting

Sequentially builds trees to improve classification performance.

### 6. XGBoost

A powerful gradient-boosting algorithm commonly used for structured/tabular ML problems.

---

# 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- 5-fold cross-validation
- Confusion matrix
- ROC-AUC

---

## 🏆 ROC-AUC Results

The current evaluation produced the following micro-average ROC-AUC values:

| Model | ROC-AUC |
|---|---:|
| **Logistic Regression** | **0.949** |
| SVM | 0.947 |
| Random Forest | 0.945 |
| XGBoost | 0.945 |
| Gradient Boosting | 0.944 |
| KNN | 0.916 |

### Interpretation

All six models perform substantially above the random-classification baseline.

The strongest ROC-AUC in the current experiment is obtained by:

> **Logistic Regression — AUC = 0.949**

SVM follows very closely with an AUC of **0.947**.

KNN produces the lowest ROC-AUC among the tested models at **0.916**, although this still indicates useful class-discrimination ability.

---

# 🧮 Confusion Matrix Analysis

The current confusion matrix for Logistic Regression is:

| Actual \ Predicted | Ball | Inner Race | Normal | Outer Race |
|---|---:|---:|---:|---:|
| **Ball** | 91 | 23 | 0 | 6 |
| **Inner Race** | 27 | 73 | 0 | 20 |
| **Normal** | 0 | 0 | 120 | 0 |
| **Outer Race** | 7 | 23 | 0 | 90 |

### Test-set accuracy

From the displayed confusion matrix:

```text
Correct predictions = 91 + 73 + 120 + 90
                    = 374

Total predictions   = 480

Accuracy            = 374 / 480
                    ≈ 77.92%
```

### Important observations

**Normal bearings:**  
The model correctly identifies all 120 Normal samples in this test set.

**Inner Race:**  
Some Inner Race samples are classified as Ball or Outer Race.

**Outer Race:**  
A noticeable number of Outer Race samples are classified as Inner Race.

**Ball:**  
Some Ball samples are confused with Inner Race.

### Engineering interpretation

The major classification challenge is distinguishing between **different fault types**, rather than distinguishing healthy from faulty bearings.

This is physically reasonable because different bearing defects can produce overlapping vibration signatures, particularly when operating conditions and vibration features are similar.

---

# ⭐ Feature Importance

The XGBoost feature-importance analysis indicates that the strongest contributors in the current experiment are:

```text
1. peak
2. kurtosis
3. peak_to_peak
4. skewness
5. rms
6. std
7. crest_factor
8. shape_factor
9. motor_load_hp
10. rpm
```

### Most important feature: `peak`

Peak vibration has the largest feature importance in the displayed XGBoost model.

This suggests that impulsive/high-amplitude vibration information is highly useful for separating the bearing conditions in this dataset.

### `kurtosis`

Kurtosis is also highly influential. This is particularly interesting from a mechanical vibration perspective because localized bearing defects can generate impulsive events.

### `peak_to_peak`

Peak-to-peak amplitude is another important feature and is strongly correlated with peak amplitude.

---

# 🔧 Mechanical Engineering Perspective

The project is not simply a generic classification problem.

Bearing faults have characteristic physical effects on machine vibration.

A localized defect can generate repeated impacts as the rolling elements pass over the damaged region.

These impacts can change:

- Vibration amplitude
- Frequency content
- Peak values
- Kurtosis
- Crest factor
- Overall vibration energy

Therefore:

```text
Mechanical fault
      ↓
Change in bearing dynamics
      ↓
Change in vibration signal
      ↓
Feature extraction
      ↓
Machine learning
      ↓
Fault diagnosis
```

This makes the project a combination of:

**Mechanical Engineering + Vibrations + Signal Processing + Machine Learning**

---

# 📁 Project Structure

Recommended GitHub structure:

```text
bearing-predictive-maintenance/
│
├── data/
│   └── bearing_predictive_maintenance_ml_ready.csv
│
├── notebooks/
│   └── bearing_predictive_maintenance_complete.ipynb
│
├── models/
│   ├── best_bearing_fault_model.joblib
│   └── bearing_fault_label_encoder.joblib
│
├── results/
│   ├── model_comparison_results.csv
│   └── cross_validation_results.csv
│
├── images/
│   ├── roc_curves.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── feature_boxplots.png
│   └── feature_distributions.png
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Model Persistence

- Joblib

### Development

- Jupyter Notebook

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/bearing-predictive-maintenance.git
cd bearing-predictive-maintenance
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/bearing_predictive_maintenance_complete.ipynb
```

---

# 📦 requirements.txt

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
joblib
jupyter
```

---

# 🚀 Future Improvements

The current project provides a complete baseline, but there are several ways to make it significantly more advanced.

## 1. Use real vibration signals

Replace the synthetic convenience dataset with raw measurements from:

- CWRU Bearing Dataset
- NASA bearing datasets
- PRONOSTIA/FEMTO dataset
- A self-built experimental test rig

---

## 2. Add FFT analysis

Instead of relying primarily on statistical features:

```text
Raw vibration
      ↓
FFT
      ↓
Frequency spectrum
      ↓
Bearing characteristic frequencies
```

Analyze:

- BPFO — Ball Pass Frequency Outer race
- BPFI — Ball Pass Frequency Inner race
- BSF — Ball Spin Frequency
- FTF — Fundamental Train Frequency

---

## 3. Real-time monitoring

A practical system could use:

```text
Accelerometer
      ↓
ESP32 / DAQ
      ↓
Python
      ↓
Feature extraction
      ↓
Trained ML model
      ↓
Bearing condition
      ↓
Dashboard
```

---

## 4. Deep Learning

Future versions could use:

- 1D CNN
- LSTM
- GRU
- CNN-LSTM
- Autoencoders

A 1D CNN can learn directly from vibration signals with less manual feature engineering.

---

## 5. Remaining Useful Life Prediction

Instead of only predicting:

```text
Normal
Inner Race
Outer Race
Ball
```

the system could estimate:

```text
Estimated Remaining Useful Life = X hours
```

This would move the project from **fault diagnosis** toward full **predictive maintenance**.

---

# ⚠️ Important ML Consideration: Data Leakage

For real vibration datasets, avoid randomly splitting highly overlapping windows from the same recording into training and test sets.

For example:

```text
One vibration recording
        ↓
1000 overlapping windows
        ↓
Random train/test split
```

can produce artificially high performance because nearly identical signal segments may appear in both sets.

A better approach is to split by:

- Recording
- Operating run
- Bearing
- Machine condition
- Experiment

depending on the dataset and research objective.

---

# 📌 Key Takeaways

### From EDA

- Fault classes show different vibration distributions.
- RMS, peak, and kurtosis are particularly informative.
- Several features are strongly correlated.
- Shape factor has substantial class overlap.
- Crest factor contains significant outliers.

### From model evaluation

- Logistic Regression achieved the highest displayed micro-average ROC-AUC: **0.949**.
- SVM was a close second at **0.947**.
- KNN produced the lowest displayed AUC: **0.916**.
- The displayed Logistic Regression confusion matrix gives **77.92% test accuracy**.
- Normal bearing samples were classified particularly well.
- The main errors occur between Ball, Inner Race, and Outer Race conditions.

### From feature importance

The strongest XGBoost features are:

> **Peak → Kurtosis → Peak-to-Peak → Skewness → RMS**

These features provide a useful connection between the ML results and vibration-based bearing diagnostics.

---

# 🎓 Academic Project Statement

A concise description suitable for a project portfolio:

> **Developed a machine-learning-based bearing fault diagnosis system using vibration-derived features. Performed exploratory vibration analysis, feature correlation analysis, multi-model classification, cross-validation, ROC-AUC evaluation, confusion-matrix analysis, and XGBoost feature-importance analysis to distinguish Normal, Inner Race, Outer Race, and Ball bearing conditions.**

---

# 👨‍💻 Author

**Your Name**

Mechanical Engineering | Machine Learning | Predictive Maintenance

---

## ⭐ If you found this project useful

If this project helped you learn something, consider giving the repository a ⭐ on GitHub.

---

### Disclaimer

This project is intended for educational and portfolio purposes. The current demonstration dataset is synthetic and should not be interpreted as measurements from a physical bearing experiment. For engineering or industrial deployment, the model must be validated using representative real-world vibration data and appropriate safety/reliability procedures.
