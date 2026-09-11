# Cardiovascular Disease Prediction Using Machine Learning

## 📌 Project Overview

Cardiovascular diseases (CVDs) are among the major causes of death worldwide. Early prediction of cardiovascular disease can help identify high-risk individuals and support timely medical intervention.

This project develops a **machine learning-based cardiovascular disease prediction system** that analyzes patient health and lifestyle information to predict whether a person is likely to have cardiovascular disease.

The project uses several machine learning algorithms and compares their performance using standard evaluation metrics such as **Accuracy, Precision, Recall, and F1-Score**.

---

## 🎯 Objectives

* Predict the presence of cardiovascular disease from patient health data.
* Perform data preprocessing and cleaning.
* Create useful medical and statistical features.
* Train multiple machine learning classification models.
* Compare the performance of different algorithms.
* Identify important features contributing to cardiovascular disease prediction.
* Build a reproducible machine learning workflow using Python and Google Colab.

---

## 📊 Dataset

The dataset contains **70,000 patient records** and includes demographic, physical, blood-pressure, biochemical, and lifestyle information.

### Dataset Features

| Feature       | Description               |
| ------------- | ------------------------- |
| `id`          | Unique patient identifier |
| `age`         | Age in days               |
| `gender`      | Gender                    |
| `height`      | Height in cm              |
| `weight`      | Weight in kg              |
| `ap_hi`       | Systolic blood pressure   |
| `ap_lo`       | Diastolic blood pressure  |
| `cholesterol` | Cholesterol level         |
| `gluc`        | Glucose level             |
| `smoke`       | Smoking status            |
| `alco`        | Alcohol consumption       |
| `active`      | Physical activity         |
| `cardio`      | Target variable           |

### Target Variable

* `0` → No cardiovascular disease
* `1` → Cardiovascular disease

The raw dataset is not included in this repository unless its redistribution rights have been verified.

---

## 🔄 Data Preprocessing

The following preprocessing operations are performed:

1. Load the dataset using Pandas.
2. Check for missing values.
3. Check for duplicate records.
4. Convert age from days to years.
5. Remove unrealistic height values.
6. Remove unrealistic weight values.
7. Remove unrealistic blood-pressure values.
8. Remove records where systolic blood pressure is not greater than diastolic blood pressure.
9. Separate input features and target variable.
10. Split the data into training and testing sets using stratified sampling.

---

## 🧮 Feature Engineering

Additional features are created to improve the representation of patient health information.

### Age in Years

```text
age_years = age / 365.25
```

### Body Mass Index

```text
BMI = weight / (height / 100)²
```

### Pulse Pressure

```text
pulse_pressure = ap_hi - ap_lo
```

### Mean Arterial Pressure

```text
MAP = ap_lo + pulse_pressure / 3
```

### Blood Pressure Ratio

```text
bp_ratio = ap_hi / ap_lo
```

Additional interaction and nonlinear features include:

* `age_squared`
* `age_cubed`
* `weight_height_ratio`
* `ap_hi_age`
* `ap_lo_age`

The original `id` field is excluded from model training because it does not represent a meaningful medical characteristic.

---

## 🤖 Machine Learning Models

The project compares the following classification algorithms:

### 1. Logistic Regression

A linear classification algorithm used as a baseline model.

### 2. K-Nearest Neighbors (KNN)

Classifies patients based on the similarity of their feature values to neighboring observations.

### 3. Linear Support Vector Machine

Uses a linear decision boundary to classify cardiovascular disease cases.

### 4. Decision Tree

Creates a tree-like structure of decision rules based on patient characteristics.

### 5. Random Forest

Combines multiple decision trees to improve generalization and reduce overfitting.

### 6. Extra Trees Classifier

Uses an ensemble of randomized decision trees to perform classification.

---

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

### Test Results

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Extra Trees         |   74.24% |    76.04% | 69.98% |   72.88% |
| Random Forest       |   73.89% |    75.72% | 69.51% |   72.48% |
| Linear SVM          |   73.68% |    75.05% | 70.10% |   72.49% |
| Logistic Regression |   73.68% |    74.86% | 70.45% |   72.59% |
| KNN                 |   72.82% |    73.84% | 69.77% |   71.75% |
| Decision Tree       |   72.59% |    75.10% | 66.70% |   70.65% |

**Best-performing model in this evaluation:** Extra Trees Classifier with **74.24% test accuracy**.

> Note: These are held-out test results from the project workflow. Machine learning performance can vary depending on preprocessing, train/test split, random seed, and hyperparameter settings.

---

## 🏗️ Project Workflow

```text
                 ┌─────────────────────┐
                 │    Patient Dataset  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 │ Cleaning & Validation│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Engineering │
                 │ BMI, MAP, BP Ratio   │
                 │ Age & BP Features    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Train/Test Split    │
                 └──────────┬──────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │       Machine Learning Models          │
        ├────────────────────────────────────────┤
        │ Logistic Regression                    │
        │ KNN                                    │
        │ Linear SVM                             │
        │ Decision Tree                          │
        │ Random Forest                          │
        │ Extra Trees                            │
        └──────────────────┬─────────────────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │ Model Evaluation    │
                 │ Accuracy            │
                 │ Precision           │
                 │ Recall              │
                 │ F1-Score            │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Prediction & Results│
                 └─────────────────────┘
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib

### Development Environment

* Google Colab
* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
cardiovascular-disease-prediction-project/
│
├── README.md
├── requirements.txt
├── cardiovascular_prediction.py
├── cardiovascular_final_results.csv
├── random_forest_feature_importance.csv
├── accuracy_comparison.png
│
├── results/
│   └── accuracy_comparison.png
│
└── data/
    └── README.md
```

If the complete Google Colab notebook is included:

```text
├── Cardiovascular_Disease_Prediction.ipynb
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Raghavendra582/cardiovascular-disease-prediction-project.git
```

Move into the project directory:

```bash
cd cardiovascular-disease-prediction-project
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the Python script:

```bash
python cardiovascular_prediction.py
```

Or open the `.ipynb` notebook using:

* Google Colab
* Jupyter Notebook
* JupyterLab

---

## 📌 Output

The project generates:

* Model accuracy comparison
* Precision, recall, and F1-score results
* Cardiovascular disease predictions
* Feature importance results
* Accuracy comparison visualization

---

## 🔍 Feature Importance

The Random Forest model is used to estimate the relative importance of input features.

Feature importance can help identify which patient characteristics contribute most to the model's predictions.

The results are saved in:

```text
random_forest_feature_importance.csv
```

---

## 🚀 Future Improvements

Future versions of the project could include:

* Hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
* XGBoost and LightGBM models.
* Neural network-based prediction.
* Explainable AI using SHAP.
* Cross-validation.
* ROC-AUC and Precision-Recall analysis.
* A web application using Flask or Streamlit.
* Real-time cardiovascular disease prediction.
* Improved handling of class imbalance.
* Model calibration and clinical validation.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

The predictions produced by this machine learning model should **not be used as a medical diagnosis**. Cardiovascular disease assessment should be performed by qualified healthcare professionals using appropriate clinical information and medical testing.

---

## 👨‍💻 Author

**Raghavendra**

### Project Title

**Cardiovascular Disease Prediction Using Machine Learning**

---

## ⭐ GitHub

If you find this project useful, consider giving the repository a ⭐.

Repository:

https://github.com/Raghavendra582/cardiovascular-disease-prediction-project
