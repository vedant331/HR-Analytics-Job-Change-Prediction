# HR Analytics – Job Change Prediction

<p align="center">

  **Predicting whether an employee is likely to switch jobs using Machine Learning.**

  <br/>

  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
  ![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
  ![XGBoost](https://img.shields.io/badge/XGBoost-189E5B?style=for-the-badge)
  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</p>

## 📌 Project Overview

Employee attrition is an important challenge for organizations. Identifying employees who are likely to look for a new job can help companies understand workforce trends and make better talent management decisions.

This project uses **Machine Learning classification techniques** to predict whether an individual is likely to change their current job based on demographic, educational, professional, and company-related information.

The project explores multiple machine learning models, handles missing data and class imbalance, performs hyperparameter tuning, and evaluates the final model on unseen test data.

## 🎯 Project Objectives

- Analyze employee and company-related data.
- Perform Exploratory Data Analysis (EDA).
- Handle missing values and categorical features.
- Prepare data for machine learning.
- Train and compare multiple classification models.
- Handle class imbalance.
- Optimize model performance using hyperparameter tuning.
- Evaluate the final model using classification metrics.
- Generate predictions for unseen employee records.

## 📂 Dataset Information

The project uses the HR Analytics Job Change Prediction dataset.

| Dataset | Description |
|---|---|
| `aug_train.csv` | Training dataset containing employee information and target labels |
| `aug_test.csv` | Unseen employee records used for prediction |

**Target variable:**

- `0` – Employee is not looking for a job change.
- `1` – Employee is looking for a job change.

**Features include:**

- City and city development index
- Gender
- Relevant work experience
- Enrolled university
- Education level
- Major discipline
- Professional experience
- Company size and company type
- Time since last job change
- Training hours

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Data preprocessing and model evaluation |
| XGBoost | Gradient boosting classification |
| Imbalanced-learn | Class imbalance handling, where applicable |
| Joblib | Saving and loading the trained model |
| Jupyter Notebook | Development environment |

## ⚙️ Project Workflow

### 1. Data Understanding and Exploration
- Loaded and inspected the dataset.
- Analyzed data types and missing values.
- Examined the target variable distribution.
- Explored numerical and categorical features.

### 2. Data Preprocessing
- Removed unnecessary identifier columns.
- Handled missing values.
- Converted experience-related categorical values into numerical values.
- Applied median imputation to numerical features.
- Applied most-frequent imputation to categorical features.
- Used StandardScaler for numerical features.
- Applied OneHotEncoder for categorical features.

### 3. Handling Class Imbalance

The target variable contains more employees belonging to class `0` than class `1`.

To address this imbalance, balanced classification approaches and class weighting were explored.

### 4. Model Building

The following classification models were explored:

- Logistic Regression
- Balanced Logistic Regression
- Random Forest
- Balanced Random Forest
- XGBoost
- Balanced XGBoost

### 5. Hyperparameter Tuning

RandomizedSearchCV was used to identify suitable XGBoost hyperparameters.

The selected parameters included:

| Parameter | Value |
|---|---:|
| n_estimators | 100 |
| max_depth | 5 |
| learning_rate | 0.05 |
| min_child_weight | 3 |
| subsample | 0.9 |
| colsample_bytree | 0.7 |

### 6. Threshold Optimization

Instead of relying only on the default classification threshold of 0.5, a separate validation dataset was used to select a threshold based on F1-score.

Selected classification threshold: **0.52239**

### 7. Model Evaluation

The final model was evaluated using an untouched test dataset.

## 📊 Final Model Performance

**Final Model: Tuned XGBoost Classifier**

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 79.54% |
| Precision (Class 1) | 0.57 |
| Recall (Class 1) | 0.78 |
| F1-Score (Class 1) | 0.65 |
| ROC-AUC | 0.8204 |

### Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2307 | 570 |
| Actual 1 | 214 | 741 |

The model achieves a recall of approximately **78% for employees looking for a job change**, helping identify a substantial proportion of positive cases.

## 📁 Project Structure

```text
HR-Analytics-Job-Change-Prediction/
│
├── Data1.ipynb
├── aug_train.csv
├── aug_test.csv
├── job_change_model.pkl
├── submission.csv
├── .gitignore
└── README.md
```

| File | Description |
|---|---|
| `Data1.ipynb` | Complete machine learning workflow |
| `aug_train.csv` | Training dataset |
| `aug_test.csv` | Unseen dataset |
| `job_change_model.pkl` | Saved model pipeline and classification threshold |
| `submission.csv` | Predictions generated for the unseen dataset |
| `.gitignore` | Excludes unnecessary files from Git |

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/vedant331/HR-Analytics-Job-Change-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd HR-Analytics-Job-Change-Prediction
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the Notebook

Open `Data1.ipynb` and execute the cells in sequence.

## 💾 Loading the Saved Model

The trained model and optimized classification threshold are stored together using Joblib.

```python
import joblib

artifact = joblib.load('job_change_model.pkl')

loaded_model = artifact['model']
loaded_threshold = artifact['threshold']
```

Generate predictions:

```python
probabilities = loaded_model.predict_proba(new_data)[:, 1]

predictions = (probabilities >= loaded_threshold).astype(int)
```

Ensure that new data receives the same preprocessing transformations used during model training.

## 🔮 Future Improvements

- Experiment with additional machine learning algorithms.
- Perform more extensive feature engineering.
- Explore explainable AI techniques such as SHAP.
- Improve model performance through advanced hyperparameter optimization.
- Develop an interactive web application for real-time predictions.
- Monitor model performance on new employee data.

## 👨‍💻 Author

**Vedant Kadam**

B.Sc. Information Technology Student

GitHub: [@vedant331](https://github.com/vedant331)

---

<p align="center">

  **Built as a hands-on Machine Learning project.**

  ⭐ If you find this project useful, consider giving it a star!

</p>
