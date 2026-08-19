# SmartCare AI No-Show Prediction

SmartCare is a machine learning project developed to predict whether a patient is likely to miss a healthcare appointment. The project includes data preprocessing, exploratory data analysis, feature engineering, model development, model evaluation, explainable AI analysis, and a Streamlit prototype.

The prediction is intended to support hospital staff when planning appointment reminders and follow-up actions. It does not replace human decision-making.

---

## Project Overview

The original dataset contains 1,000 healthcare appointment records and 33 columns. The target was corrected by keeping only appointments with confirmed outcomes.

- `Completed = 0`
- `No-Show = 1`

Scheduled appointments were removed because their outcomes were not yet known. Cancelled appointments were also removed because cancellation is different from both attendance and no-show behaviour.

After this correction, the final modelling dataset contains:

- 875 confirmed appointment records
- 24 input features
- One binary target variable

---

## Project Workflow

```text
Original Appointment Dataset
            │
            ▼
Data Preprocessing and Target Correction
            │
            ▼
Feature Engineering
            │
            ▼
Exploratory Data Analysis
            │
            ▼
Model Development and Comparison
            │
            ▼
Model Evaluation and Calibration
            │
            ▼
Explainable AI and Fairness Analysis
            │
            ▼
Streamlit Prototype
```

---

## Data Preprocessing and Feature Engineering

The preprocessing stage includes:

- Checking missing values
- Checking duplicate records
- Correcting data types
- Reviewing numerical outliers
- Converting appointment dates to datetime
- Removing Scheduled and Cancelled appointments
- Removing identifiers
- Removing post-appointment leakage variables
- One-hot encoding department and diagnosis
- Extracting appointment month and day of week
- Creating `missed_ratio`
- Creating `has_missed_before`

The continuous `waiting_days` feature is retained directly. An additional long-wait flag is not used because waiting days already contains the required information.

The following processed files are created:

```text
data/processed/smartcare_preprocessed_unscaled.csv
data/processed/smartcare_preprocessed_scaled.csv
data/processed/fairness_columns.csv
```

---

## Exploratory Data Analysis

The exploratory analysis reviews patterns related to:

- Appointment outcomes
- Waiting days
- Previous appointments
- Previous missed appointments
- Patient age
- Department
- Diagnosis
- Appointment month
- Appointment day of week

Waiting days shows the strongest numerical association with the no-show target in the processed dataset.

---

## Model Development

The processed dataset is divided using stratification and `random_state = 42`:

| Split | Records |
| --- | ---: |
| Training | 525 |
| Validation | 175 |
| Testing | 175 |

Five-fold stratified cross-validation is applied to the training set.

The following models are compared:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine with RBF kernel

Scaling and feature selection are placed inside the relevant model pipelines. This prevents validation or test information from affecting preprocessing during training.

F1 score is used as the main model-selection metric because it considers both precision and recall. Random Forest achieved the highest training cross-validation F1 score and was selected as the final model.

The final model uses all 24 input features and a fixed decision threshold of `0.39`.

---

## Model Evaluation

The final Random Forest model produced the following test results:

| Metric | Result |
| --- | ---: |
| Accuracy | 58.86% |
| Precision | 58.71% |
| Recall | 91.92% |
| F1 Score | 71.65% |
| ROC-AUC | 56.34% |

The selected threshold provides high recall and identifies most actual no-shows. However, it also produces false-positive alerts. Therefore, predictions should be reviewed by hospital staff before action is taken.

Probability calibration is evaluated using a calibration curve and Brier score. The prototype displays the probability produced by the validation-calibrated model.

---

## Explainable AI

The project uses Feature Importance Analysis for explainability.

### Global Explanation

Permutation importance is used to measure how model performance changes when each feature is shuffled. This provides an overall view of the features used by the model.

### Local Explanation

A one-feature-at-a-time sensitivity method is used to explain an individual prediction. Each selected value is compared with its training-set median while the other values remain fixed.

These results describe model behaviour. They do not prove that a feature causes a patient to miss an appointment.

The project also compares recall and false-positive rates across gender and age groups as part of the fairness review.

---

## Streamlit Prototype

The Streamlit prototype demonstrates how the trained model can be used within an appointment-management workflow.

The prototype includes:

- Staff login and sign-up
- Client creation
- Client search
- Appointment history
- Admission records
- Appointment booking
- Doctor and date selection
- No-show prediction
- Calibrated no-show risk
- Local prediction explanation
- Appointment list
- High-risk appointment filtering
- Attendance confirmation
- Appointment cancellation
- Payment step

The prototype uses the selected Random Forest pipeline, calibrated probability model, selected model information, and training-feature medians.

---

## Repository Structure

```text
smartcare-ai-no-show-prediction/
│
├── data/
│   ├── raw/
│   │   ├── smartcare_ai_dataset_1000.csv
│   │   └── smartcare_ai_dataset_data_dictionary.csv
│   │
│   └── processed/
│       ├── smartcare_preprocessed_unscaled.csv
│       ├── smartcare_preprocessed_scaled.csv
│       └── fairness_columns.csv
│
├── notebooks/
│   ├── Task03_Data_Preprocessing_and_Feature_Engineering.ipynb
│   ├── Task04_Exploratory_Data_Analysis.ipynb
│   ├── Task05_Model_Development_and_Optimization.ipynb
│   ├── Task06_Model_Evaluation.ipynb
│   └── Task07_Explainable_AI.ipynb
│
├── models_checkpoints/
│   ├── selected_model_pipeline.pkl
│   ├── calibrated_probability_model.pkl
│   ├── selected_model_info.json
│   ├── logistic_regression_pipeline.pkl
│   ├── random_forest_pipeline.pkl
│   ├── gradient_boosting_pipeline.pkl
│   ├── svm_rbf_pipeline.pkl
│   ├── training, validation, and test split files
│   ├── Task 06 evaluation result files
│   └── Task 07 explainability result files
│
├── prototype/
│   ├── frontend.py
│   ├── backend.py
│   ├── database.py
│   ├── auth.py
│   ├── style.css
│   ├── requirements.txt
│   └── models/
│
├── reports/
├── src/
│   └── models/
├── .github/
└── README.md
```

---

## Running the Notebooks in Google Colab

Clone the repository:

```python
!git clone https://github.com/NavodyaRupasinghe2003/smartcare-ai-no-show-prediction.git
%cd smartcare-ai-no-show-prediction
```

Run the notebooks in the following order:

```text
Task03_Data_Preprocessing_and_Feature_Engineering.ipynb
        ↓
Task04_Exploratory_Data_Analysis.ipynb
        ↓
Task05_Model_Development_and_Optimization.ipynb
        ↓
Task06_Model_Evaluation.ipynb
        ↓
Task07_Explainable_AI.ipynb
```

The notebooks should be run in order because later tasks use files created by earlier tasks.

---

## Running the Prototype

Create and activate a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the prototype dependencies:

```bash
pip install -r prototype/requirements.txt
```

Create or initialise the prototype database:

```bash
python prototype/database.py
```

Start the Streamlit application:

```bash
python -m streamlit run prototype/frontend.py
```

The application is normally available at:

```text
http://localhost:8501
```

---

## Main Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Google Colab
- Streamlit
- SQLite
- Git and GitHub

---

## Team Collaboration

| Assigned Tasks | Group Member |
| --- | --- |
| Tasks 01 and 02 | Amaya |
| Tasks 03 and 04 | Navodya Rupasinghe |
| Task 05 | Ranudi Nethmini |
| Tasks 06 and 07 | Nuwani Hansika |
| Task 08 | Nisal Damsika |

Each member completed the assigned task and shared the relevant outputs through the project repository. The notebooks, processed datasets, trained model files, evaluation outputs, explainability files, and prototype were combined in the `main` branch.

---

## Important Notes

- Run the notebooks in task order.
- Keep the same feature order when loading the trained model.
- Do not use the test set for model selection or threshold tuning.
- Keep the calibrated model and selected model information with the final pipeline.
- Predictions should be interpreted as decision support, not as confirmed patient behavior.
- Real patient information should not be uploaded to a public repository.

---

## Disclaimer

This project was developed for academic purposes. The model and prototype have not been validated for real clinical deployment. Further testing, security controls, external validation, and healthcare approval would be required before practical use.

