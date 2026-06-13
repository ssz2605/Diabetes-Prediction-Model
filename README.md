# Diabetes Prediction Model

An end-to-end Machine Learning project for predicting diabetes using the Pima Indians Diabetes Dataset. The project includes data preprocessing, exploratory data analysis (EDA), model training with Logistic Regression, Random Forest, and SVM, hyperparameter tuning using GridSearchCV, model evaluation, and deployment through a Flask web application for real-time predictions.

## Features
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Scaling with StandardScaler
- Logistic Regression, Random Forest & SVM
- Cross-Validation & Hyperparameter Tuning
- Model Evaluation (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Feature Importance Analysis
- Flask-Based Web Application
- Real-Time Diabetes Prediction
- Prediction Confidence Score

## Tech Stack
**Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Flask, Joblib**

## Dataset
**Pima Indians Diabetes Dataset**

### Features
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target
- **0** → No Diabetes
- **1** → Diabetes

## Installation

```bash
git clone https://github.com/yourusername/Diabetes-Prediction-Model.git
cd Diabetes-Prediction-Model
pip install -r requirements.txt
```

## Run Application

```bash
python app.py
```

Open your browser and visit:

```text
http://localhost:5000
```
