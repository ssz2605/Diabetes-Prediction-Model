"""
Diabetes Prediction Model Training Script
Author: Shreya Saxena
GitHub: https://github.com/ssz2605/Diabetes-Prediction-Model

This script trains 3 machine learning models on the Pima Indians Diabetes Dataset:
1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import joblib
import os
from pathlib import Path

# Configure display settings
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print(" DIABETES PREDICTION MODEL - TRAINING PIPELINE")
print("=" * 70)

# ============================================================
# STEP 1: DATA LOADING AND EXPLORATION
# ============================================================
print("\n STEP 1: Loading and Exploring Data...")

# Define paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'diabetes.csv'
MODEL_DIR = BASE_DIR / 'models'

# Create models directory if it doesn't exist
MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
    print(f" Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f" Error: Dataset not found at {DATA_PATH}")
    exit(1)

# Display basic info
print("\n Dataset Information:")
print(df.info())
print("\n Statistical Summary:")
print(df.describe())

# Check for missing values
print("\n Missing Values:")
print(df.isnull().sum())

# Class distribution
print("\n Target Variable Distribution:")
print(df['Outcome'].value_counts())
print(f"   Diabetes: {df['Outcome'].sum()} ({df['Outcome'].sum()/len(df)*100:.1f}%)")
print(f"   No Diabetes: {(df['Outcome']==0).sum()} ({(df['Outcome']==0).sum()/len(df)*100:.1f}%)")

# ============================================================
# STEP 2: FEATURE ENGINEERING AND PREPROCESSING
# ============================================================
print("\n STEP 2: Preprocessing Features...")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle zero values in medical features (replace with median)
features_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\n  Handling zero values in medical features...")
for feature in features_to_check:
    zero_count = (X[feature] == 0).sum()
    if zero_count > 0:
        median_val = X[X[feature] != 0][feature].median()
        X[feature].replace(0, median_val, inplace=True)
        print(f"   {feature}: Replaced {zero_count} zero values with median ({median_val:.1f})")

# Split data into train and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n Data split completed:")
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Features scaled using StandardScaler")

# ============================================================
# STEP 3: MODEL TRAINING AND EVALUATION
# ============================================================
print("\n STEP 3: Training Machine Learning Models...")

# Dictionary to store models and their results
models = {}
results = {}

# ---- Model 1: Logistic Regression ----
print("\n  Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Cross-validation
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"   Cross-validation scores: {lr_cv_scores}")
print(f"   Mean CV Score: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)

# Metrics
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_pred_proba_lr[:, 1])

models['Logistic Regression'] = lr_model
results['Logistic Regression'] = {
    'accuracy': lr_accuracy,
    'precision': lr_precision,
    'recall': lr_recall,
    'f1': lr_f1,
    'auc': lr_auc,
    'y_pred': y_pred_lr,
    'y_pred_proba': y_pred_proba_lr
}

print(f"    Logistic Regression Results:")
print(f"      Accuracy:  {lr_accuracy:.4f}")
print(f"      Precision: {lr_precision:.4f}")
print(f"      Recall:    {lr_recall:.4f}")
print(f"      F1-Score:  {lr_f1:.4f}")
print(f"      AUC:       {lr_auc:.4f}")

# ---- Model 2: Random Forest Classifier ----
print("\n Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"   Cross-validation scores: {rf_cv_scores}")
print(f"   Mean CV Score: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)

# Metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf[:, 1])

models['Random Forest'] = rf_model
results['Random Forest'] = {
    'accuracy': rf_accuracy,
    'precision': rf_precision,
    'recall': rf_recall,
    'f1': rf_f1,
    'auc': rf_auc,
    'y_pred': y_pred_rf,
    'y_pred_proba': y_pred_proba_rf
}

print(f"      Random Forest Results:")
print(f"      Accuracy:  {rf_accuracy:.4f}")
print(f"      Precision: {rf_precision:.4f}")
print(f"      Recall:    {rf_recall:.4f}")
print(f"      F1-Score:  {rf_f1:.4f}")
print(f"      AUC:       {rf_auc:.4f}")

# ---- Model 3: Support Vector Machine (SVM) ----
print("\n   Training Support Vector Machine (SVM)...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Cross-validation
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"   Cross-validation scores: {svm_cv_scores}")
print(f"   Mean CV Score: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std():.4f})")

# Predictions
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)

# Metrics
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_auc = roc_auc_score(y_test, y_pred_proba_svm[:, 1])

models['SVM'] = svm_model
results['SVM'] = {
    'accuracy': svm_accuracy,
    'precision': svm_precision,
    'recall': svm_recall,
    'f1': svm_f1,
    'auc': svm_auc,
    'y_pred': y_pred_svm,
    'y_pred_proba': y_pred_proba_svm
}

print(f"     SVM Results:")
print(f"      Accuracy:  {svm_accuracy:.4f}")
print(f"      Precision: {svm_precision:.4f}")
print(f"      Recall:    {svm_recall:.4f}")
print(f"      F1-Score:  {svm_f1:.4f}")
print(f"      AUC:       {svm_auc:.4f}")

# ============================================================
# STEP 4: MODEL COMPARISON AND SELECTION
# ============================================================
print("\n STEP 4: Model Comparison...")

# Create comparison dataframe
comparison_df = pd.DataFrame(results).T
print("\n Model Performance Comparison:")
print(comparison_df[['accuracy', 'precision', 'recall', 'f1', 'auc']])

# Select best model based on accuracy
best_model_name = comparison_df['accuracy'].idxmax()
best_model = models[best_model_name]
best_accuracy = comparison_df.loc[best_model_name, 'accuracy']

print(f"\n  Best Model: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# ============================================================
# STEP 5: SAVE BEST MODEL AND SCALER
# ============================================================
print("\n STEP 5: Saving Best Model and Scaler...")

model_save_path = MODEL_DIR / 'best_diabetes_model.pkl'
scaler_save_path = MODEL_DIR / 'scaler.pkl'

joblib.dump(best_model, model_save_path)
joblib.dump(scaler, scaler_save_path)

print(f"Model saved to: {model_save_path}")
print(f"Scaler saved to: {scaler_save_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\n Summary:")
print(f"   Best Model: {best_model_name}")
print(f"   Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   Features Used: {X.shape[1]}")
print(f"   Training Samples: {X_train.shape[0]}")
print(f"   Test Samples: {X_test.shape[0]}")
print(f"\n Next Steps:")
print(f"   1. Run: python app.py")
print(f"   2. Open: http://localhost:5000")
print(f"   3. Enter health metrics and get predictions!")

print("\n" + "=" * 70)
