import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# Load preprocessed data
df = pd.read_csv("cleaned_breast_cancer_data.csv")

# Train-test split
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and hyperparameter tuning
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
}

# Hyperparameter tuning
param_grid = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"n_estimators": [50, 100], "max_depth": [10, 20], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]},
}

best_models = {}

for name, model in models.items():
    if name in param_grid:
        grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_scaled if name == "Logistic Regression" else X_train, y_train)
        best_models[name] = grid_search.best_estimator_
    else:
        model.fit(X_train_scaled if name == "Logistic Regression" else X_train, y_train)
        best_models[name] = model

# Save trained models
import joblib
for name, model in best_models.items():
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")
