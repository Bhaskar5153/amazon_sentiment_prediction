import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, solver='lbfgs', class_weight="balanaced", random_state=42
        ),
        "Linear SVC": LinearSVC(
            max_iter=2000, class_weight="balanced", random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        )
    }



def train_all_models(models: dict, X_train, y_train):
    fitted = {}
    for name, model in models.items():
        print(f"..... Training {name} .....")
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted



def hyperparameter_tuning(X_train, y_train, model_type: str, scoring: str, cv: int = 5):
    if model_type == "Linear SVC":
        estimator = LinearSVC(
            max_iter=2000, class_weight="balanced", random_state=42
        )
        param_grid = {"c": [0.01, 0.1, 1, 10]}

    elif model_type == "Logistic Regression":
        estimator = LogisticRegression(
            max_iter=5000, solver='lbfgs', class_weight="balanaced", random_state=42
        )
        param_grid = {"c": [0.01, 0.1, 1, 10]}

    elif model_type == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        )
        param_grid = {"c": [0.01, 0.1, 1, 10]}

    else:
        raise ValueError(f"Unknown model type")
    
    gs = GridSearchCV(
        estimator=estimator, 
        param_grid=param_grid, 
        scoring=scoring, cv=cv, n_jobs=-1, verbose=1
    )

    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_



def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Model saved to -> {path}")


def load_model(path: str):
    return joblib.load(path)

    



        

    




