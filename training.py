import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model(X, y, model_type="random_forest"):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        )

    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")

    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred
    }
