import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score


def select_best_model(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1)
    }

    best_name = None
    best_rmse = float("inf")

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="neg_mean_squared_error",
            cv=5
        )

        rmse = np.sqrt(-scores).mean()

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    best_model.fit(X_train, y_train)

    return best_name, best_model


def save_model(model, scaler, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(
        {"model": model, "scaler": scaler},
        model_path
    )


def load_model(model_path):
    saved = joblib.load(model_path)

    return saved["model"], saved["scaler"]