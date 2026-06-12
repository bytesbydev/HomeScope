from src.data_preprocessing import preprocess_pipeline
from src.train_model import select_best_model, save_model
from src.evaluate_model import evaluate_model

DATA_PATH = "data/housing.csv"
MODEL_PATH = "models/linear_regression_model.pkl"


def main():

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = (
        preprocess_pipeline(DATA_PATH)
    )

    # Train and select the best model
    model_name, model = select_best_model(
        X_train,
        y_train
    )

    # Save model and scaler
    save_model(
        model,
        scaler,
        MODEL_PATH
    )

    # Evaluate model
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Print summary
    print("\nBest Model")
    print("-" * 40)
    print(f"Model Name                    : {model_name}")
    print(f"Mean Absolute Error (MAE)     : {metrics['MAE']:.4f}")
    print(f"Mean Squared Error (MSE)      : {metrics['MSE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"R-squared Score (R²)          : {metrics['R2']:.4f}")


if __name__ == "__main__":
    main()