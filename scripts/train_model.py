import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def load_data():
    """
    Load the preprocessed data (scaled X_train, X_test, y_train, y_test).
    """
    X_train_scaled = pd.read_csv("X_train_scaled.csv")
    X_test_scaled = pd.read_csv("X_test_scaled.csv")
    y_train = pd.read_csv("y_train.csv")
    y_test = pd.read_csv("y_test.csv")

    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train_scaled, y_train):
    """
    Train a Linear Regression model using the training data.

    Parameters:
        X_train_scaled (DataFrame): The scaled training features.
        y_train (Series): The training target variable.

    Returns:
        model: The trained Linear Regression model.
    """

    model = LinearRegression()

    model.fit(X_train_scaled, y_train)

    train_score = model.score(X_train_scaled, y_train)
    print(f"Training R^2 score: {train_score}")

    return model


def save_model(model):
    """
    Save the trained model to a file using joblib.

    Parameters:
        model: The trained machine learning model.
    """
    joblib.dump(model, "trained_model.pkl")
    print("Model saved successfully as 'trained_model.pkl'")


def main():
    X_train_scaled, X_test_scaled, y_train, y_test = load_data()

    model = train_model(X_train_scaled, y_train)

    save_model(model)


if __name__ == "__main__":
    main()
