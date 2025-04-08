import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model_performance(y_test, y_pred):
    """
    Function to evaluate the model using metrics such as MSE and R-squared.

    Parameters:
    y_test (array-like): True target values.
    y_pred (array-like): Predicted target values from the model.

    Returns:
    None
    """
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")

    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2}")


def plot_residuals(y_test, y_pred):
    """
    Function to plot residuals to check the assumptions of linear regression.

    Parameters:
    y_test (array-like): True target values.
    y_pred (array-like): Predicted target values from the model.

    Returns:
    None
    """
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, color="blue")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.show()


def compare_model_performance(model, X_train, X_test, y_train, y_test):
    """
    Function to compare model performance with different feature sets or preprocessing steps.

    Parameters:
    model (sklearn model): Trained model.
    X_train (array-like): Training feature data.
    X_test (array-like): Test feature data.
    y_train (array-like): Training target data.
    y_test (array-like): Test target data.

    Returns:
    None
    """
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate the performance
    evaluate_model_performance(y_test, y_pred)

    # Plot residuals
    plot_residuals(y_test, y_pred)


def main():
    """
    Main function to execute model evaluation steps.

    Parameters:
    None

    Returns:
    None
    """


if __name__ == "__main__":
    main()
