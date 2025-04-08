import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(file_path):
    """
    Loads the dataset from the provided file path.
    Args:
    - file_path: str, path to the CSV file containing the data.
    Returns:
    - DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path)
    print(f"Data loaded with shape: {data.shape}")
    return data


def split_features_target(data):
    """
    Splits the dataset into features (X) and target (y).
    Args:
    - data: DataFrame, the loaded dataset.
    Returns:
    - X: DataFrame, the feature columns.
    - y: Series, the target variable.
    """
    X = data.drop("medv", axis=1)
    y = data["medv"]
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def scale_features(X_train, X_test):
    """
    Scales the features using StandardScaler.
    Args:
    - X_train: DataFrame, the training features.
    - X_test: DataFrame, the testing features.
    Returns:
    - X_train_scaled: DataFrame, the scaled training features.
    - X_test_scaled: DataFrame, the scaled testing features.
    - scaler: StandardScaler object, the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaled features: {X_train_scaled.shape}, {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler


def save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler):
    """
    Saves the scaled datasets and the scaler to files.
    Args:
    - X_train_scaled: Scaled training features.
    - X_test_scaled: Scaled testing features.
    - y_train: Target variable for training.
    - y_test: Target variable for testing.
    - scaler: Fitted scaler object.
    """
    pd.DataFrame(X_train_scaled).to_csv("X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv("X_test_scaled.csv", index=False)
    pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

    joblib.dump(scaler, "scaler.pkl")
    print("Data and scaler saved successfully.")


def main():
    file_path = "../data/BostonHousing.csv"
    data = load_data(file_path)

    X, y = split_features_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train/test split: {X_train.shape}, {X_test.shape}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler)


if __name__ == "__main__":
    main()
