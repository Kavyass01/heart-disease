import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """
    Handle missing values and duplicates
    """
    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    return df


def encode_features(df):
    """
    Convert categorical columns to numeric
    """
    df = df.copy()

    mappings = {
        "Sex": {"Female": 0, "Male": 1},
        "Smoking": {"No": 0, "Yes": 1},
        "Diabetes": {"No": 0, "Yes": 1}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    return df


def split_data(df, target_column="Heart Attack"):
    """
    Split dataset into train and test
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_features(X_train, X_test):
    """
    Scale numerical features
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


def preprocess_pipeline(
    file_path,
    target_column="Heart Attack"
):
    """
    Full preprocessing pipeline
    """

    df = load_data(file_path)

    df = clean_data(df)

    df = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column
    )

    (
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


def prepare_single_input(data_dict):
    """
    Prepare a single patient record
    for prediction
    """

    df = pd.DataFrame([data_dict])

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].replace({
            "Female": 0,
            "Male": 1
        })

    if "Smoking" in df.columns:
        df["Smoking"] = df["Smoking"].replace({
            "No": 0,
            "Yes": 1
        })

    if "Diabetes" in df.columns:
        df["Diabetes"] = df["Diabetes"].replace({
            "No": 0,
            "Yes": 1
        })

    return df
