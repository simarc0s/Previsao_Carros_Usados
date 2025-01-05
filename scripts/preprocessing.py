import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(train_data, test_data, target_column):
    """
    Pré-processa os dados para Machine Learning:
    - Trata valores ausentes;
    - Codifica colunas categóricas;
    - Normaliza colunas numéricas.
    """
    # Preencher valores ausentes
    train_data.fillna(
        {"fuel_type": "Unknown", "accident": "Unknown", "clean_title": "Unknown"},
        inplace=True,
    )
    test_data.fillna(
        {"fuel_type": "Unknown", "accident": "Unknown", "clean_title": "Unknown"},
        inplace=True,
    )
    train_data.fillna(0, inplace=True)
    test_data.fillna(0, inplace=True)

    # Separar o target
    y_train = train_data[target_column]
    X_train = train_data.drop(columns=[target_column, "id"])
    X_test = test_data.drop(columns=["id"])

    # Identificar colunas categóricas e numéricas
    categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Configurar o ColumnTransformer
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    # Aplicar as transformações
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, y_train, X_test
