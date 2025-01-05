import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(train_data, test_data, target_column, test_size=0.2):
    """
    Pré-processa os dados para Machine Learning:
    - Trata valores ausentes;
    - Codifica colunas categóricas;
    - Normaliza colunas numéricas;
    - Divide o conjunto de treino em treino e validação, se necessário.

    Parâmetros:
    - train_data: DataFrame com os dados de treino.
    - test_data: DataFrame com os dados de teste.
    - target_column: Nome da coluna alvo (target).
    - test_size: Proporção dos dados de treino usada para validação (0 a 1).
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

    # Dividir os dados de treino (opcional)
    if test_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42
        )
    else:
        X_val, y_val = None, None

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
    if X_val is not None:
        X_val = transformer.transform(X_val)
    X_test = transformer.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test
