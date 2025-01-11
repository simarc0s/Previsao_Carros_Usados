import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(
    train_data, test_data, target_column, test_size=0.2, rare_threshold=0.01
):
    """
    Pré-processa os dados para Machine Learning:
    - Trata valores ausentes;
    - Reduz categorias raras;
    - Codifica colunas categóricas;
    - Normaliza colunas numéricas;
    - Seleciona features com alta relevância para a variável alvo;
    - Divide o conjunto de treino em treino e validação, se necessário.

    Parâmetros:
    - train_data: DataFrame com os dados de treino.
    - test_data: DataFrame com os dados de teste.
    - target_column: Nome da coluna alvo (target).
    - test_size: Proporção dos dados de treino usada para validação (0 a 1).
    - rare_threshold: Proporção mínima para categorias raras (valores de 0 a 1).
    """
    # Preencher valores ausentes
    train_data.fillna(0, inplace=True)
    test_data.fillna(0, inplace=True)

    # Separar o target
    y_train = train_data[target_column]
    X_train = train_data.drop(columns=[target_column, "id"])
    X_test = test_data.drop(columns=["id"])

    # Remover outliers do target
    upper_limit = np.percentile(y_train, 99)
    y_train = y_train[y_train <= upper_limit]
    X_train = X_train.loc[y_train.index]

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

    # Reduzir categorias raras nas colunas categóricas
    for col in categorical_columns:
        # Calcular proporções das categorias
        freqs = X_train[col].value_counts(normalize=True)
        rare_categories = freqs[freqs < rare_threshold].index
        # Substituir categorias raras por "Other"
        X_train[col] = X_train[col].replace(rare_categories, "Other")
        X_test[col] = X_test[col].replace(rare_categories, "Other")
        if X_val is not None:
            X_val[col] = X_val[col].replace(rare_categories, "Other")

    # Converter todas as colunas categóricas para string
    for col in categorical_columns:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        if X_val is not None:
            X_val[col] = X_val[col].astype(str)

    # Selecionar features numéricas relevantes (baseado em correlação)
    corr_threshold = 0.05
    mi_scores = mutual_info_regression(X_train[numerical_columns], y_train)
    relevant_num_features = [
        col
        for col, score in zip(numerical_columns, mi_scores)
        if score > corr_threshold
    ]

    # Atualizar as colunas numéricas relevantes
    numerical_columns = relevant_num_features

    # Configurar o ColumnTransformer
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )

    # Aplicar as transformações
    X_train = transformer.fit_transform(X_train)
    if X_val is not None:
        X_val = transformer.transform(X_val)
    X_test = transformer.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test
