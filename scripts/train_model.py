from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def train_models(X_train, y_train):
    """
    Treina diferentes modelos e retorna um dicionário com os modelos treinados.
    """
    models = {}

    # Regressão Linear
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    models["Linear Regression"] = linear_model

    # Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    models["Random Forest"] = rf_model

    # K-Nearest Neighbors
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    models["K-Nearest Neighbors"] = knn_model

    return models
