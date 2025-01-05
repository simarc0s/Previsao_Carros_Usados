from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor


def train_models(X_train, y_train):
    models = {}

    # Regressão Linear
    linear_model = LinearRegression()
    scores = cross_val_score(
        linear_model, X_train, y_train, cv=2, scoring="neg_mean_squared_error"
    )
    models["Linear Regression"] = {
        "model": linear_model.fit(X_train, y_train),
        "cv_rmse": (-scores.mean()) ** 0.5,
    }

    # Random Forest
    rf_param_grid = {
        "n_estimators": [10],
        "max_depth": [None],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    }
    rf = RandomForestRegressor(random_state=42)
    rf_grid_search = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        cv=2,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    rf_grid_search.fit(X_train, y_train)
    models["Random Forest"] = {
        "model": rf_grid_search.best_estimator_,
        "best_params": rf_grid_search.best_params_,
        "cv_rmse": (-rf_grid_search.best_score_) ** 0.5,
    }

    # K-Nearest Neighbors
    knn_param_grid = {"n_neighbors": [5]}
    knn = KNeighborsRegressor()
    knn_grid_search = GridSearchCV(
        estimator=knn,
        param_grid=knn_param_grid,
        cv=2,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    knn_grid_search.fit(X_train, y_train)
    models["K-Nearest Neighbors"] = {
        "model": knn_grid_search.best_estimator_,
        "best_params": knn_grid_search.best_params_,
        "cv_rmse": (-knn_grid_search.best_score_) ** 0.5,
    }

    return models
