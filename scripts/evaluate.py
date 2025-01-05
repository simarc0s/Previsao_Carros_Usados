from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_train, y_train):
    """
    Avalia o modelo usando métricas de erro quadrático médio (MSE) e R².
    """
    # Fazer previsões no conjunto de treino
    predictions = model.predict(X_train)
    
    # Calcular métricas
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, predictions)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
