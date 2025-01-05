import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(train_data, test_data, target_column):
    """
    Pré-processa os dados para Machine Learning.
    
    - Remove valores ausentes.
    - Separa a variável target do resto dos dados.
    - Normaliza os dados (StandardScaler).
    """
    # Remover valores ausentes
    train_data = train_data.dropna()
    test_data = test_data.fillna(0)  # Preencher valores ausentes com 0 no teste
    
    # Separar target
    y_train = train_data[target_column]
    X_train = train_data.drop(columns=[target_column])
    X_test = test_data
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test
