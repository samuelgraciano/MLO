
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def split_features_target(data, target_column='Divorce'):
    """
    Separa características y variable objetivo.
    
    Args:
        data: DataFrame de pandas
        target_column: Nombre de la columna objetivo
        
    Returns:
        X: DataFrame con características
        y: Series con variable objetivo
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y


def create_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Divide datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Características
        y: Variable objetivo
        test_size: Proporción del conjunto de prueba
        random_state: Semilla aleatoria
        stratify: Si True, mantiene la proporción de clases
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=stratify_param
    )
    
    print(f"✅ División completada:")
    print(f"   - Entrenamiento: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   - Prueba: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   - Distribución en entrenamiento: {y_train.value_counts().to_dict()}")
    print(f"   - Distribución en prueba: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, method='standard'):
    """
    Escala las características usando StandardScaler o MinMaxScaler.
    
    Args:
        X_train: Características de entrenamiento
        X_test: Características de prueba
        method: 'standard' o 'minmax'
        
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method debe ser 'standard' o 'minmax'")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de vuelta a DataFrame si la entrada era DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(X_train_scaled, 
                                       columns=X_train.columns, 
                                       index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, 
                                      columns=X_test.columns, 
                                      index=X_test.index)
    
    print(f"✅ Escalado completado usando {method}")
    
    return X_train_scaled, X_test_scaled, scaler


def get_high_correlation_features(data, target='Divorce', threshold=0.3):
    """
    Identifica características con alta correlación con la variable objetivo.
    
    Args:
        data: DataFrame de pandas
        target: Nombre de la variable objetivo
        threshold: Umbral de correlación (valor absoluto)
        
    Returns:
        DataFrame con características y sus correlaciones
    """
    # Calcular correlaciones con el objetivo
    correlations = data.corr()[target].drop(target)
    
    # Filtrar por umbral
    high_corr = correlations[abs(correlations) >= threshold].sort_values(
        key=abs, ascending=False
    )
    
    # Crear DataFrame de resultados
    result = pd.DataFrame({
        'Feature': high_corr.index,
        'Correlation': high_corr.values,
        'Abs_Correlation': abs(high_corr.values)
    })
    
    print(f"✅ Encontradas {len(result)} características con |correlación| >= {threshold}")
    
    return result


def remove_highly_correlated_features(X, threshold=0.95):
    """
    Remueve características altamente correlacionadas entre sí.
    
    Args:
        X: DataFrame con características
        threshold: Umbral de correlación para remover
        
    Returns:
        DataFrame con características filtradas, lista de características removidas
    """
    # Calcular matriz de correlación
    corr_matrix = X.corr().abs()
    
    # Obtener triángulo superior de la matriz
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Encontrar características con correlación > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Remover características
    X_filtered = X.drop(columns=to_drop)
    
    print(f"✅ Removidas {len(to_drop)} características con correlación > {threshold}")
    if to_drop:
        print(f"   Características removidas: {to_drop}")
    
    return X_filtered, to_drop


def create_feature_groups(feature_names):
    """
    Agrupa características por categorías temáticas basadas en el cuestionario.
    
    Args:
        feature_names: Lista de nombres de características
        
    Returns:
        Diccionario con grupos de características
    """
    groups = {
        'Resolución de Conflictos': [f'Atr{i}' for i in range(1, 5)],
        'Tiempo de Calidad': [f'Atr{i}' for i in range(5, 10)],
        'Objetivos Compartidos': [f'Atr{i}' for i in range(10, 21)],
        'Conocimiento Mutuo': [f'Atr{i}' for i in range(21, 31)],
        'Comunicación Agresiva': [f'Atr{i}' for i in range(31, 42)],
        'Evitación': [f'Atr{i}' for i in range(42, 47)],
        'Actitud Defensiva': [f'Atr{i}' for i in range(47, 55)]
    }
    
    # Filtrar solo las características que existen en feature_names
    filtered_groups = {}
    for group_name, features in groups.items():
        existing_features = [f for f in features if f in feature_names]
        if existing_features:
            filtered_groups[group_name] = existing_features
    
    return filtered_groups


def calculate_group_statistics(data, groups):
    """
    Calcula estadísticas por grupo de características.
    
    Args:
        data: DataFrame de pandas
        groups: Diccionario con grupos de características
        
    Returns:
        DataFrame con estadísticas por grupo
    """
    stats = []
    
    for group_name, features in groups.items():
        group_data = data[features]
        stats.append({
            'Grupo': group_name,
            'N_Features': len(features),
            'Media': group_data.mean().mean(),
            'Std': group_data.std().mean(),
            'Min': group_data.min().min(),
            'Max': group_data.max().max()
        })
    
    return pd.DataFrame(stats)


def handle_outliers(data, method='iqr', threshold=1.5):
    """
    Identifica y maneja valores atípicos.
    
    Args:
        data: DataFrame de pandas
        method: Método para detectar outliers ('iqr' o 'zscore')
        threshold: Umbral para considerar outliers
        
    Returns:
        DataFrame con outliers identificados, máscara de outliers
    """
    outliers_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    
    if method == 'iqr':
        for col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers_mask[col] = (data[col] < lower_bound) | (data[col] > upper_bound)
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(data))
        outliers_mask = z_scores > threshold
    
    n_outliers = outliers_mask.sum().sum()
    print(f"✅ Detectados {n_outliers} valores atípicos usando método {method}")
    
    return data, outliers_mask


def encode_likert_to_binary(data, threshold=2):
    """
    Convierte escala Likert a binaria (bajo/alto).
    
    Args:
        data: DataFrame de pandas
        threshold: Umbral para la conversión (valores > threshold = 1)
        
    Returns:
        DataFrame con valores binarios
    """
    binary_data = (data > threshold).astype(int)
    
    print(f"✅ Convertido a binario usando umbral {threshold}")
    print(f"   0 = Respuestas 0-{threshold} (Bajo)")
    print(f"   1 = Respuestas {threshold+1}-4 (Alto)")
    
    return binary_data
