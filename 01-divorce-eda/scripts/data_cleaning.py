"""
Utilidades para limpieza de datos del proyecto Divorce Predictors.

Este módulo contiene funciones reutilizables para:
- Detección y manejo de valores faltantes
- Identificación de duplicados
- Detección de outliers
- Validación de rangos de valores
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, List


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta y resume valores faltantes en el DataFrame.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        DataFrame con conteo y porcentaje de valores faltantes por columna
    """
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Valores_Faltantes': missing_counts,
        'Porcentaje': missing_pct
    }).sort_values('Valores_Faltantes', ascending=False)
    
    return missing_df[missing_df['Valores_Faltantes'] > 0]


def detect_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Tuple[pd.DataFrame, int]:
    """
    Detecta filas duplicadas en el DataFrame.
    
    Args:
        df: DataFrame a analizar
        subset: Lista de columnas a considerar (None = todas)
        
    Returns:
        Tupla con (DataFrame de duplicados, número de duplicados)
    """
    duplicates = df.duplicated(subset=subset, keep=False)
    n_duplicates = df.duplicated(subset=subset).sum()
    
    return df[duplicates], n_duplicates


def detect_outliers_iqr(df: pd.DataFrame, column: str, 
                        multiplier: float = 1.5) -> Tuple[pd.DataFrame, float, float]:
    """
    Detecta outliers usando el método IQR.
    
    Args:
        df: DataFrame a analizar
        column: Nombre de la columna
        multiplier: Multiplicador para IQR (default: 1.5)
        
    Returns:
        Tupla con (DataFrame de outliers, límite inferior, límite superior)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(df: pd.DataFrame, column: str, 
                           threshold: float = 3.0) -> pd.DataFrame:
    """
    Detecta outliers usando el método Z-score.
    
    Args:
        df: DataFrame a analizar
        column: Nombre de la columna
        threshold: Umbral de Z-score (default: 3.0)
        
    Returns:
        DataFrame con outliers detectados
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outlier_indices = np.where(z_scores > threshold)[0]
    
    return df.iloc[outlier_indices]


def validate_value_range(df: pd.DataFrame, columns: List[str], 
                        min_val: float, max_val: float) -> Dict[str, Dict]:
    """
    Valida que los valores estén dentro del rango esperado.
    
    Args:
        df: DataFrame a validar
        columns: Lista de columnas a verificar
        min_val: Valor mínimo esperado
        max_val: Valor máximo esperado
        
    Returns:
        Diccionario con columnas que tienen valores fuera de rango
    """
    invalid_values = {}
    
    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min < min_val or col_max > max_val:
            invalid_values[col] = {
                'min': col_min,
                'max': col_max,
                'expected_range': (min_val, max_val),
                'n_invalid': len(df[(df[col] < min_val) | (df[col] > max_val)])
            }
    
    return invalid_values


def remove_duplicates(df: pd.DataFrame, subset: List[str] = None, 
                     keep: str = 'first') -> pd.DataFrame:
    """
    Elimina filas duplicadas del DataFrame.
    
    Args:
        df: DataFrame a limpiar
        subset: Lista de columnas a considerar (None = todas)
        keep: 'first', 'last', o False
        
    Returns:
        DataFrame sin duplicados
    """
    return df.drop_duplicates(subset=subset, keep=keep)


def impute_missing_values(df: pd.DataFrame, strategy: str = 'median', 
                         columns: List[str] = None) -> pd.DataFrame:
    """
    Imputa valores faltantes usando la estrategia especificada.
    
    Args:
        df: DataFrame a limpiar
        strategy: 'mean', 'median', 'mode', o valor específico
        columns: Lista de columnas a imputar (None = todas con missing)
        
    Returns:
        DataFrame con valores imputados
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns[df_clean.isnull().any()].tolist()
    
    for col in columns:
        if df_clean[col].isnull().sum() > 0:
            if strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif strategy == 'median':
                fill_value = df_clean[col].median()
            elif strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            else:
                fill_value = strategy
            
            df_clean[col].fillna(fill_value, inplace=True)
    
    return df_clean


def generate_cleaning_report(df_original: pd.DataFrame, df_clean: pd.DataFrame,
                            decisions: List[str]) -> str:
    """
    Genera un reporte de limpieza de datos.
    
    Args:
        df_original: DataFrame original
        df_clean: DataFrame limpio
        decisions: Lista de decisiones tomadas
        
    Returns:
        String con el reporte formateado
    """
    report = []
    report.append("=" * 60)
    report.append("REPORTE DE LIMPIEZA DE DATOS")
    report.append("=" * 60)
    report.append("")
    
    report.append("ESTADÍSTICAS INICIALES:")
    report.append(f"  Filas: {len(df_original)}")
    report.append(f"  Columnas: {len(df_original.columns)}")
    report.append(f"  Valores totales: {df_original.size}")
    report.append(f"  Valores faltantes: {df_original.isnull().sum().sum()}")
    report.append("")
    
    report.append("ESTADÍSTICAS FINALES:")
    report.append(f"  Filas: {len(df_clean)} (cambio: {len(df_clean) - len(df_original):+d})")
    report.append(f"  Columnas: {len(df_clean.columns)} (cambio: {len(df_clean.columns) - len(df_original.columns):+d})")
    report.append(f"  Valores totales: {df_clean.size}")
    report.append(f"  Valores faltantes: {df_clean.isnull().sum().sum()}")
    report.append(f"  Porcentaje conservado: {(df_clean.size/df_original.size)*100:.2f}%")
    report.append("")
    
    report.append("DECISIONES DE LIMPIEZA:")
    for decision in decisions:
        report.append(f"  • {decision}")
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def clean_divorce_data(df: pd.DataFrame, 
                      remove_dups: bool = True,
                      impute_strategy: str = 'median',
                      validate_ranges: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pipeline completo de limpieza para el dataset de Divorce Predictors.
    
    Args:
        df: DataFrame a limpiar
        remove_dups: Si eliminar duplicados
        impute_strategy: Estrategia de imputación
        validate_ranges: Si validar rangos de valores
        
    Returns:
        Tupla con (DataFrame limpio, lista de decisiones)
    """
    df_clean = df.copy()
    decisions = []
    
    # 1. Eliminar duplicados
    if remove_dups:
        n_before = len(df_clean)
        df_clean = remove_duplicates(df_clean)
        n_removed = n_before - len(df_clean)
        if n_removed > 0:
            decisions.append(f"Eliminados {n_removed} duplicados")
        else:
            decisions.append("No se encontraron duplicados")
    
    # 2. Validar rangos (features deben estar en 0-4)
    if validate_ranges:
        feature_cols = [col for col in df_clean.columns if col != 'Divorce']
        invalid = validate_value_range(df_clean, feature_cols, 0, 4)
        
        if invalid:
            decisions.append(f"Advertencia: {len(invalid)} columnas con valores fuera de rango [0-4]")
            for col, info in invalid.items():
                decisions.append(f"  - {col}: rango [{info['min']}, {info['max']}], {info['n_invalid']} valores inválidos")
        else:
            decisions.append("Todos los valores en rango válido [0-4]")
    
    # 3. Imputar valores faltantes
    missing_before = df_clean.isnull().sum().sum()
    if missing_before > 0:
        df_clean = impute_missing_values(df_clean, strategy=impute_strategy)
        decisions.append(f"Imputados {missing_before} valores faltantes usando {impute_strategy}")
    else:
        decisions.append("No se encontraron valores faltantes")
    
    # 4. Verificar variable objetivo
    if 'Divorce' in df_clean.columns:
        unique_vals = df_clean['Divorce'].nunique()
        if unique_vals == 2:
            decisions.append("Variable objetivo validada (binaria)")
        else:
            decisions.append(f"Advertencia: Variable objetivo tiene {unique_vals} valores únicos")
    
    return df_clean, decisions
