"""
Análisis Estadístico Descriptivo Completo
Divorce Predictors Dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro
import warnings
warnings.filterwarnings('ignore')


def calculate_descriptive_stats(series):
    """
    Calcula estadísticas descriptivas completas para una serie.
    
    Returns:
        dict con todas las medidas estadísticas
    """
    return {
        # Tendencia central
        'media': series.mean(),
        'mediana': series.median(),
        'moda': series.mode()[0] if len(series.mode()) > 0 else np.nan,
        
        # Dispersión
        'desv_std': series.std(),
        'varianza': series.var(),
        'rango': series.max() - series.min(),
        'IQR': series.quantile(0.75) - series.quantile(0.25),
        'min': series.min(),
        'max': series.max(),
        
        # Forma
        'asimetria': series.skew(),
        'curtosis': series.kurtosis(),
        
        # Percentiles
        'p25': series.quantile(0.25),
        'p50': series.quantile(0.50),
        'p75': series.quantile(0.75),
        'p90': series.quantile(0.90),
        'p95': series.quantile(0.95),
        
        # Coeficiente de variación
        'CV': (series.std() / series.mean() * 100) if series.mean() != 0 else np.nan,
        
        # Conteo
        'n': len(series),
        'n_missing': series.isna().sum()
    }


def frequency_distribution(series):
    """
    Calcula distribución de frecuencias para datos ordinales.
    """
    freq = series.value_counts().sort_index()
    prop = series.value_counts(normalize=True).sort_index()
    
    return pd.DataFrame({
        'frecuencia': freq,
        'proporcion': prop,
        'porcentaje': prop * 100
    })


def cliffs_delta(group1, group2):
    """
    Calcula Cliff's Delta como medida de tamaño del efecto para datos ordinales.
    
    Interpretación:
    |d| < 0.147: negligible
    |d| < 0.330: small
    |d| < 0.474: medium
    |d| >= 0.474: large
    """
    n1, n2 = len(group1), len(group2)
    
    # Contar pares donde group1 > group2 y group1 < group2
    dominance = 0
    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
    
    delta = dominance / (n1 * n2)
    
    # Interpretación
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = 'negligible'
    elif abs_delta < 0.330:
        interpretation = 'small'
    elif abs_delta < 0.474:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return delta, interpretation


def compare_groups(df, feature, group_col='Divorce'):
    """
    Compara dos grupos para una feature específica.
    """
    group0 = df[df[group_col] == 0][feature].dropna()
    group1 = df[df[group_col] == 1][feature].dropna()
    
    # Estadísticas descriptivas por grupo
    stats0 = calculate_descriptive_stats(group0)
    stats1 = calculate_descriptive_stats(group1)
    
    # Diferencia de medias
    mean_diff = stats1['media'] - stats0['media']
    
    # Mann-Whitney U test
    statistic, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
    
    # Cliff's Delta
    delta, effect_interpretation = cliffs_delta(group0, group1)
    
    # Prueba de normalidad (Shapiro-Wilk)
    _, p_normal_0 = shapiro(group0) if len(group0) >= 3 else (np.nan, np.nan)
    _, p_normal_1 = shapiro(group1) if len(group1) >= 3 else (np.nan, np.nan)
    
    return {
        'feature': feature,
        'n_no_divorce': stats0['n'],
        'n_divorce': stats1['n'],
        'mean_no_divorce': stats0['media'],
        'mean_divorce': stats1['media'],
        'median_no_divorce': stats0['mediana'],
        'median_divorce': stats1['mediana'],
        'std_no_divorce': stats0['desv_std'],
        'std_divorce': stats1['desv_std'],
        'mean_diff': mean_diff,
        'mann_whitney_U': statistic,
        'p_value': p_value,
        'cliffs_delta': delta,
        'effect_size': effect_interpretation,
        'shapiro_p_no_divorce': p_normal_0,
        'shapiro_p_divorce': p_normal_1
    }


def bonferroni_correction(p_values, alpha=0.05):
    """
    Aplica corrección de Bonferroni.
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    significant = p_values < adjusted_alpha
    
    return adjusted_alpha, significant


def create_summary_table(df, features):
    """
    Crea tabla resumen con todas las estadísticas descriptivas.
    """
    summary_data = []
    
    for feature in features:
        stats_dict = calculate_descriptive_stats(df[feature])
        stats_dict['feature'] = feature
        summary_data.append(stats_dict)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Reordenar columnas
    cols = ['feature', 'n', 'n_missing', 'media', 'mediana', 'moda', 
            'desv_std', 'varianza', 'CV', 'min', 'max', 'rango', 'IQR',
            'p25', 'p50', 'p75', 'p90', 'p95', 'asimetria', 'curtosis']
    
    return summary_df[cols]


def create_comparison_table(df, features, group_col='Divorce'):
    """
    Crea tabla de comparación entre grupos con pruebas estadísticas.
    """
    comparison_data = []
    
    for feature in features:
        comp_dict = compare_groups(df, feature, group_col)
        comparison_data.append(comp_dict)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Aplicar corrección de Bonferroni
    adjusted_alpha, significant = bonferroni_correction(comparison_df['p_value'])
    comparison_df['bonferroni_significant'] = significant
    comparison_df['adjusted_alpha'] = adjusted_alpha
    
    return comparison_df


def interpret_distribution(skewness, kurtosis):
    """
    Interpreta la forma de la distribución.
    """
    # Asimetría
    if abs(skewness) < 0.5:
        skew_interp = 'simétrica'
    elif skewness > 0:
        if skewness < 1:
            skew_interp = 'asimetría positiva moderada'
        else:
            skew_interp = 'asimetría positiva fuerte'
    else:
        if skewness > -1:
            skew_interp = 'asimetría negativa moderada'
        else:
            skew_interp = 'asimetría negativa fuerte'
    
    # Curtosis
    if abs(kurtosis) < 0.5:
        kurt_interp = 'mesocúrtica (normal)'
    elif kurtosis > 0:
        kurt_interp = 'leptocúrtica (colas pesadas)'
    else:
        kurt_interp = 'platicúrtica (colas ligeras)'
    
    return f"{skew_interp}, {kurt_interp}"
