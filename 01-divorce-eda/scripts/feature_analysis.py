"""
Análisis de Features Predictivas
Divorce Predictors Dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, pointbiserialr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def cliffs_delta(group1, group2):
    """
    Calcula Cliff's Delta como medida de tamaño del efecto.
    
    Interpretación:
    |d| < 0.147: negligible
    |d| < 0.330: small
    |d| < 0.474: medium
    |d| >= 0.474: large
    """
    n1, n2 = len(group1), len(group2)
    
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


def rank_biserial(u_statistic, n1, n2):
    """
    Calcula correlación rank-biserial desde Mann-Whitney U.
    
    r = 1 - (2U)/(n1*n2)
    """
    r = 1 - (2 * u_statistic) / (n1 * n2)
    return r


def comprehensive_feature_test(df, feature, target='Divorce', alpha=0.05):
    """
    Realiza análisis completo de una feature.
    
    Returns:
    --------
    dict con todas las estadísticas
    """
    # Separar grupos
    group0 = df[df[target] == 0][feature].dropna()
    group1 = df[df[target] == 1][feature].dropna()
    
    n0, n1 = len(group0), len(group1)
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
    
    # Cliff's Delta
    delta, effect_interpretation = cliffs_delta(group0, group1)
    
    # Rank-biserial
    r_rb = rank_biserial(u_stat, n0, n1)
    
    # Point-biserial correlation
    # Necesitamos combinar los datos
    combined_feature = pd.concat([group0, group1])
    combined_target = pd.concat([
        pd.Series([0] * n0),
        pd.Series([1] * n1)
    ])
    r_pb, p_pb = pointbiserialr(combined_target, combined_feature)
    
    # Estadísticas descriptivas
    mean0, mean1 = group0.mean(), group1.mean()
    median0, median1 = group0.median(), group1.median()
    std0, std1 = group0.std(), group1.std()
    
    # Diferencias
    mean_diff = mean1 - mean0
    median_diff = median1 - median0
    pct_change = (mean_diff / mean0 * 100) if mean0 != 0 else np.nan
    
    # Significancia con Bonferroni
    bonferroni_alpha = alpha / 54  # 54 features
    is_significant = p_value < bonferroni_alpha
    
    # Marcador de significancia
    if p_value < 0.001:
        sig_marker = '***'
    elif p_value < 0.01:
        sig_marker = '**'
    elif p_value < 0.05:
        sig_marker = '*'
    else:
        sig_marker = ''
    
    return {
        'feature': feature,
        'n_no_divorce': n0,
        'n_divorce': n1,
        'mean_no_divorce': mean0,
        'mean_divorce': mean1,
        'median_no_divorce': median0,
        'median_divorce': median1,
        'std_no_divorce': std0,
        'std_divorce': std1,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'pct_change': pct_change,
        'mann_whitney_u': u_stat,
        'p_value': p_value,
        'cliffs_delta': delta,
        'effect_size': effect_interpretation,
        'rank_biserial': r_rb,
        'point_biserial': r_pb,
        'p_pointbiserial': p_pb,
        'bonferroni_alpha': bonferroni_alpha,
        'is_significant': is_significant,
        'sig_marker': sig_marker
    }


def analyze_all_features(df, features, target='Divorce'):
    """
    Analiza todas las features y retorna DataFrame con resultados.
    """
    results = []
    
    for feature in features:
        result = comprehensive_feature_test(df, feature, target)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Ordenar por tamaño del efecto (valor absoluto)
    results_df['abs_cliffs_delta'] = results_df['cliffs_delta'].abs()
    results_df = results_df.sort_values('abs_cliffs_delta', ascending=False)
    
    return results_df


def ml_feature_importance(df, features, target='Divorce', n_cv=5):
    """
    Calcula importancia de features usando ML.
    
    Returns:
    --------
    dict con importancias de RF y LR
    """
    X = df[features].values
    y = df[target].values
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'feature': features,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # Cross-validation score
    rf_cv_scores = cross_val_score(rf, X, y, cv=n_cv, scoring='roc_auc')
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)
    lr_importance = pd.DataFrame({
        'feature': features,
        'lr_coefficient': np.abs(lr.coef_[0])
    }).sort_values('lr_coefficient', ascending=False)
    
    # Cross-validation score
    lr_cv_scores = cross_val_score(lr, X, y, cv=n_cv, scoring='roc_auc')
    
    return {
        'rf_importance': rf_importance,
        'lr_importance': lr_importance,
        'rf_cv_mean': rf_cv_scores.mean(),
        'rf_cv_std': rf_cv_scores.std(),
        'lr_cv_mean': lr_cv_scores.mean(),
        'lr_cv_std': lr_cv_scores.std()
    }


def calculate_confidence_intervals(group, confidence=0.95):
    """
    Calcula intervalos de confianza para la media.
    """
    n = len(group)
    mean = group.mean()
    se = stats.sem(group)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - ci, mean + ci
