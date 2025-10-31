"""
Utilidades de Visualización
Divorce Predictors Dataset
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')

# Colores personalizados
COLORS = {
    'no_divorce': '#3498db',  # Azul
    'divorce': '#e74c3c',      # Rojo
    'neutral': '#95a5a6'       # Gris
}

# Tamaños de fuente
FONT_SIZES = {
    'title': 14,
    'label': 12,
    'tick': 10
}


def setup_plot_style():
    """Configura el estilo global de los gráficos."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = FONT_SIZES['tick']
    plt.rcParams['axes.labelsize'] = FONT_SIZES['label']
    plt.rcParams['axes.titlesize'] = FONT_SIZES['title']
    plt.rcParams['xtick.labelsize'] = FONT_SIZES['tick']
    plt.rcParams['ytick.labelsize'] = FONT_SIZES['tick']


def plot_distribution(data, feature, ax=None, show_kde=True):
    """
    Crea histograma con KDE para una feature.
    
    Parameters:
    -----------
    data : DataFrame
    feature : str
    ax : matplotlib axis
    show_kde : bool
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(data=data, x=feature, kde=show_kde, ax=ax, 
                 color=COLORS['neutral'], bins=5)
    ax.set_xlabel('Valor', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Frecuencia', fontsize=FONT_SIZES['label'])
    ax.set_title(f'Distribución de {feature}', fontsize=FONT_SIZES['title'])
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_by_group(data, feature, target='Divorce', ax=None):
    """
    Crea violin plot dividido por grupo.
    
    Parameters:
    -----------
    data : DataFrame
    feature : str
    target : str
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.violinplot(data=data, x=target, y=feature, ax=ax,
                   palette=[COLORS['no_divorce'], COLORS['divorce']],
                   split=False)
    ax.set_xlabel('Estado de Divorcio', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Valor', fontsize=FONT_SIZES['label'])
    ax.set_title(f'{feature} por Grupo', fontsize=FONT_SIZES['title'])
    ax.set_xticklabels(['No Divorciado', 'Divorciado'])
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_boxplot(data, feature, ax=None):
    """
    Crea boxplot para una feature.
    
    Parameters:
    -----------
    data : DataFrame
    feature : str
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=data, y=feature, ax=ax, color=COLORS['neutral'])
    ax.set_ylabel('Valor', fontsize=FONT_SIZES['label'])
    ax.set_title(f'Boxplot de {feature}', fontsize=FONT_SIZES['title'])
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def save_plot(fig, filename, output_dir='../visualizations/univariate'):
    """
    Guarda el gráfico en el directorio especificado.
    
    Parameters:
    -----------
    fig : matplotlib figure
    filename : str
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"💾 Gráfico guardado: {filepath}")


def create_qq_plot(data, feature, ax=None):
    """
    Crea Q-Q plot para evaluar normalidad.
    
    Parameters:
    -----------
    data : DataFrame
    feature : str
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    stats.probplot(data[feature].dropna(), dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot: {feature}', fontsize=FONT_SIZES['title'])
    ax.grid(True, alpha=0.3)
    
    return ax


def interpret_distribution(skewness, kurtosis):
    """
    Interpreta la forma de la distribución basándose en asimetría y curtosis.
    
    Parameters:
    -----------
    skewness : float
    kurtosis : float
    
    Returns:
    --------
    str : Interpretación en español
    """
    # Asimetría
    if abs(skewness) < 0.5:
        skew_text = "simétrica"
    elif skewness > 0:
        if skewness < 1:
            skew_text = "asimetría positiva moderada (sesgo derecha)"
        else:
            skew_text = "asimetría positiva fuerte (sesgo derecha)"
    else:
        if skewness > -1:
            skew_text = "asimetría negativa moderada (sesgo izquierda)"
        else:
            skew_text = "asimetría negativa fuerte (sesgo izquierda)"
    
    # Curtosis
    if abs(kurtosis) < 0.5:
        kurt_text = "mesocúrtica (similar a normal)"
    elif kurtosis > 0:
        kurt_text = "leptocúrtica (colas pesadas)"
    else:
        kurt_text = "platicúrtica (colas ligeras)"
    
    return f"Distribución {skew_text}, {kurt_text}"
