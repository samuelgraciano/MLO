
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_distribution(data, column, title=None, figsize=(10, 6)):
    """
    Grafica la distribución de una variable.
    
    Args:
        data: DataFrame de pandas
        column: Nombre de la columna a graficar
        title: Título del gráfico (opcional)
        figsize: Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histograma
    axes[0].hist(data[column], bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Histograma: {column}' if not title else title)
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot
    axes[1].boxplot(data[column])
    axes[1].set_title(f'Boxplot: {column}')
    axes[1].set_ylabel(column)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data, figsize=(16, 14), method='pearson'):
    """
    Grafica una matriz de correlación.
    
    Args:
        data: DataFrame de pandas
        figsize: Tamaño de la figura
        method: Método de correlación ('pearson', 'spearman', 'kendall')
    """
    # Calcular correlaciones
    corr = data.corr(method=method)
    
    # Crear figura
    plt.figure(figsize=figsize)
    
    # Crear heatmap
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title(f'Matriz de Correlación ({method.capitalize()})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    return corr


def plot_feature_importance(importances, feature_names, top_n=20, figsize=(10, 8)):
    """
    Grafica la importancia de características.
    
    Args:
        importances: Array con valores de importancia
        feature_names: Lista con nombres de características
        top_n: Número de características principales a mostrar
        figsize: Tamaño de la figura
    """
    # Crear DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Crear gráfico
    plt.figure(figsize=figsize)
    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Características Más Importantes', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_class_comparison(data, feature, target='Divorce', figsize=(12, 5)):
    """
    Compara la distribución de una característica entre clases.
    
    Args:
        data: DataFrame de pandas
        feature: Nombre de la característica a comparar
        target: Nombre de la variable objetivo
        figsize: Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Separar por clase
    class_0 = data[data[target] == 0][feature]
    class_1 = data[data[target] == 1][feature]
    
    # Histogramas superpuestos
    axes[0].hist(class_0, bins=15, alpha=0.6, label='Casado', color='#2ecc71')
    axes[0].hist(class_1, bins=15, alpha=0.6, label='Divorciado', color='#e74c3c')
    axes[0].set_title(f'Distribución: {feature}')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplots lado a lado
    data.boxplot(column=feature, by=target, ax=axes[1])
    axes[1].set_title(f'Comparación por Clase: {feature}')
    axes[1].set_xlabel('Divorce (0=Casado, 1=Divorciado)')
    axes[1].set_ylabel(feature)
    plt.suptitle('')  # Remover título automático
    
    plt.tight_layout()
    plt.show()


def plot_pca_variance(explained_variance_ratio, n_components=20, figsize=(12, 5)):
    """
    Grafica la varianza explicada por componentes principales.
    
    Args:
        explained_variance_ratio: Array con ratios de varianza explicada
        n_components: Número de componentes a mostrar
        figsize: Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Varianza individual
    axes[0].bar(range(1, n_components+1), 
                explained_variance_ratio[:n_components])
    axes[0].set_title('Varianza Explicada por Componente')
    axes[0].set_xlabel('Componente Principal')
    axes[0].set_ylabel('Ratio de Varianza Explicada')
    axes[0].grid(True, alpha=0.3)
    
    # Varianza acumulada
    cumsum = np.cumsum(explained_variance_ratio[:n_components])
    axes[1].plot(range(1, n_components+1), cumsum, marker='o')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
    axes[1].set_title('Varianza Explicada Acumulada')
    axes[1].set_xlabel('Número de Componentes')
    axes[1].set_ylabel('Varianza Acumulada')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=['Casado', 'Divorciado'], 
                         figsize=(8, 6), normalize=False):
    """
    Grafica una matriz de confusión.
    
    Args:
        cm: Matriz de confusión
        class_names: Nombres de las clases
        figsize: Tamaño de la figura
        normalize: Si True, normaliza los valores
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.tight_layout()
    plt.show()


def plot_likert_distribution(data, features=None, figsize=(15, 10)):
    """
    Grafica la distribución de respuestas en escala Likert.
    
    Args:
        data: DataFrame de pandas
        features: Lista de características a graficar (None = todas)
        figsize: Tamaño de la figura
    """
    if features is None:
        features = [col for col in data.columns if col != 'Divorce']
    
    # Calcular distribución de valores
    value_counts = {}
    for feature in features:
        value_counts[feature] = data[feature].value_counts(normalize=True).sort_index()
    
    # Crear DataFrame para el gráfico
    df_plot = pd.DataFrame(value_counts).T
    
    # Graficar
    plt.figure(figsize=figsize)
    df_plot.plot(kind='bar', stacked=True, 
                 color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db'])
    
    plt.title('Distribución de Respuestas en Escala Likert', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Características')
    plt.ylabel('Proporción')
    plt.legend(title='Escala', labels=['0 (Nunca)', '1 (Rara vez)', 
                                        '2 (A veces)', '3 (Frecuentemente)', 
                                        '4 (Siempre)'])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
