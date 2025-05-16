'''
    Módulo para la generación de visualizaciones en análisis de equidad.
    
    Este módulo proporciona funciones para crear visualizaciones informativas
    y accesibles que facilitan el análisis de datos, con enfoque en la evaluación
    de equidad en modelos de aprendizaje automático.
'''

import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Configuraciones para visualización
plt.style.use('seaborn-v0_8-whitegrid')   

def visualizar_distribuciones(dataframe: pd.DataFrame, variables, filas=2, columnas=2, figsize=(12, 10)):
    """
    Visualiza las distribuciones de variables numéricas clave en un conjunto de datos.

    Parameters:
        dataframe (pd.DataFrame): Conjunto de datos que contiene las variables.
        variables (list): Lista de nombres de las columnas a visualizar.
        filas (int): Número de filas en la cuadrícula de subgráficos.
        columnas (int): Número de columnas en la cuadrícula de subgráficos.
        figsize (tuple): Tamaño de la figura.
    """
    fig, axes = plt.subplots(filas, columnas, figsize=figsize)
    axes = axes.flatten()

    for i, var in enumerate(variables):
        sns.histplot(dataframe[var], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribución de {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Frecuencia')

    plt.tight_layout()
    plt.show()


def graficar_columnas(dataframe: pd.DataFrame, 
                      columns: list[str], 
                      target_label: str, 
                      plot_type: str, 
                      palette: str = 'colorblind'):
    """
    Genera gráficos para las columnas especificadas en relación con la variable objetivo.

    Parameters:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        columns (list): Lista de nombres de columnas a visualizar.
        target_label (str): Nombre de la columna objetivo para el hue o eje x.
        plot_type (str): Tipo de gráfico a generar ('countplot' o 'boxplot').
        palette (str): Paleta de colores a utilizar. Por defecto 'colorblind' 
                      para accesibilidad.
    
    Raises:
        ValueError: Si el tipo de gráfico no es 'countplot' o 'boxplot'.
    """
    # Verificar que el tipo de gráfico sea válido
    if plot_type not in ['countplot', 'boxplot']:
        raise ValueError("El tipo de gráfico debe ser 'countplot' o 'boxplot'.")
    
    n = len(columns)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    for i, col in enumerate(columns):
        r, c = divmod(i, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        
        if plot_type == 'countplot':
            sns.countplot(data=dataframe, x=col, hue=target_label, ax=ax, palette=palette)
            ax.set_xlabel('')
            ax.set_ylabel('Frecuencia')
            if len(dataframe[col].unique()) > 3:
                ax.tick_params(axis='x', rotation=45)
            ax.legend(title=target_label, loc='best')
        elif plot_type == 'boxplot':
            sns.boxplot(data=dataframe, x=target_label, y=col, ax=ax, palette=palette, hue=target_label)
            ax.set_ylabel(col)
            ax.set_xlabel('')
        
        ax.set_title(f"{col}")
        

    # Si sobran ejes, los desactivo
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar para el título principal
    plt.show()
    
    
def graficar_matriz_de_confusion_por_genero(dataframes: list[pd.DataFrame], 
                                           labels: list[str], 
                                           genre_column: str, 
                                           y_test_column: str, 
                                           y_pred_column: str,
                                           cmap: str = 'Blues'):
    """
    Genera matrices de confusión para diferentes subconjuntos de datos segregados por género.

    Parameters:
        dataframes (list): Lista de DataFrames, cada uno representando un subconjunto de datos.
        labels (list): Lista de etiquetas para las clases (e.g., ['Bajo riesgo', 'Alto riesgo']).
        genre_column (str): Nombre de la columna que contiene el género.
        y_test_column (str): Nombre de la columna con los valores reales.
        y_pred_column (str): Nombre de la columna con las predicciones del modelo.
        cmap (str): Mapa de colores para las matrices. Por defecto 'Blues'.
    """
    if not dataframes:
        raise ValueError("La lista de DataFrames no puede estar vacía")
    
    fig, axes = plt.subplots(1, len(dataframes), figsize=(15, 5))
    fig.suptitle('Matrices de confusión por género', fontsize=16)
    
    # Manejar el caso de un solo dataframe
    if len(dataframes) == 1:
        axes = [axes]
    
    for ax, df in zip(axes, dataframes):
        if len(df) == 0:
            ax.text(0.5, 0.5, f"No hay datos para este grupo", 
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            continue
            
        cm = confusion_matrix(df[y_test_column], df[y_pred_column], labels=[1, 0])
        
        # Calcular métricas para el título
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, 
                   xticklabels=labels, yticklabels=labels)
        
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor real')
        gender = df[genre_column].iloc[0] if not df.empty else "Desconocido"
        ax.set_title(f'{gender.capitalize()} (Acc: {accuracy:.2f}, Prec: {precision:.2f})')
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels, rotation=0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para el título principal
    plt.show()