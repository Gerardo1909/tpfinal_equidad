'''
    Este módulo contiene funciones para la generación de gráficos.
'''

import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def graficar_columnas(dataframe: pd.DataFrame, columns: list[str], target_label: str, plot_type:str):
    
    """
    Genera gráficos para las columnas especificadas en relación con la variable objetivo.

    Parameters:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        columns (list): Lista de nombres de columnas a visualizar.
        target_label (str): Nombre de la columna objetivo para el hue o eje x.
        plot_type (str): Tipo de gráfico a generar ('countplot' o 'boxplot').
    """
    
    # Verifica que el tipo de gráfico sea válido
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
            sns.countplot(data=dataframe, x=col, hue=target_label, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
        elif plot_type == 'boxplot':
            sns.boxplot(data=dataframe, x=target_label, y=col, ax=ax)
            ax.set_ylabel(col)
            ax.set_xlabel('')
        
        ax.set_title(f"{col}")
        ax.tick_params(axis='x', rotation=45)

    # Si sobran ejes, los desactivo
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
def graficar_matriz_de_confusion_por_genero(dataframes: list[pd.DataFrame], labels: list[str], genre_column: str, y_test_column: str, y_pred_column:str):
    """
    Genera matrices de confusión para diferentes subconjuntos de datos.

    Parameters:
        dataframes (list): Lista de DataFrames, cada uno representando un subconjunto de datos.
        labels (list): Lista de etiquetas para las clases (e.g., ['Bajo riesgo (0)', 'Alto riesgo (1)']).
        genre_column (str): Nombre de la columna que contiene el género.
        y_test_column (str): Nombre de la columna con los valores reales.
        y_pred_column (str): Nombre de la columna con las predicciones del modelo.
    """
    fig, axes = plt.subplots(1, len(dataframes), figsize=(15, 5))
    
    for ax, df in zip(axes, dataframes):
        cm = confusion_matrix(df[y_test_column], df[y_pred_column])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor real')
        ax.set_title(f'Matriz de confusión - {df[genre_column].iloc[0]}')
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels, rotation=0)
    
    plt.tight_layout()
    plt.show()