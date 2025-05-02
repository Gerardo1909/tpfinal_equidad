'''
    Este módulo contiene funciones para la generación de gráficos.
'''

import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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