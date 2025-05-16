'''
    Módulo para la evaluación de fairness en grupos.
    
    Este módulo proporciona funciones para la evaluación de group fairness.
'''

import pandas as pd


def calcular_metricas_group_fairness(df: pd.DataFrame, categorias: str) -> pd.DataFrame:
    resultados = []

    for grupo, categoria in df.groupby(categorias):
        # Verdaderos positivos y falsos positivos
        tp = ((categoria['y_test'] == 1) & (categoria['y_pred_test'] == 1)).sum()
        fp = ((categoria['y_test'] == 0) & (categoria['y_pred_test'] == 1)).sum()
        
        total = len(categoria)
        positivos_reales = (categoria['y_test'] == 1).sum()
        negativos_reales = (categoria['y_test'] == 0).sum()
        predichos_positivos = (categoria['y_pred_test'] == 1).sum()

        # Métricas
        pprev = predichos_positivos / total if total else 0
        tpr = tp / positivos_reales if positivos_reales else 0
        fpr = fp / negativos_reales if negativos_reales else 0
        precision = tp / (tp + fp) if (tp + fp) else 0

        resultados.append({
            'attribute_value': grupo,
            'pprev': pprev,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision
        })

    return pd.DataFrame(resultados)