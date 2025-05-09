'''
    Módulo para modelos de aprendizaje automático.
    
    Este módulo proporciona funciones para la creación, entrenamiento y evaluación 
    de modelos de aprendizaje automático.
'''

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional, Union


def sigmoide(x: np.ndarray) -> np.ndarray:
    """
    Aplica la función sigmoide a un array de valores.
    
    Parameters:
        x (np.ndarray): Array de valores de entrada.
        
    Returns:
        np.ndarray: Valores transformados mediante la función sigmoide.
    """

    return 1 / (1 + np.exp(-x))


def ajustar_regresion_logistica(X: np.ndarray, 
                               y: np.ndarray) -> np.ndarray:
    """
    Ajusta un modelo de regresión logística mediante optimización numérica.
    
    Parameters:
        X (np.ndarray): Matriz de características de forma (n_samples, n_features).
        y (np.ndarray): Vector de etiquetas binarias de forma (n_samples,).
        max_iter (int): Número máximo de iteraciones para el optimizador.
        tol (float): Tolerancia para considerar convergencia.
        
    Returns:
        np.ndarray: Vector de coeficientes estimados de forma (n_features,).
        
    Raises:
        ValueError: Si X y y no tienen dimensiones compatibles o si y no es binaria.
    """
    # Verificación de los datos de entrada
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"El número de muestras en X ({X.shape[0]}) no coincide con y ({y.shape[0]})")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("La variable objetivo debe ser binaria (0 o 1)")
    
    def neg_log_likelihood(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula el negativo del log-likelihood para la regresión logística.
        
        Parameters:
            beta (np.ndarray): Vector de coeficientes.
            X (np.ndarray): Matriz de características.
            y (np.ndarray): Vector de etiquetas.
            
        Returns:
            float: Valor del negativo del log-likelihood.
        """
        nu = X @ beta
        vinculo = sigmoide(nu)
        epsilon = 1e-10  # Evitar logaritmo de cero
        log_likelihood = np.sum(y * np.log(vinculo + epsilon) + 
                               (1 - y) * np.log(1 - vinculo + epsilon))
        return -log_likelihood


    print("Entrenando modelo de regresión logística...")
    
    # Inicialización de coeficientes
    initial_beta = np.zeros(X.shape[1])
    
    # Ajuste con minimize
    resultado = minimize(
        neg_log_likelihood, 
        initial_beta, 
        args=(X, y), 
        method='BFGS'
        )
    
    # Coeficientes resultantes
    beta_hat = resultado.x
    
    return beta_hat


def predecir_regresion_logistica(X: np.ndarray, 
                                beta_hat: np.ndarray, 
                                umbral: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera predicciones utilizando un modelo de regresión logística.
    
    Parameters:
        X (np.ndarray): Matriz de características de forma (n_samples, n_features).
        beta_hat (np.ndarray): Vector de coeficientes estimados.
        umbral (float): Umbral de probabilidad para clasificación binaria.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tupla que contiene:
            - Vector de predicciones binarias (0 o 1)
            - Vector de probabilidades estimadas
            
    Raises:
        ValueError: Si las dimensiones de X y beta_hat no son compatibles.
    """
    if X.shape[1] != beta_hat.shape[0]:
        raise ValueError(f"Dimensiones incompatibles: X tiene {X.shape[1]} características, pero beta_hat tiene {beta_hat.shape[0]} coeficientes")
    
    # Calcular probabilidades
    y_pred_probs = sigmoide(X @ beta_hat)
    
    # Convertir a predicciones binarias
    y_pred = (y_pred_probs >= umbral).astype(int)
    
    print("Predicciones completadas.")
    
    return y_pred
