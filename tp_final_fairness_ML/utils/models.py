'''
    Este módulo contiene funciones para la generación de modelos de aprendizaje automático.
'''

import numpy as np
from scipy.optimize import minimize

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def ajustar_regresion_logistica(X, y): 

    def neg_log_likelihood(beta, X, y):
        nu = X @ beta
        vinculo = sigmoide(nu)
        return -np.sum(y * np.log(vinculo + 1e-10) + (1 - y) * np.log(1 - vinculo + 1e-10))
    
    # Inicializamos coeficientes
    initial_beta = np.zeros(X.shape[1])

    # Ajuste con minimize
    res = minimize(neg_log_likelihood, initial_beta, args=(X, y), method='BFGS')
    
    # Coeficientes resultantes
    beta_hat = res.x
    
    return beta_hat

def predecir_regresion_logistica(X, beta_hat):
    y_pred_probs = sigmoide(X @ beta_hat)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    return y_pred