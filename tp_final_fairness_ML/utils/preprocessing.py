'''
    Módulo para el preprocesamiento de datos del German Credit Dataset.
    
    Este módulo proporciona funciones para limpiar, transformar y preparar
    los datos del dataset German Credit para análisis de equidad y entrenamiento
    de modelos de machine learning.
'''

import pandas as pd


def mapear_german_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapea los códigos originales del German Credit Dataset a valores descriptivos.
    
    Parameters:
        df (pd.DataFrame): DataFrame con los datos originales sin nombres de columnas.
        
    Returns:
        pd.DataFrame: DataFrame con nombres de columnas y valores transformados.
    """
    # Nombres de las columnas según la documentación del dataset
    nombres_columnas = [
        "checking_account", "duration", "credit_history", "purpose", "credit_amount",
        "savings_account", "employment_since", "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property", "age", "other_installment_plans",
        "housing", "number_credits", "job", "people_liable", "telephone", "foreign_worker",
        "target"
    ]
    df.columns = nombres_columnas

    # Diccionario de mapeo para transformar códigos a valores descriptivos
    mapeos = {
        "checking_account": {
            "A11": "< 0 DM",
            "A12": "0 <= ... < 200 DM",
            "A13": ">= 200 DM or salary assigned",
            "A14": "no account"
        },
        "credit_history": {
            "A30": "no credits / all paid",
            "A31": "paid at this bank",
            "A32": "paid duly till now",
            "A33": "delay in past",
            "A34": "critical account / other bank"
        },
        "purpose": {
            "A40": "car (new)",
            "A41": "car (used)",
            "A42": "furniture/equipment",
            "A43": "radio/TV",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "vacation?",
            "A48": "retraining",
            "A49": "business",
            "A410": "others"
        },
        "savings_account": {
            "A61": "< 100 DM",
            "A62": "100 <= ... < 500 DM",
            "A63": "500 <= ... < 1000 DM",
            "A64": ">= 1000 DM",
            "A65": "unknown / no account"
        },
        "employment_since": {
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1 <= ... < 4 years",
            "A74": "4 <= ... < 7 years",
            "A75": ">= 7 years"
        },
        "personal_status_sex": {
            "A91": "male, divorced/separated",
            "A92": "female, div/sep/married",
            "A93": "male, single",
            "A94": "male, married/widowed",
            "A95": "female, single"
        },
        "other_debtors": {
            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor"
        },
        "property": {
            "A121": "real estate",
            "A122": "building society / insurance",
            "A123": "car / other",
            "A124": "unknown / no property"
        },
        "other_installment_plans": {
            "A141": "bank",
            "A142": "stores",
            "A143": "none"
        },
        "housing": {
            "A151": "rent",
            "A152": "own",
            "A153": "for free"
        },
        "job": {
            "A171": "unemployed / non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled / official",
            "A174": "management / self-employed / qualified"
        },
        "telephone": {
            "A191": "none",
            "A192": "yes, registered"
        },
        "foreign_worker": {
            "A201": "yes",
            "A202": "no"
        }
    }

    # Aplico los mapeos de manera eficiente
    for col, mapeo in mapeos.items():
        df[col] = df[col].map(mapeo)
        
    return df


def convertir_object_a_categ(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas de tipo 'object' a tipo 'category' para optimizar memoria
    y mejorar el rendimiento de operaciones.
    
    Parameters:
        df (pd.DataFrame): DataFrame con columnas a convertir.
        
    Returns:
        pd.DataFrame: DataFrame con las columnas convertidas.
    """
    columnas_object = df.select_dtypes(include=['object']).columns
    for col in columnas_object:
        df[col] = df[col].astype('category')
    return df


def extraer_genero(df: pd.DataFrame, nombre_nueva_columna: str = 'gender') -> pd.DataFrame:
    """
    Extrae el género de la columna 'personal_status_sex' y crea una nueva columna.
    
    Parameters:
        df (pd.DataFrame): DataFrame que contiene la columna 'personal_status_sex'.
        nombre_nueva_columna (str): Nombre de la columna a crear para almacenar el género.
            Por defecto es 'gender'.
        
    Returns:
        pd.DataFrame: DataFrame con la nueva columna de género añadida.
    """
    # Crear una copia para evitar SettingWithCopyWarning
    df = df.copy()
    
    # Inicializar columna con valores nulos
    df[nombre_nueva_columna] = None
    
    # Extraer género de manera vectorizada
    mask_male = df['personal_status_sex'].str.contains('male')
    mask_female = df['personal_status_sex'].str.contains('female')
    
    df.loc[mask_male, nombre_nueva_columna] = 'male'
    df.loc[mask_female, nombre_nueva_columna] = 'female'
    
    return df