# 📓 Notebooks del Proyecto

Este directorio contiene los distintos notebooks utilizados durante el desarrollo del trabajo práctico final. Se optó por una estructura **modular** con el objetivo de mantener el flujo de trabajo claro, ordenado y fácilmente reproducible.

## 🧱 Estructura Modular

El trabajo se divide en varias etapas, cada una implementada en un notebook independiente:

- **01_preprocesamiento.ipynb**  
  Realiza la carga de datos y el preprocesamiento inicial.

- **02_EDA.ipynb**  
  Explora el dataset con análisis descriptivos y visualizaciones, incluyendo análisis de género.

- **03_primer_modelo.ipynb**  
  Entrena un modelo base y evalúa su desempeño.

- **04_evaluacion_fairness.ipynb**  
  Evalúa la equidad del modelo base utilizando métricas de fairness.

- **05_modelo_mitigado.ipynb**  
  Aplica técnicas de mitigación de sesgos y entrena un nuevo modelo ajustado en base al original.

- **06_evaluacion_final_fairness.ipynb**  
  Evalúa la equidad y el rendimiento del modelo mitigado, comparándolos con el original.

## 🧾 main.ipynb — Notebook Final de Entrega

Este archivo **resume y condensa** todo el trabajo desarrollado en los notebooks anteriores, integrando los principales resultados, visualizaciones y reflexiones del proyecto. Es el archivo que se entrega como producto final del trabajo práctico, cumpliendo con el requerimiento del enunciado.

> ⚠️ Aunque `main.ipynb` es suficiente para entender el proyecto completo, recomendamos revisar los notebooks individuales si se desea profundizar en cada etapa de desarrollo o revisar detalles del código.

## 💡 Justificación de la Modularización

- Mejora la legibilidad y organización del trabajo.
- Facilita la colaboración entre miembros del grupo.
- Permite ejecutar y testear cada etapa por separado.
- Reduce errores y facilita el debugging.