# Trabajo Práctico Final - Equidad en Aprendizaje Automático (Licenciatura en Ciencia de Datos)

**Participantes del proyecto:**

* Gerardo Toboso - getobosobarrios@estudiantes.unsam.edu.ar
* Gianni Bevilacqua - gbevilacqua@estudiantes.unsam.edu.ar
* Javier Spina - jaspina@estudiantes.unsam.edu.ar

Este proyecto corresponde al trabajo práctico final de la materia **Equidad en Aprendizaje Automático** de la Lienciatura en Ciencia de Datos (1er cuatrimestre 2025). El objetivo principal es evaluar y mitigar sesgos en modelos de clasificación aplicados al dataset **German Credit Data**, con foco en cuestiones de equidad de género en la asignación de créditos.

## 📋 Enunciado del Trabajo

El trabajo consiste en desarrollar un modelo que prediga si una persona debería recibir un crédito bancario. Además de evaluar su rendimiento con métricas clásicas como *accuracy*, *precision*, *recall* y *f1-score*, se debe realizar un análisis de equidad (*fairness*), especialmente enfocado en la dimensión de género. Posteriormente, se deben aplicar técnicas de mitigación de sesgos y comparar los resultados con el modelo original.

El desarrollo incluye:

- Análisis exploratorio del dataset.
- Evaluación de métricas de rendimiento y equidad.
- Aplicación de técnicas de mitigación de sesgos.
- Comparación entre modelos con y sin mitigación.
- Reflexión sobre la equidad en contextos reales de aplicación de ML.

## 🎯 Objetivos del Proyecto

- Comprender y aplicar conceptos de equidad en aprendizaje automático.
- Identificar y analizar sesgos en modelos de clasificación.
- Implementar técnicas de mitigación de sesgos y evaluar su impacto.
- Desarrollar una solución completa, reproducible y bien documentada en Python.
- Presentar los resultados de manera clara y profesional.

## 📁 Estructura del Proyecto

```
tp_final_fairness_ML/
│
├── notebooks/                    
│   ├── german_credit_fairness.ipynb               # Notebook principal que contiene el desarrollo del proyecto
│   └── README.md
│
├── informe_trabajo_practico.pdf                   # Informe en formato PDF con los resultados del proyecto 
│
├── requirements.txt              # Dependencias del proyecto
├── setup.py                      # Instalación del proyecto en modo editable
├── README.md                     # Archivo actual
└── .gitignore                    # Exclusiones de Git
```

## ⚙️ Cómo clonar y correr este proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/Gerardo1909/tpfinal_equidad.git
cd tpfinal_equidad
```

### 2. Crear y activar un entorno virtual

```bash
python -m venv venv
source venv/bin/activate      # En Linux/macOS
venv\Scripts\activate.bat     # En Windows
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar el proyecto en modo editable
Esto permite importar los módulos de `utils` desde cualquier notebook sin problemas:

```bash
pip install -e .
```
