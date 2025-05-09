# Seguimiento de Progreso del Proyecto

## Objetivos Cumplidos

1. **Estructura del Proyecto**
   - ✅ Creación de estructura modular del proyecto con directorios para notebooks, utilidades y datos
   - ✅ Organización de código para facilitar la reproducibilidad

2. **Conjunto de datos**
   - ✅ Obtención del conjunto de datos del repositorio [Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
   - ✅ Exploración inicial del conjunto de datos

3. **Análisis Exploratorio**
   - ✅ Exploración de la distribución de etiquetas
   - ✅ Exploración de las edades
   - ✅ Exploración de géneros
   - ✅ Identificación de sesgos potenciales

4. **Creación de un modelo inicial**
   - ✅ Elegir e implementar un modelo clasificación (Regresión logística)
   - ✅ Entrenar y evaluar su performance con métricas clásicas
   - ✅ Crear e interpretar la matriz de confusión
   - ✅ Interpretar los resultados desde la perspectiva de un banco

## Objetivos Pendientes

1. **Conjunto de datos - Documentación**
   - ⬜ Completar la documentación sobre motivación del conjunto de datos
   - ⬜ Documentar composición detallada
   - ⬜ Documentar proceso de recopilación
   - ⬜ Documentar preprocesamiento/limpieza/etiquetado
   - ⬜ Documentar usos

2. **Evaluación de equidad del modelo inicial**
   - ⬜ Describir criterios de fairness para este contexto (Statistical Parity, Equalized Odds, Equal Opportunity, Predictive Parity)
   - ⬜ Considerando como métrica de disparidad el módulo de la diferencia junto con un umbral seleccionado por el equipo, analizar si el 
         modelo inicial es justo para cada una de las posibles definiciones estudiadas en el punto anterior
   - ⬜ Justificar elección de criterios relevantes en este contexto bancario

3. **Mitigación de sesgos**
   - ⬜ Seleccionar al menos 2 técnicas de mitigación
   - ⬜ Entrenar modelos ajustados con estas técnicas
   - ⬜ Evaluar performance usando métricas clásicas y crear matrices de confusión
   - ⬜ Evaluar con las mismas métricas de equidad del inciso anterior

4. **Conclusiones**
   - ⬜ Comparar resultados del modelo original y ajustado
   - ⬜ Discutir mejoras en fairness y métricas de performance
   - ⬜ Reflexionar sobre cómo estos cambios impactan en aplicaciones del 
         mundo real y la importancia de la equidad en machine learning