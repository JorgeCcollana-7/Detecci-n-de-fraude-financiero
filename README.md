# Acerca del conjunto de datos
## Contexto
Existe una falta de conjuntos de datos públicos disponibles sobre servicios financieros y, especialmente, en el ámbito emergente de las transacciones de dinero móvil. Los conjuntos de datos financieros son importantes para muchos investigadores y, en particular, para nosotros, que realizamos investigaciones en el ámbito de la detección de fraudes. Parte del problema es la naturaleza intrínsecamente privada de las transacciones financieras, que conduce a que no haya conjuntos de datos disponibles públicamente.

## DESCRIPCION DEL PROYECTO

Este proyecto utiliza técnicas de Machine Learning para detectar transacciones fraudulentas en un conjunto de datos. El objetivo es identificar patrones que puedan distinguir entre transacciones legítimas y fraudulentas.

## Herramientas utilizadas
- Colab
- Python
- Librerías:
  - Pandas
  - Numpy
  - Time
  - Matplotlib
  - StringIO
  - Sklearn
  - Imblearn
  - Warnings
 
Los datos base fueron extraidos de fuentes oficiales, [https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data?select=PS_20174392719_1491204439457_log.csv)

## 1. Preprocesamiento de datos
En esta sección, realizamos lo siguiente:

Limpieza de datos, incluido el manejo de valores faltantes y nulos.
Ajuste y normalización de datos, y conversión de tipos de datos de la fuente.
Ingeniería de características para mejorar la construcción de modelos de aprendizaje automático.
## 2. Exploración de datos
Exploremos los datos del marco de datos a través de visualizaciones.

#### Distribución de Fraudes
![distribucion de fraudes](https://github.com/user-attachments/assets/f52d0e8a-b75f-48fc-a87e-36a1d60bfb36)

El gráfico muestra la distribución de transacciones fraudulentas frente a no fraudulentas. 
La gran mayoría de las transacciones (99.87%) no son fraudulentas, mientras que solo el 0.13% corresponde a fraudes. 
Esta desbalanceada proporción resalta la necesidad de un modelo robusto para detectar fraudes de manera efectiva.

Existe un desbalanceo muy grande entre la variable objetivo, es necesario hacer un balanceo más adelante para obtener un mejor modelo.










