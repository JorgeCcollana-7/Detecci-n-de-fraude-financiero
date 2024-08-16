# DETECCION DE FRAUDE FINANCIERO CON MACHINE LEARNING
## Autor
 [@JORGE](https://github.com/JorgeCcollana-7)
# Acerca del conjunto de datos
## Contexto
Existe una falta de conjuntos de datos públicos disponibles sobre servicios financieros y, especialmente, en el ámbito emergente de las transacciones de dinero móvil. Los conjuntos de datos financieros son importantes para muchos investigadores y, en particular, para nosotros, que realizamos investigaciones en el ámbito de la detección de fraudes. Parte del problema es la naturaleza intrínsecamente privada de las transacciones financieras, que conduce a que no haya conjuntos de datos disponibles públicamente.

## DESCRIPCION DEL PROYECTO

Este proyecto utiliza técnicas de Machine Learning para detectar transacciones fraudulentas en un conjunto de datos. El objetivo es identificar patrones que puedan distinguir entre transacciones legítimas y fraudulentas.

## Herramientas utilizadas

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

#### Boxplot de Monto
![box monto](https://github.com/user-attachments/assets/86141d35-90b6-411e-b244-67d6219124d0)

Hay un gran desbalance en la variable objetivo, por lo que será necesario realizar un balanceo más adelante para mejorar el modelo.
Existen muchos outliers en los montos de las transacciones, pero según el boxplot, la mayor concentración, sin contar los outliers, se encuentra entre 10K y 210K aproximadamente

![box monto sin outliers](https://github.com/user-attachments/assets/f671a623-7cb4-4fd7-8346-fc4df8b267dd)


#### Frecuencia de Tipos de Transacciones
El gráfico muestra la frecuencia de diferentes tipos de transacciones en un sistema financiero. Los tipos de transacciones se enumeran en el eje vertical, mientras que el eje horizontal representa la cantidad de transacciones en millones. 
![frecuencia de transacciones](https://github.com/user-attachments/assets/5407cf6f-3edf-4960-ae13-417670e30233)

**Retiro:** Es el tipo de transacción más frecuente, con un poco más de 2 millones de transacciones.  
**Pago:** Tiene una frecuencia ligeramente inferior a los retiros, con casi 2 millones de transacciones.  
**Depósito:** Es el tercer tipo de transacción más común, con alrededor de 1.5 millones de transacciones.  
**Transferencia:** Tiene una menor frecuencia en comparación con los anteriores, con aproximadamente 1 millón de transacciones.  
**Débito:** Es el tipo de transacción menos frecuente, con menos de 0.5 millones de transacciones.  

#### Total de Transacciones por Día del Mes
![Total de Transacciones por Día del Mes](https://github.com/user-attachments/assets/6cc19c2e-f61d-495d-aa1c-de4a2a2b2efa)

Una gran cantidad de transacciones ocurren los primeros días del mes, se puede deber a que a la mayoría de las personas perciben su salario mensualmente y deciden hacer transacciones.
También existe una gran concentración de transacciones entre los días 6-17 del mes, algunas perconas podrían recibir sus ingresos cada quince días o semanalmente.
Es muy común que para los días finales del mes, las personas hayan terminado con gran parte de su dinero, lo que explicaría las pocas transacciones entre los días 18-31.

#### Porcentaje de Transacciones Fraudulentas por Día
![Porcentaje de Transacciones Fraudulentas por Día](https://github.com/user-attachments/assets/56d42483-792c-40c8-999b-1663adb1a247)
**Bajo porcentaje general:** Durante la mayor parte del mes, el porcentaje de transacciones fraudulentas es muy bajo, lo que sugiere que el sistema de seguridad o las medidas preventivas han sido efectivas en esos días.

**Picos de fraude:** Hay algunos días con picos aislados (por ejemplo, 4.5%), lo que podría indicar eventos específicos que llevaron a un aumento temporal en el fraude. Estos días podrían requerir un análisis más profundo para identificar patrones o vulnerabilidades específicas.

**Día con 100% de fraude:** El día final con un 100% de transacciones fraudulentas es extremadamente inusual y alarmante. Esto podría indicar un fallo grave en la seguridad del sistema, un ataque dirigido, o un error en los datos. Este día debe ser investigado de manera prioritaria para entender las causas subyacentes y prevenir futuros incidentes similares.

#### Mapa de Calor de Correlaciones
![Mapa de Calor de Correlaciones](https://github.com/user-attachments/assets/0f956f78-e04c-4e6c-a0cc-029199896f70)  

El mapa de calor muestra la correlación entre diversas variables relacionadas con las transacciones, representadas en una matriz donde cada celda refleja la relación entre dos variables específicas. El color de cada celda indica la fuerza de la correlación: el rojo intenso señala una correlación positiva fuerte y el azul intenso una correlación negativa fuerte.  

Según las correlaciones observadas, el tipo de pago con mayor probabilidad de fraude es el tipo de transferencia (type_TRANSFER).  

Los métodos de pago en efectivo ("type_CASH_IN" y "type_CASH_OUT") y los métodos de pago con tarjeta ("type2_CC" y "type2_CM") están vinculados a una menor probabilidad de fraude. Esto podría ser porque estos métodos son más difíciles de falsificar o utilizar de forma fraudulenta. Además, las transacciones de mayor monto tienden a presentar una menor probabilidad de fraude, posiblemente porque los defraudadores prefieren realizar transacciones de menor monto para evitar levantar sospechas.  

## 3. Construcción de Modelos
Matriz de confusión de 2 modelos seleccionados
![Matriz de Confusión](https://github.com/user-attachments/assets/656b4c5a-fd94-4c46-bc19-a9ba893bee24)
Observamos que los dos modelos con mejorres métricas es el Random Forest 

## 4. Evaluación y Selección del Modelo
![Métricas de Prueba por Modelo](https://github.com/user-attachments/assets/69187bd8-b1df-45e0-9be5-3f6c8479c1a4)
**Accuracy (Exactitud):** 

Random Forest: 94.5%  
Regresión Logística: 90.1%  
El modelo de Random Forest muestra una mejor exactitud en comparación con la Regresión Logística, indicando que clasifica correctamente una mayor proporción de observaciones.  

**Precision:**

Random Forest: 96.0%  
Regresión Logística: 90.7%  
El modelo de Random Forest tiene una mayor precisión, lo que significa que es más efectivo para identificar correctamente las transacciones fraudulentas entre las que predice como tales.  

**Recall (Sensibilidad):**

Random Forest: 91.3%  
Regresión Logística: 86.3%  
El modelo de Random Forest tiene un mejor recall, lo que indica que es más efectivo para identificar las transacciones   fraudulentas reales dentro del total de transacciones fraudulentas.  

**F1-Score:**

Random Forest: 93.6%  
Regresión Logística: 88.5%  
El F1-Score, que es una media armónica de la precisión y el recall, también es superior en el modelo de Random Forest, lo que sugiere un mejor equilibrio entre precisión y sensibilidad.  

**AUC-ROC:**

Random Forest: 98.8%  
Regresión Logística: 96.8%  
El modelo de Random Forest muestra un área bajo la curva ROC (AUC-ROC) más alta, lo que indica un mejor rendimiento general en la distinción entre transacciones fraudulentas y no fraudulentas.

## CONCLUSION
En general, el modelo de Random Forest supera al modelo de Regresión Logística en todas las métricas evaluadas. Esto sugiere que Random Forest es el modelo más adecuado para la detección de fraudes en este conjunto de datos, ya que ofrece una mejor combinación de precisión, sensibilidad y rendimiento global.

