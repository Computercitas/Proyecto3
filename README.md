# Proyecto3

### Autores

|                                                                             | Nombre                                                                   | GitHub                                                     |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------- |
| ![Mariel](https://github.com/MarielUTEC.png?size=50)                        | [Mariel Carolina Tovar Tolentino](https://github.com/MarielUTEC)         | [@MarielUTEC](https://github.com/MarielUTEC)               |
| ![Noemi](https://github.com/NoemiHuarino-utec.png?size=50)                  | [Noemi Alejandra Huarino Anchillo](https://github.com/NoemiHuarino-utec) | [@NoemiHuarino-utec](https://github.com/NoemiHuarino-utec) |
| <img src="https://github.com/Sergio-So.png?size=50" width="50" height="50"> | [Sergio Sebastian Sotil Lozada](https://github.com/Sergio-So)            | [@Sergio-So](https://github.com/Sergio-So)                 |
| ![Davi](https://github.com/CS-DaviMagalhaes.png?size=50)                    | [Davi Magalhaes Eler](https://github.com/CS-DaviMagalhaes)               | [@CS-DaviMagalhaes](https://github.com/CS-DaviMagalhaes)   |
| ![Jose](https://github.com/EddisonPinedoEsp.png?size=50)                    | [Jose Eddison Pinedo Espinoza](https://github.com/EddisonPinedoEsp)      | [@EddisonPinedoEsp](https://github.com/EddisonPinedoEsp)   |


## 1. Introducción

### Objetivo del Proyecto
El objetivo principal de este proyecto es construir un sistema de recuperación de información que permita realizar búsquedas eficientes de objetos multimedia (imágenes o audio) utilizando técnicas de indexación multidimensional.

### Descripción del Dominio de Datos e Importancia de la Indexación
El proyecto se centra en el dominio de la búsqueda de objetos multimedia, como imágenes de productos de moda. La indexación es crucial en este dominio debido a la gran cantidad de datos multimedia disponibles. Un índice multidimensional permite realizar búsquedas rápidas y eficientes al organizar los datos en función de sus características extraídas.

---

## 2. Backend: Índice Multidimensional

### Extracción de Características
Para la extracción de características se utilizó el algoritmo SIFT (Scale-Invariant Feature Transform), el cual es ampliamente reconocido por su robustez en la detección y descripción de puntos clave en imágenes.

#### Proceso de Extracción de Características
El proceso de extracción de características se realiza de manera incremental, permitiendo reanudar el procesamiento desde el último punto guardado en caso de interrupciones. A continuación, se detallan las partes más importantes del código:

#### Función `extract_sift_features`
Esta función se encarga de descargar la imagen desde la URL proporcionada, convertirla a escala de grises si es necesario, y aplicar el algoritmo SIFT para extraer los puntos clave y descriptores. Si no se encuentran descriptores.

### Reducción de Dimensionalidad con PCA

#### Aplicación de PCA
Para reducir la dimensionalidad de los descriptores SIFT, que originalmente tienen 128 componentes, se utilizó el método de Análisis de Componentes Principales (PCA). PCA es una técnica estadística que transforma los datos a un nuevo sistema de coordenadas, donde las nuevas variables (componentes principales) son combinaciones lineales de las variables originales y están ordenadas de manera que las primeras retienen la mayor parte de la varianza presente en los datos originales.

#### Proceso de Reducción de Dimensionalidad
El objetivo de aplicar PCA es reducir el número de componentes de 128 a un número menor, manteniendo la mayor cantidad de información posible. En este caso, se determinó que para retener diferentes porcentajes de la varianza total, se necesitan los siguientes números de componentes:

<!-- Imagen de pca -->
![PCA](imgs/pca.png)

> 90.0% de varianza: 9 componentes  | 
95.0% de varianza: 16 componentes |
99.0% de varianza: 44 componentes |

Para nuestro análisis, se decidió reducir los descriptores a 44 componentes, lo cual permite retener el 99.0% de la varianza. Esta reducción es significativa ya que disminuye la dimensionalidad de los datos, facilitando su manejo y procesamiento, sin perder una cantidad considerable de información.

### KNN Search y Range Search
- **Búsqueda KNN (K-Nearest Neighbors):** Se implementarán algoritmos para encontrar los K objetos más similares a una consulta dada utilizando una cola de prioridad.
- **Búsqueda por Rango:** Recuperará objetos dentro de un radio específico de la consulta. Se experimentará con tres valores de radio diferentes analizando la distribución de las distancias.

### KNN-HighD
El KNN-HighD (K-Nearest Neighbors en alta dimensionalidad) aborda los desafíos asociados con la búsqueda eficiente en espacios de alta dimensionalidad, como la "maldición de la dimensionalidad". Para mitigar estos problemas, se utilizó la biblioteca FAISS (Facebook AI Similarity Search), específicamente con la estructura de índice IVFFlat (Inverted File Flat). Este enfoque combina algoritmos de agrupamiento y optimización para reducir el costo computacional y mejorar la escalabilidad.

### Teoría: La Maldición de la Dimensionalidad

La **maldición de la dimensionalidad** afecta a algoritmos como KNN al trabajar con espacios de alta dimensionalidad, debido a:
- **Pérdida de discriminación:** Las distancias entre puntos tienden a ser similares, lo que reduce la utilidad de métricas como la distancia Euclidiana (L2).
- **Crecimiento exponencial del espacio:** La cantidad de datos necesarios para representar un espacio aumenta exponencialmente con su dimensionalidad.

Esta situación dificulta encontrar los vecinos más cercanos de manera eficiente. Por ello, es necesario implementar métodos que reduzcan el costo computacional y la complejidad.

### Mitigación con FAISS e IVFFlat

FAISS (Facebook AI Similarity Search) utiliza el índice **IVFFlat (Inverted File Flat)** para mitigar estos problemas:
1. **Indexación por Agrupamiento:**
   - Divide los datos en varios clústeres mediante algoritmos como k-means.
   - Cada vector se asigna al clúster más cercano para evitar búsquedas en todo el espacio.

2. **Estrategia Multinivel:**
   - Los datos se organizan en listas invertidas basadas en los centroides de los clústeres.
   - Durante una consulta, solo se evalúan los vectores del clúster más cercano.

3. **Optimización de Distancias (L2):**
   - FAISS utiliza cálculos optimizados para arquitecturas modernas, acelerando las búsquedas.

### Beneficios de IVFFlat

1. **Velocidad:**  
   Reduce el tiempo de búsqueda al limitar las comparaciones al subconjunto más relevante.

2. **Escalabilidad:**  
   Maneja grandes volúmenes de datos al particionarlos eficientemente.

3. **Ahorro de Recursos:**  
   Los índices entrenados se pueden guardar en disco y reutilizar, evitando costos repetitivos de construcción.

### Implementación del KNN-HighD

## 3. Frontend

### Diseño de la GUI
Se diseñará una interfaz gráfica de usuario (GUI) intuitiva para que los usuarios puedan ingresar consultas y visualizar los resultados de búsqueda.

- [Repositorio para correr el Frontend]([https://github.com/Dateadores/Proyecto2](https://github.com/Computercitas/Proyecto2y3-Frontend))

### Visualización de Resultados
Los resultados de búsqueda se mostrarán de forma interactiva y estarán asociados a la búsqueda textual.

---

## 4. Experimentación

En esta sección se detallan los experimentos realizados para evaluar la eficiencia de los diferentes métodos implementados para el sistema de recuperación de información. 

#### Configuración
Se trabajó con un conjunto de datos multimedia de diferentes tamaños (N) y se evaluaron los siguientes algoritmos:
- **KNN-RTree**: Implementación basada en árboles R para realizar búsquedas eficientes.
- **KNN-Secuencial**: Búsqueda lineal que compara cada elemento del conjunto con la consulta.
- **KNN-HighD (FAISS)**: Biblioteca optimizada para espacios de alta dimensionalidad.

Todas las pruebas se realizaron utilizando **K = 8** vecinos más cercanos.


### Visualización de Resultados
Los resultados experimentales se tabularon y graficaron para facilitar el análisis.

#### Tiempo de búsqueda

En esta prueba se evaluó el tiempo necesario para encontrar los K vecinos más cercanos para diferentes tamaños de datos. Los resultados se presentan en la siguiente gráfica:

<img src="imgs/busqueda_KNN.png" width="700"/>

| Tamaño de Datos (N) | KNN Secuencial (ms) | KNN RTree (ms) | KNN HighD (ms) |
|-----------------------|---------------------|----------------|----------------|
| 1000                 | 5.35                | 1.020           | 7.0596           |
| 2000                 | 10.33               | 4.020           | 4.7847           |
| 4000                 | 27.92               | 9.560           | 3.9297          |
| 8000                 | 43.26               | 11.990          | 8.1943          |
| 16000                | 85.27               | 41.125         |  7.8857          |
| 32000                | 173.13              | 65.330          | 8.7014         |
| 64000                | 389.67              | 93.520          | 8.4053          |


**Análisis:**
- **KNN-Secuencial** presenta un crecimiento exponencial en tiempo a medida que aumenta el tamaño del conjunto de datos.
- **KNN-RTree** inicia con mejor rendimiento, pero su eficiencia disminuye para colecciones grandes.
- **KNN-HighD** (FAISS) mantiene un tiempo de búsqueda estable, superando a los demás métodos para tamaños superiores a 4000 datos.



#### Tiempo de construcción de los índices

Esta prueba evaluó el tiempo necesario para construir los índices de los algoritmos que los requieren. Los resultados se muestran a continuación:

<img src="imgs/construccion_KNN.png" width="700"/>

| Tamaño de Datos (N) | KNN RTree (ms)      | KNN HighD (ms) |
|-----------------------|--------------------|----------------|
| 1000                 | 2011.31           | 13.0126          |
| 2000                 | 5063.83           | 17.3626          |
| 4000                 | 9496.43           | 31.0382          |
| 8000                 | 22061.49          | 20.3894          |
| 16000                | 44662.79          | 30.9122          |
| 32000                | 92836.16          | 62.6820          |
| 64000                | 191183.86         | 96.8397          |


**Análisis:**
- **KNN-RTree** requiere un tiempo considerablemente mayor para construir su índice, especialmente a medida que aumenta el tamaño del conjunto de datos.
- **KNN-HighD** demuestra una construcción de índices extremadamente rápida en comparación con KNN-RTree.



### Análisis y Discusión

**Ventajas y Desventajas:**
1. **KNN-Secuencial:**
   - *Ventajas*: Implementación simple y sin necesidad de preprocesamiento.
   - *Desventajas*: Ineficiente para conjuntos de datos grandes debido a su crecimiento exponencial en tiempo de búsqueda.

2. **KNN-RTree:**
   - *Ventajas*: Bueno para tamaños de datos pequeños o medianos.
   - *Desventajas*: Tiempo de construcción de índices muy alto y decrecimiento de eficiencia en colecciones grandes.

3. **KNN-HighD (FAISS):**
   - *Ventajas*: Excelente rendimiento tanto en tiempo de búsqueda como en construcción de índices; escalable y eficiente.
   - *Desventajas*: Dependencia de librerías externas y optimizaciones específicas para hardware.

**Conclusión:**
FAISS se destaca como la mejor solución para el sistema de recuperación de información debido a su equilibrio entre eficiencia y escalabilidad, mitigando los desafíos de la alta dimensionalidad.

### Video 

[Video de Proyecto 3 en Drive](https://drive.google.com/drive/folders/1nNQXjQip6cyppLHK_kav3mBTBc67AnCn?usp=sharing)


