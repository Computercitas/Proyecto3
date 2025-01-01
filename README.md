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

### Dominio de Datos
El proyecto se centra en el dominio de la búsqueda de objetos multimedia, como imágenes de productos de moda. La indexación es crucial en este dominio debido a la gran cantidad de datos multimedia disponibles. Un índice multidimensional permite realizar búsquedas rápidas y eficientes al organizar los datos en función de sus características extraídas.

**Descripción del Dataset:**

- **URL**: [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data)
- **Contenido**: El conjunto de datos incluye imágenes de productos de moda, con atributos asociados como categorías de productos, marcas, colores, precios y tipos de ropa.
  
**Atributos del Dataset**:
  - **Imágenes**: Cada producto está representado por una o más imágenes, lo que permite realizar análisis visuales.
  - **Categorías**: Las imágenes están etiquetadas con categorías específicas como ropa de hombre, mujer, accesorios, etc.
  - **Atributos Adicionales**: Los productos también tienen información sobre su marca, precio y color, lo cual es útil para realizar tareas de clasificación, análisis de tendencias y recomendaciones.

Este dataset se utiliza principalmente para entrenar modelos de aprendizaje automático en tareas como la clasificación de productos por categoría, el análisis de características visuales o la creación de sistemas de recomendación.

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

### KNN Secuencial

La búsqueda KNN secuencial es un enfoque simple para encontrar los `k` vecinos más cercanos a un vector de consulta. En lugar de utilizar estructuras de datos especializadas como R-trees o KD-trees, este enfoque recorre todos los descriptores de manera secuencial y calcula la distancia entre el vector de consulta y cada descriptor.

#### Explicación del Código: Búsqueda KNN Secuencial

##### Función `knn_sequential`:

- **Entrada**:
  - `query_vector`: El vector de consulta.
  - `descriptors`: El conjunto de descriptores contra los cuales se realiza la búsqueda.
  - `k`: El número de vecinos más cercanos a devolver (por defecto, 8).

- **Proceso**:
  1. **Cálculo de Distancia Euclidiana**: 
     La función utiliza `np.linalg.norm(descriptors - query_vector, axis=1)` para calcular la distancia euclidiana entre el `query_vector` y cada descriptor en el conjunto de datos. Esto se hace a nivel de fila, comparando cada descriptor con el vector de consulta.
  
  2. **Ordenación de Distancias**:
     Se ordenan las distancias calculadas y se seleccionan los primeros `k` descriptores más cercanos con `np.argsort(distances)[:k]`. Esto devuelve los índices de los descriptores más cercanos.

  3. **Almacenamiento de Resultados**:
     Los resultados se almacenan en una lista de tuplas `(índice, distancia)`, que se devuelve como resultado de la búsqueda.

  4. **Cálculo del Tiempo de Ejecución**:
     El tiempo total de la operación se calcula utilizando `perf_counter()`, y se imprime en milisegundos para medir el rendimiento del algoritmo.

##### Cálculo de Distancias Euclidianas:

La distancia euclidiana entre dos vectores \(A\) y \(B\) se calcula con la siguiente fórmula:

$$
\text{Distancia}(A, B) = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}
$$

Donde:
- \(n\) es el número de dimensiones del vector.
- \(A_i\) y \(B_i\) son los componentes del vector en la dimensión \(i\).
- La distancia euclidiana calcula la suma de las diferencias al cuadrado entre cada componente de los dos vectores, y luego toma la raíz cuadrada de esa suma.

##### Optimización de Búsqueda:

- **`np.argsort(distances)`**: Devuelve los índices que ordenarían el array de distancias en orden ascendente.
- **`[:k]`**: Selecciona los primeros `k` índices, correspondientes a los `k` descriptores más cercanos.

##### Ventajas y Limitaciones:

- **Ventajas**:
  - Es un enfoque simple y fácil de entender.
  - No requiere estructuras de datos avanzadas como R-trees o KD-trees, lo que lo hace adecuado para datasets pequeños a medianos.

- **Limitaciones**:
  - La búsqueda es costosa en términos de tiempo computacional cuando el número de descriptores es grande. La complejidad temporal es \(O(n)\), donde \(n\) es el número de descriptores.
  - No es escalable para grandes volúmenes de datos, ya que requiere calcular la distancia a todos los puntos en el espacio, lo cual es ineficiente para conjuntos de datos masivos.

##### Resultados:

- La función devuelve los `k` vecinos más cercanos junto con sus respectivas distancias.
- El tiempo total de la búsqueda también se imprime en milisegundos, lo que permite evaluar el rendimiento del algoritmo en función del tamaño del conjunto de datos y la eficiencia de la búsqueda.

### KNN Rtree
El algoritmo KNN-RTree se utiliza para realizar búsquedas de los k vecinos más cercanos de manera eficiente. El R-tree es una estructura de índice multidimensional, especialmente adecuada para manejar datos espaciales y multidimensionales, como los descriptores de imágenes. A continuación, se explica cómo se implementa:

1. **Creación del Índice R-tree:**
   El índice R-tree se crea utilizando la librería `rtree` que nos permite insertar los descriptores y organizar la información en un formato jerárquico. Cada descriptor se mapea a un nodo en el R-tree, y se asignan límites superiores e inferiores en el espacio de características.

2. **Búsqueda KNN:**
   Cuando se realiza una consulta para encontrar los k vecinos más cercanos, el R-tree organiza y reduce el espacio de búsqueda, permitiendo encontrar rápidamente los puntos más cercanos a la consulta.

#### Explicación del Código:

##### Creación del R-Tree:
- La clase `Rtree` recibe los descriptores (vectores de características) y la dimensión de los vectores. La propiedad `idx` contiene el índice R-tree que permite realizar búsquedas eficientes.
- En el método `create_rtree_index()`, cada descriptor se inserta en el índice R-tree utilizando sus valores como límites superiores e inferiores.

##### Búsqueda de K Vecinos más Cercanos:
- El método `knn_rtree()` realiza la búsqueda de los k vecinos más cercanos para un vector de consulta específico. La consulta se realiza utilizando la función `nearest()` del índice R-tree, lo que optimiza el tiempo de búsqueda.
- El resultado es una lista de los vecinos más cercanos junto con sus distancias a la consulta.

##### Beneficios de KNN-RTree:
- **Eficiencia:** La estructura R-tree reduce significativamente la cantidad de comparaciones necesarias al organizar los datos en una jerarquía espacial.
- **Escalabilidad:** A medida que el conjunto de datos crece, el índice R-tree permite realizar búsquedas de manera eficiente sin necesidad de explorar todos los puntos en el espacio.

  
### KNN-HighD
El enfoque KNN-HighD (K-Nearest Neighbors en alta dimensionalidad) permite realizar búsquedas eficientes de los k vecinos más cercanos en espacios vectoriales de alta dimensionalidad, como los generados por descriptores SIFT. Sin embargo, trabajar con datos de alta dimensionalidad presenta desafíos significativos debido a la maldición de la dimensionalidad.

### La Maldición de la Dimensionalidad

En espacios de alta dimensionalidad, muchos algoritmos, como KNN, enfrentan problemas como:
- **Pérdida de discriminación:** Las distancias entre puntos tienden a ser similares, lo que reduce la utilidad de métricas como la distancia Euclidiana (L2).
- **Crecimiento exponencial del espacio:** La cantidad de datos necesarios para representar un espacio aumenta exponencialmente con su dimensionalidad.
- **Costos Computacionales:** Los cálculos en alta dimensionalidad son intensivos en tiempo y memoria.

Para mitigar estos problemas, en nuestro proyecto utilizamos las siguientes técnicas:
- **Reducción de Dimensionalidad con PCA:** Los descriptores SIFT, originalmente de 128 dimensiones, fueron reducidos a 44 componentes principales, reteniendo el 99% de la varianza. Esto permitió disminuir el tamaño del espacio de búsqueda sin perder información crítica.
- **Indexación Multidimensional con IVFFlat:** Se empleó la estructura IVFFlat de FAISS para optimizar las búsquedas KNN al dividir el espacio en clústeres más pequeños y manejables.


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

### Implementación en el código:

- **Construcción del Índice IVFFlat:**
El índice **FAISS IVFFlat** se creó a partir de los descriptores reducidos. En el archivo `knn-highd.py`, esto se implementa con:

```python
d = descriptors.shape[1]  # Dimensión de los descriptores
quantizer = faiss.IndexFlatL2(d)  # Estructura base con distancia Euclidiana (L2)
index = faiss.IndexIVFFlat(quantizer, d, 10)  # Índice IVFFlat con 10 centroides
index.train(descriptors)  # Entrenamiento del índice con los datos
index.add_with_ids(descriptors, np.arange(descriptors.shape[0]))  # Agregar datos al índice
```

El índice se guardó en disco para su reutilización futura con la función `faiss.write_index`.

- **Búsqueda KNN**:

Para realizar búsquedas KNN, se utilizó la función `index.search`, que identifica los k vecinos más cercanos a un vector de consulta:

```python
query_vector = descriptors[random].reshape(1, -1)  # Vector de consulta
distances, indices = index.search(query_vector, k=8)  # Buscar 8 vecinos más cercanos
```

Esto devuelve las distancias y los índices de los vectores más cercanos al vector de consulta. Los resultados se visualizan utilizando una función auxiliar:

```python
funcs.show_results(faiss_results, output_dir, random, k)
```


## 3. Frontend

El frontend de este proyecto fue desarrollado utilizando **React** en combinación con **Vite**, lo que permite un desarrollo ágil y eficiente gracias a las características de rendimiento y simplicidad que Vite ofrece como herramienta de construcción.

### Estructura e Implementación

El código fuente del frontend se encuentra en el siguiente repositorio:
[Proyecto2y3-Frontend](https://github.com/Computercitas/Proyecto2y3-Frontend/tree/main/Proyecto3/frontend3)

Dentro de este repositorio, la funcionalidad principal está implementada en el componente **Consulta** (`Consulta.tsx`), donde se incluye una interfaz para realizar búsquedas KNN basadas en imágenes. A continuación, se detalla cómo funciona esta sección:

#### Formulario de Búsqueda
- El usuario puede ingresar el índice de un descriptor de imagen para realizar la búsqueda.
- Se pueden seleccionar diferentes métodos de búsqueda KNN (`KNN-Secuencial`, `KNN-RTree` y `KNN-HighD`) mediante botones interactivos.
- Permite especificar el número de resultados `Top-K`.

#### Ejecución de la Consulta
- Al hacer clic en el botón "Buscar", se envía una solicitud a la API backend (almacenada en `api.py`) utilizando el método POST.
- La API devuelve los resultados más cercanos según el descriptor ingresado y el método KNN seleccionado.

#### Visualización de Resultados
- Los resultados incluyen información sobre los archivos encontrados, distancias, y enlaces a las imágenes correspondientes.
- Si hay resultados, se muestran en una cuadrícula interactiva; en caso contrario, se informa al usuario que no se encontraron coincidencias.


### Detalles Técnicos

#### Librerías Utilizadas
- **React**: Para la construcción de componentes.
- **Vite**: Para la construcción y optimización del proyecto.
- **Fetch API**: Para realizar solicitudes al backend.

#### Estilos
Los estilos están definidos en el archivo `Consulta.css`, proporcionando una interfaz intuitiva y limpia.

#### Características Clave
- Renderizado dinámico de los resultados en función de la respuesta de la API.
- Manejo de errores para garantizar que el usuario ingrese información válida antes de realizar una consulta.
- Uso de `useState` para gestionar el estado del descriptor, método seleccionado, resultados y tiempo de consulta.

Este diseño asegura que el frontend sea flexible, fácil de usar, y se integre sin problemas con el backend para proporcionar resultados en tiempo real.

## Imágenes del frontend

### Página Principal
![Página Principal](ruta/a/tu/imagen_pagina_principal.png)

### Búsqueda
![Búsqueda](ruta/a/tu/imagen_resultados.png)

![Resultados de la búsqueda](ruta/a/tu/imagen_resultados.png)

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


