import json
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

def load_features(output_dir):
    """
    Carga descriptores y mapeo desde archivos
    """
    descriptors = np.load(f"{output_dir}/descriptores_pca44.npy")
    with open(f"{output_dir}/checkpoint.json", 'r') as f:
        mapping = json.load(f)
    return descriptors, mapping


# Métrica de distancia: Euclidiana
def euclidean_distance(P, Q):
    return np.sqrt(np.sum((P - Q) ** 2))

def show_results(results, output_dir, query_idx=None, num_results=5):
    """
    Muestra imagen de consulta y resultados usando el mapping de imágenes
    """
    json_path= output_dir + "/image_mapping.json"
    # Leer archivo de mapping
    with open(json_path, 'r') as f:
        image_mapping = json.load(f)
    
    n_imgs = min(len(results), num_results)
    
    # Calcular disposición de la matriz
    if n_imgs > 5:
        n_cols = 5
        n_rows = (n_imgs - 1) // n_cols + 1
        fig = plt.figure(figsize=(15, 3*n_rows))
    else:
        fig = plt.figure(figsize=(15, 5))
    
    try:
        # Obtener información de la imagen de consulta
        query_info = image_mapping[str(query_idx)]
        query_url = query_info['link']
        query_filename = query_info['filename']
        query_img = imread(query_url)
        
        # Mostrar imagen de consulta
        if n_imgs > 5:
            plt.subplot(n_rows + 1, n_cols, 3)
        else:
            plt.subplot(1, n_imgs + 1, 1)
        
        # Agregar borde rojo
        plt.imshow(query_img)
        plt.gca().spines['bottom'].set_color('red')
        plt.gca().spines['top'].set_color('red')
        plt.gca().spines['left'].set_color('red')
        plt.gca().spines['right'].set_color('red')
        plt.gca().spines['bottom'].set_linewidth(5)
        plt.gca().spines['top'].set_linewidth(5)
        plt.gca().spines['left'].set_linewidth(5)
        plt.gca().spines['right'].set_linewidth(5)
        
        plt.title(f"Consulta:\n{query_filename}", fontsize=10)
        plt.axis('on')
        plt.xticks([])
        plt.yticks([])
        
        # Mostrar resultados
        for i, (idx, distance) in enumerate(results[:num_results]):
            try:
                # Obtener información del resultado desde el mapping
                result_info = image_mapping[str(idx)]
                image_url = result_info['link']
                filename = result_info['filename']
                img = imread(image_url)
                
                if n_imgs > 5:
                    plt.subplot(n_rows + 1, n_cols, n_cols + i + 1)
                else:
                    plt.subplot(1, n_imgs + 1, i + 2)
                
                plt.imshow(img)
                plt.title(f"Resultado {i+1}:\n{filename}\nDist: {distance:.4f}", fontsize=10)
                plt.axis('off')
                
            except Exception as e:
                print(f"Error mostrando resultado {idx}: {e}")
    
    except Exception as e:
        print(f"Error mostrando imagen de consulta: {e}")
    
    plt.tight_layout()
    plt.show()

#Seleccionar imagen aleatoria de los descriptores
def select_random_query(descriptors):
    return np.random.randint(len(descriptors))