from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import knn_highd  # KNN-HighD
import knn_rtree  # KNN-RTree
import knn_secuencial  # KNN-Secuencial

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permitir CORS para solicitudes desde cualquier origen

# Cargar descriptores
output_dir = "./Extraccion/features15k"
descriptors, mapping = funcs.load_features(output_dir)

# Método para ejecutar el KNN seleccionado
def execute_knn(query_vector, k, method="KNN-HighD"):
    start_time = time.time()

    if method == "KNN-HighD":
        results = knn_highd.knn_faiss(query_vector, k=k)
    elif method == "KNN-RTree":
        results = knn_rtree.knn_rtree(query_vector, k=k)
    elif method == "KNN-Secuencial":
        results = knn_secuencial.knn_sequential(query_vector, descriptors, k=k)
    else:
        raise ValueError(f"Método KNN no reconocido: {method}")

    elapsed_time = time.time() - start_time
    return results, elapsed_time

# Endpoint para la búsqueda KNN
@app.route('/search/knn', methods=['POST'])
def search_knn():
    try:
        # Obtener parámetros de la solicitud
        data = request.get_json()
        query_image_url = data.get('query_image_url')
        k = int(data.get('k', 8))
        knn_method = data.get('knn_method', 'KNN-HighD')  # Método KNN a ejecutar

        if not query_image_url:
            return jsonify({'error': 'query_image_url parameter is required'}), 400

        # Cargar la imagen de consulta
        query_image = imread(query_image_url)
        query_vector = funcs.extract_features(query_image)  # Suponiendo que tienes una función para extraer los descriptores

        # Ejecutar el KNN seleccionado
        results, query_time = execute_knn(query_vector, k, method=knn_method)

        # Preparar la respuesta
        return jsonify({
            'method': knn_method,
            'query_time': query_time,
            'results': results
        })

    except Exception as e:
        logger.error(f"Error en búsqueda KNN: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
