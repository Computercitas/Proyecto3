import numpy as np

def adjust_descriptors_size(descriptors, target_size):
    current_size = descriptors.shape[0]

    if current_size < target_size:
        num_to_add = target_size - current_size
        indices_to_duplicate = np.random.choice(current_size, num_to_add, replace=True)
        duplicated_descriptors = descriptors[indices_to_duplicate]
        adjusted_descriptors = np.vstack([descriptors, duplicated_descriptors])
    
    elif current_size > target_size:
        num_to_remove = current_size - target_size
        indices_to_remove = np.random.choice(current_size, num_to_remove, replace=False)
        adjusted_descriptors = np.delete(descriptors, indices_to_remove, axis=0)
    
    else:
        adjusted_descriptors = descriptors
    
    return adjusted_descriptors

descriptors = np.load("features15k/descriptores.npy")
target_sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000]

for size in target_sizes:
    adjusted_descriptors = adjust_descriptors_size(descriptors, size)
    np.save(f"features15k/descriptores_adjusted_{size}.npy", adjusted_descriptors)
    print(f"Archivo con {size} descriptores guardado como 'descriptores_adjusted_{size}.npy'")
