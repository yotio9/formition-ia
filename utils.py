import numpy as np


def embedding_functions(a: np.ndarray, b: np.ndarray) -> float:
    normes = np.linalg.norm(a) * np.linalg.norm(b)
    if normes == 0:
        return 0.0
    return float(np.dot(a, b) / normes)