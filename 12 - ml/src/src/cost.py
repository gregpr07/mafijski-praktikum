import numpy as np


def meanSquare(A, B, derivative=False):
    if not derivative:
        if type(A) == np.ndarray:
            return np.sum(np.square(A - B)) / A.size
        return np.sum((A - B) ** 2)
    elif derivative:
        return 2 * (A - B)


cost = {
    'meanSquare': meanSquare
}
