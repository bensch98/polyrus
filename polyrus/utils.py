import numpy as np
import scipy

def unpack_dict_list(d):
    return [i for s in d.values() for i in s]

def fit_plane(points: np.ndarray) -> tuple[3]:
    """
    Args:

    Returns:
        n: normal
        d: offset
        mean: point on plane 
    """
    mean = np.mean(points, axis=0)
    points -= mean
    _, _, v = np.linalg.svd(points)
    n = v[2,:]
    d = -np.dot(n, mean)
    return n, d, mean

def fit_line(points: np.ndarray) -> tuple[2]:
    mean = points.mean(axis=0)
    points -= mean
    _, _, v = scipy.linalg.svd(points)
    return v[0], mean

def ray_triangle_intersection(direction: np.ndarray, point:np.ndarray, face:np.ndarray) -> np.ndarray:
    # TODO: change naming -> maybe use conventions from linear algebra
    EPSILON = 1e-6
    v0, v1, v2 = face[0], face[1], face[2]
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(direction, e2)
    a = np.dot(e1, h)
    if -EPSILON < a < EPSILON:
        return np.array([])
    f = 1.0 / a
    s = point - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return np.array([])
    q = np.cross(s, e1)
    v = f * np.dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return np.array([])
    t = f * np.dot(e2, q)
    if t > EPSILON:
        return point + direction * t
    return np.array([])

def plane_line_intersection(n, p, v, t):
    numerator = np.dot(n, (p - t))
    denominator = np.dot(n, v)
    if denominator == 0:
        return None
    else:
        return t + (numerator / denominator) * v

def point_to_plane_distance(n: np.ndarray, p: np.ndarray, d: float) -> tuple[2]:
    distance = np.dot(n, p) + d
    direction = 1 if distance > 0 else -1
    return distance, direction

def point_to_pcd_distance(p: np.ndarray, pcd: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pcd - p, axis=1)