import numpy as np


def vec_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=2, keepdims=True)
    return vectors / norms

def vec_dot(v1, v2):
    return np.sum(v1 * v2, axis=2)

def vec_edge_direction(points0, points1):
    return points1 - points0

def vec_orthogonal(vectors):
    return np.stack((vectors[..., 1], -vectors[..., 0]), axis=-1)

def vec_vertices_to_edges(vertices):
    return np.roll(vertices, -1, axis=1) - vertices

def vec_project(vertices, axis):
    dots = np.einsum('ijk,ilk->ijl', vertices, axis)
    return np.array([np.min(dots, axis=1), np.max(dots, axis=1)]).transpose(1, 0, 2)

def vec_overlap(projection1, projection2):
    return (projection1[:, 0, :] <= projection2[:, 1, :]) & (projection2[:, 0, :] <= projection1[:, 1, :])

def vec_separating_axis_theorem(polygons_a, polygons_b):
    batch_size = polygons_a.shape[0]
    edges_a = vec_vertices_to_edges(polygons_a)
    edges_b = vec_vertices_to_edges(polygons_b)
    axes_a = vec_normalize(vec_orthogonal(edges_a))
    axes_b = vec_normalize(vec_orthogonal(edges_b))
    results = np.ones(batch_size, dtype=bool)

    for i in range(polygons_a.shape[1]):
        axis = np.concatenate([axes_a[:, [i], :], axes_b[:, [i], :]], axis=1)
        projection_a = vec_project(polygons_a, axis)
        projection_b = vec_project(polygons_b, axis)
        overlapping = vec_overlap(projection_a, projection_b)
        results &= overlapping.all(axis=1)

    return results

def oobb_to_polygon(center, extent, rotation):
    cos_angle = np.cos(rotation)
    sin_angle = np.sin(rotation)

    half_extents = np.array([
        [extent[0], extent[1]],
        [-extent[0], extent[1]],
        [-extent[0], -extent[1]],
        [extent[0], -extent[1]]
    ])

    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    rotated_vertices = np.dot(half_extents, rotation_matrix.T)
    return rotated_vertices + center

def batch_oobb_to_polygons(centers, extents, rotations):
    polygons = []
    for center, extent, rotation in zip(centers, extents, rotations):
        polygons.append(oobb_to_polygon(center, extent, rotation))
    return np.array(polygons)

# Test cases
def test_vec_separating_axis_theorem():
    centers_a = np.array([[0, 0], [2, 2], [4, 4]])
    extents_a = np.array([[1, 1], [1, 1], [1, 1]])
    rotations_a = np.array([0, np.pi / 4, np.pi / 2])

    centers_b = np.array([[1, 1], [4, 4], [7, 7]])
    extents_b = np.array([[1, 1], [1, 1], [1, 1]])
    print(centers_b.shape)
    rotations_b = np.array([0, np.pi / 4, np.pi / 2])

    polygons_a = batch_oobb_to_polygons(centers_a, extents_a, rotations_a)
    polygons_b = batch_oobb_to_polygons(centers_b, extents_b, rotations_b)

    result = vec_separating_axis_theorem(polygons_a, polygons_b)
    print("Test results:", result)


if __name__ == '__main__':
    test_vec_separating_axis_theorem()