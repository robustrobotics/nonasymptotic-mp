from shapely.geometry import Point
from shapely.affinity import affine_transform
from shapely.coordinates import get_coordinates
from shapely import get_num_points, MultiPolygon

import numpy as np
import triangle as tr


def random_point_in_mpolygon(mpolygon, rng=None):
    """Return list of k points chosen uniformly at random inside the polygon."""
    # This is a helper method in this class so we can share the random seed.
    # someone wrote this so we didn't have to:
    # https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
    # (but I had to correct it because it wrote down the matrix slightly wrong).
    if rng is None:
        rng = np.random.default_rng()

    poly_boundaries = np.array(mpolygon.boundary.geoms) \
        if isinstance(mpolygon, MultiPolygon) \
        else np.array([mpolygon.boundary])

    # coordinates present in a single list.
    vertices = get_coordinates(poly_boundaries)

    # compute the repeating vertex to delete
    num_pts_in_polys = get_num_points(poly_boundaries)
    n_polys = num_pts_in_polys.shape[0]

    del_vert_indices = np.copy(num_pts_in_polys)
    del_vert_indices[0] -= 1
    del_vert_indices = np.cumsum(del_vert_indices)
    vertices = np.delete(vertices, del_vert_indices, axis=0)

    # now we make the edge arrays
    roll_indices = num_pts_in_polys - 1
    roll_indices[0] -= 1
    roll_indices = np.cumsum(roll_indices)

    n_verts = vertices.shape[0]
    conn_from_arr = np.arange(n_verts)
    conn_to_arr = np.delete(conn_from_arr, roll_indices)
    conn_to_arr = np.insert(
        conn_to_arr,  # subtract arange to account for shift of deleted vertices
        np.concatenate([[0], num_pts_in_polys[:-1] - 1]) - np.arange(n_polys),
        roll_indices
    )
    segments = np.hstack([conn_from_arr.reshape(-1, 1), conn_to_arr.reshape(-1, 1)])

    # compute triangulation
    triangle_out = tr.triangulate(
        {
            'vertices': vertices,
            'segments': segments,
            'holes': [[1000.0, 1000.0]]  # put a point that will always be outside because of boundedness of input space
        },
        'p'  # so the boundaries are included in the triangulation
    )

    # extract triangles (in terms of coordinates) using advanced indexing
    # dims: (N tries) X (3 points in tries) X (2D coords)
    triangles = triangle_out['vertices'][triangle_out['triangles']]

    # compute areas by evaluating det explicitly (easier than forming explicit matrices)
    x1s = triangles[:, 0, 0]
    x2s = triangles[:, 1, 0]
    x3s = triangles[:, 2, 0]

    y1s = triangles[:, 0, 1]
    y2s = triangles[:, 1, 1]
    y3s = triangles[:, 2, 1]

    areas = 0.5 * np.abs(
        x1s * (y2s - y3s) + x2s * (y3s - y1s) + x3s * (y1s - y2s)
    )

    areas = np.array(areas)
    areas /= np.sum(areas)

    x, y = rng.uniform(size=(2,))
    if x + y > 1:
        p = Point(1 - x, 1 - y)
    else:
        p = Point(x, y)

    (x0, y0), (x1, y1), (x2, y2) = rng.choice(triangles, p=areas)
    transform = [x1 - x0, x2 - x0, y1 - y0, y2 - y0, x0, y0]
    tp = affine_transform(p, transform)
    return tp
