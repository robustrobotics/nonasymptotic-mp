from shapely.geometry import Point
from shapely.plotting import plot_polygon
from shapely.affinity import affine_transform
from shapely.coordinates import get_coordinates
from shapely import get_num_points, MultiPolygon, Polygon

import numpy as np
import triangle as tr
import scipy
import matplotlib.pyplot as plt
import sys



def compute_vol_unit_sphere(_dim):
    return np.pi ** (_dim / 2) / scipy.special.gamma(_dim / 2 + 1)


def compute_rho(delta, epsilon, dim, vol_env):
    measures_unit_sphere = compute_vol_unit_sphere(dim) / vol_env
    return measures_unit_sphere * (epsilon * delta / np.sqrt(1 + epsilon ** 2)) ** dim


def compute_sauer_shelah_bound(m_samples, rho, vc_dim):
    ss_comb_sum = np.sum(scipy.special.comb(2 * m_samples, np.arange(vc_dim + 1)))
    decay_rate = 2 * np.exp2(-rho * m_samples / 2)

    return ss_comb_sum * decay_rate


def compute_numerical_bound(clearance, success_prob, coll_free_volume, dim, tol):
    """
    :param clearance: delta-clearance of the environment
    :param coll_free_volume: volume of the collision-free configuration space. could be an upper bound.
    :param dim: dimension of system (in the linear algebraic/DOF sense!)
    :param tol: stretch factor on the optimal path (epsilon), if tol is None, will bound for existence of path
    but not optimality of the approximation of the path.
    :param success_prob: probability PRM well-approximates all optimal delta clear paths up to tol.
    :return: an upper bound of number of samples required, and the connection radius of the PRM.
    """
    if success_prob <= 0.0 or success_prob >= 1.0:
        raise ArithmeticError("Success probability must be between 0.0 and 1.0, noninclusive.")

    failure_prob = 1.0 - success_prob
    alpha = tol / np.sqrt(1 + tol ** 2) if tol is not None else 1.0

    conn_r = 2 * (alpha + np.sqrt(1 - alpha ** 2)) * clearance
    rho = compute_rho(clearance, tol, dim, coll_free_volume)

    vc_dim = dim + 1

    # first, forward search on a doubling scheme to overshoot on the number of samples.
    # we stop if the probability is #1 decaying, and #2 we've exceeded desired probability

    # we're exploiting the monotonicity of the increase/decrease behavior of the probability bound.
    m_samples_lb = 1
    m_samples_ub = 2
    while True:
        cand_gamma = compute_sauer_shelah_bound(m_samples_ub, rho, vc_dim)
        next_gamma = compute_sauer_shelah_bound(m_samples_ub + 1, rho, vc_dim)
        if failure_prob >= cand_gamma > next_gamma:
            break

        m_samples_lb = m_samples_ub
        m_samples_ub *= 2

        if m_samples_ub > sys.maxsize:
            raise OverflowError("Reached sample threshold for useful computation.")

    # next, binary search down to the right number of samples.
    while True:
        if m_samples_ub == m_samples_lb + 1 :
            return m_samples_ub, conn_r
        elif m_samples_ub <= m_samples_lb:
            raise ArithmeticError('Something wrong happened.')

        test_samples = int((m_samples_lb + m_samples_ub) / 2)
        # test if the tester is the sample count we're looking for
        cand_gamma = compute_sauer_shelah_bound(test_samples, rho, vc_dim)

        if cand_gamma > failure_prob:
            m_samples_lb = test_samples
        else:
            m_samples_ub = test_samples


def random_point_in_mpolygon(mpolygon, rng=None, vis=False):
    """Return list of k points chosen uniformly at random inside the polygon."""
    # TODO: need to consider if there are internal holes!
    # This is a helper method in this class so we can share the random seed.
    # someone wrote this so we didn't have to:
    # https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
    # (but I had to correct it because they didn't get the sampler exactly right either).
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
        np.cumsum(np.concatenate([[0], num_pts_in_polys[:-1] - 1])) - np.arange(n_polys),
        roll_indices
    )
    segments = np.hstack([conn_from_arr.reshape(-1, 1), conn_to_arr.reshape(-1, 1)])

    # compute triangulation
    data = {
        'vertices': vertices,
        'segments': segments,
        'holes': [[101.0, 100.0]]
        # put a point that will always be outside because of boundedness of input space
    }

    triangle_out = tr.triangulate(data, 'p')  # so the boundaries are included in the triangulation

    # extract triangles (in terms of coordinates) using advanced indexing
    # dims: (N tries) X (3 points in tries) X (2D coords)
    triangles = triangle_out['vertices'][triangle_out['triangles']]

    if vis:
        fig, ax = plt.subplots()
        plot_polygon(mpolygon, ax=ax, color='blue')
        for triangle in triangles:
            plot_polygon(Polygon(triangle), ax=ax, color='purple')
        plt.show()

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
