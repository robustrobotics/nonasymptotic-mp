import numpy as np
import triangle as tr
from matplotlib import pyplot as plt
from shapely import MultiPolygon, get_exterior_ring, get_num_interior_rings, get_interior_ring, covers, difference, \
    get_coordinates, point_on_surface, Polygon, get_num_points, Point
from shapely.affinity import affine_transform
from shapely.plotting import plot_polygon


def random_point_in_mpolygon(mpolygon, rng=None, vis=False):
    """Return list of k points chosen uniformly at random inside the (multi-)polygon."""
    # This is a helper method in this class so we can share the random seed.
    # someone wrote this so we didn't have to:
    # https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
    # (but I had to correct it because they didn't get the sampler exactly right either).
    if rng is None:
        rng = np.random.default_rng()

    # find points inside holes so we don't sample from holes
    if isinstance(mpolygon, MultiPolygon):
        polys = np.array(mpolygon.geoms)

        mpoly_exterior_rings = get_exterior_ring(polys)
        exterior_polys = np.array(MultiPolygon(zip(mpoly_exterior_rings, [[]] * mpoly_exterior_rings.size)).geoms)

        # gather all the interior rings
        num_mpoly_holes = get_num_interior_rings(polys)
        max_num_holes = np.max(num_mpoly_holes)
        mpoly_interior_rings = np.concatenate(
            [get_interior_ring(polys, i) for i in range(max_num_holes)]
        ) if max_num_holes > 0 else np.array([])
        mpoly_interior_rings = mpoly_interior_rings[mpoly_interior_rings != np.array(None)]
        interior_polys = np.array(MultiPolygon(zip(mpoly_interior_rings, [[]] * mpoly_interior_rings.size)).geoms)

        # check nestedness (numpy vectorized brute force)
        _inpolys, _expolys = np.meshgrid(interior_polys, exterior_polys, indexing="ij")
        _covers = covers(_inpolys, _expolys)
        _cover_inds = np.moveaxis(
            np.mgrid[0:interior_polys.size, 0:exterior_polys.size],
            0, 2
        )[_covers]

        # we're not expecting many intersections, so for loop is okay
        for i_inpoly, i_expoly in _cover_inds:
            interior_polys[i_inpoly] = difference(interior_polys[i_inpoly], exterior_polys[i_expoly])

        # compute correct representative points
        interior_points = get_coordinates(point_on_surface(interior_polys))

    elif isinstance(mpolygon, Polygon):
        mpoly_exterior_rings = np.array([mpolygon.exterior])
        mpoly_interior_rings = mpolygon.interiors

        # convert to Polygons (note: this is the most elegant way I can think of rn)
        mpoly_interior_polys = np.array(MultiPolygon(
            zip(mpoly_interior_rings, [[]] * len(mpoly_interior_rings))
        ).geoms)

        # finally get the interior representative points so triangle can make holes
        interior_points = get_coordinates(point_on_surface(mpoly_interior_polys))
    else:
        raise NotImplementedError('Did not implement sampling from: %s.' % str(type(mpolygon)))

    # coordinates present in a single list.
    mpoly_rings = np.concatenate([mpoly_exterior_rings, mpoly_interior_rings])
    vertices = get_coordinates(mpoly_rings)

    # compute the repeating vertex to delete
    num_pts_in_polys = get_num_points(mpoly_rings)
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
        'holes': np.concatenate([
            np.array([[-100.0, -100.0]]),
            interior_points
        ])
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
