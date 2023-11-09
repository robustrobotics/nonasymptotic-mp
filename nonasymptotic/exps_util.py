# Implements any more general algorithms to be run in experiments
import numpy as np


def prm_well_approximates_env_path(curve_env, compute_advance, conn_rad, epsilon):
    # Curve is a map from [0, 1] \to \mathbb R^d of the curve we are trying to approximate

    # Compute advance is a map from a graph path length (w/o edges connected query points),
    # the query points, and epsilon to compute how far to advance t.

    # Epsilon is the solution tolerance

    basepoint = curve_env.curve_start()
    endpoint = curve_env.curve_end()
    while np.isclose(basepoint, endpoint, rtol=1e-5, atol=1e-8):
        # Step 1: find query points
        testpoint = curve_env.march_along_curve(basepoint, rad=conn_rad)  # march along

        # Step 2: run PRM algorithm on basepoint and test point.

        # if failed, then return false immediately
        if False:
            return False

        # 1: compute the stretch difference
        # 2: find the point in the graph that the basepoint is connected to
        stretch_diff = 0
        basepoint_nn = None
        d_first_edge = 0

        search_rad = min(d_first_edge + stretch_diff, conn_rad)

        # Step 3: compute the advancement along the curve
        basepoint = curve_env.march_along_curve(basepoint_nn, rad=search_rad)

    # if we manage to get all the way through without returning false, then
    # we know the curve is okay and PRMmable

    return True
