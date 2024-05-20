import numpy as np


def cross_2d(v, w):
    return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]


def detect_intersect(s1e1, s1e2, s2e1, s2e2):
    """
    Vectorized numerical segment intersection check using
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    I think this is just a direct calculation in the dual point/half-plane space.
    :param s1e1: segment 1 endpoint 1
    :param s1e2: segment 1 endpoint 2
    :param s2e1: segment 2 endpoint 1
    :param s2e2: segment 2 endpoint 2
    :return:
    """
    assert s1e1.shape == s1e2.shape and s2e1.shape == s2e2.shape

    if len(s1e1.shape) < 2:
        s1e1 = s1e1[np.newaxis, :]
        s1e2 = s1e2[np.newaxis, :]

    if len(s2e1.shape) < 2:
        s2e1 = s2e1[np.newaxis, :]
        s2e2 = s2e2[np.newaxis, :]

    s1_dir = s1e2 - s1e1
    s2_dir = s2e2 - s2e1

    cross_dir = cross_2d(s1_dir, s2_dir)
    intersect_coeff_unnormed1 = cross_2d(s2e1 - s1e1, s2_dir)
    intersect_coeff_unnormed2 = cross_2d(s2e1 - s1e1, s1_dir)

    # then do some vectorized casework (we're really going to rely on some numpy ufunc stuff)

    # first, check for collinearity (avoid some bad numerical cases we'd miss)
    is_collinear = np.equal(cross_dir, 0.0) & np.equal(intersect_coeff_unnormed2, 0.0)

    # next, check parallel  (avoid some bad numerics)
    # not actually necessary because of nice numpy numerics handling
    # is_parallel = np.equal(cross_dir, 0.0) & np.not_equal(intersect_coeff_unnormed2, 0.0)

    # check for intersection in nondenerate cases
    intersect_coeff1 = intersect_coeff_unnormed1 / cross_dir
    intersect_coeff2 = intersect_coeff_unnormed2 / cross_dir
    is_cleanly_intersecting = (
            (0 <= intersect_coeff1) & (intersect_coeff1 <= 1) &
            (0 <= intersect_coeff2) & (intersect_coeff2 <= 1)
    )

    return np.logical_or(is_cleanly_intersecting, is_collinear)
