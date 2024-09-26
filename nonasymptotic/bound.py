import sys

import math
import numpy as np
import scipy
from decimal import Decimal


def compute_vol_unit_sphere(_dim):
    return np.pi ** (_dim / 2) / scipy.special.gamma(_dim / 2 + 1)


def compute_epsilon_net_radius(clearance, tol):
    alpha = tol / np.sqrt(1 + tol ** 2) if tol is not None else 1.0
    return alpha * clearance


def compute_rho(clearance, tol, dim, vol_env):
    measures_unit_sphere = compute_vol_unit_sphere(dim) / vol_env
    return measures_unit_sphere * (compute_epsilon_net_radius(clearance, tol) ** dim)

def compute_ss_comb_sum(m_samples, vc_dim):
    ss_comb_sum = sum([scipy.special.comb(2 * m_samples, _d, exact=True) for _d in range(vc_dim + 1)])
    return ss_comb_sum

def compute_sauer_shelah_bound_log2(m_samples, rho, vc_dim):
    # switched to exact computation so scipy can handle big integers,
    # but the function can no longer be vectorized :(

    ss_comb_sum = compute_ss_comb_sum(m_samples, vc_dim)

    # to dodge some numerical imprecision of the exponentiation, let's do the calculation
    #   in 2-logspace. math.log2 doesn't convert to float (unlike numpy), so we use that here.

    log2_prob = math.log2(ss_comb_sum) + (-rho * m_samples / 2) + 1
    return log2_prob

def compute_net_radius_from_prob(dim, n_samples, failure_prob, vol_env, ad_hoc_magnitude_correction=1):
    n_samples = np.ceil(n_samples / ad_hoc_magnitude_correction)
    ss_comb_sum = compute_ss_comb_sum(n_samples, dim + 1)
    vol_unit_sphere = compute_vol_unit_sphere(dim)
    measure_unit_sphere = vol_unit_sphere / vol_env

    return (
            (2 / n_samples / measure_unit_sphere)
            * (math.log2(ss_comb_sum) - math.log2(failure_prob))
    ) ** (1 / dim)


def compute_blumer_bound(net_rad, dim, gamma, vol_env):
    vol_net_ball = compute_vol_unit_sphere(dim) * net_rad ** dim
    measure_vol_net_ball = vol_net_ball / vol_env
    vc_dim = dim + 1
    return np.maximum(4 / measure_vol_net_ball * np.log2(2 / gamma),
                      8 * vc_dim / measure_vol_net_ball * np.log2(13 / measure_vol_net_ball))


def doubling_sample_search_over_log2_prob_bound(samples_to_log2_prob, success_prob):
    """
    :param samples_to_log2_prob: A function that consumes an integer number of samples
    and outputs a probability.
    :param success_prob: Probability for random construction to be met.
    :param log_base: Log base of probability bound function.
    :return: Number of samples required to meet success probability.
    """
    # we're exploiting the monotonicity of the increase/decrease behavior of
    # the probability bound.
    log_failure_prob = math.log2(1 - success_prob)

    m_samples_lb = 1
    m_samples_ub = 2
    while True:
        cand_gamma = samples_to_log2_prob(m_samples_ub)
        next_gamma = samples_to_log2_prob(m_samples_ub + 1)

        # move past the desired probability and ensure we are 'downhill'
        if log_failure_prob >= cand_gamma >= next_gamma:
            break

        m_samples_ub *= 2

        if m_samples_ub > 1e300:
            raise OverflowError("Reached sample threshold for useful computation.")

    # next, binary search down to the right number of samples.
    while True:
        if m_samples_ub == m_samples_lb + 1:
            return m_samples_ub
        elif m_samples_ub <= m_samples_lb:
            raise ArithmeticError('Something wrong happened.')

        test_samples = int((m_samples_lb + m_samples_ub) / 2)
        # test if the tester is the sample count we're looking for
        cand_gamma = samples_to_log2_prob(test_samples)
        next_gamma = samples_to_log2_prob(test_samples + 1)

        # if the candidate failure probability is still too high, or
        # if we're not going 'downhill' yet, then move up the lower bound.
        if cand_gamma > log_failure_prob or next_gamma > cand_gamma:
            m_samples_lb = test_samples

        # otherwise, move down the upper bound
        else:
            m_samples_ub = test_samples


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

    alpha = tol / np.sqrt(1 + tol ** 2) if tol is not None else 1.0

    conn_r = 2 * (alpha + np.sqrt(1 - alpha ** 2)) * clearance
    rho = compute_rho(clearance, tol, dim, coll_free_volume)

    vc_dim = dim + 1

    # first, forward search on a doubling scheme to overshoot on the number of samples.
    # we stop if the probability is #1 decaying, and #2 we've exceeded desired probability

    return doubling_sample_search_over_log2_prob_bound(
        lambda m: compute_sauer_shelah_bound_log2(m, rho, vc_dim),
        success_prob
    ), conn_r


def compute_connection_radius(clearance, tol):
    return 2 * clearance * (tol + 1) / np.sqrt(1 + tol ** 2)


def compute_tail_knn_radius_log_prob_hoeffding(m_samples, k_neighbors, conn_rad, dim, vol_env):
    measures_unit_sphere = compute_vol_unit_sphere(dim) / vol_env
    p_r_ball = measures_unit_sphere * (conn_rad ** dim)
    tail_val = k_neighbors / (m_samples - 1) - p_r_ball

    # removing a degenerate case (if p_r_ball is too large, then
    # all points will land in it and mess everything up)
    # but preserving some monotonicity properties of the prob bound
    tail_val = np.maximum(tail_val, 0.0)

    log_p_fail = np.log(m_samples) - 2 * (m_samples - 1) * tail_val ** 2
    return log_p_fail


def compute_tail_knn_radius_log_prob_hoeffding_mean_variant(m_samples, k_neighbors, conn_rad, dim, vol_env):
    measures_unit_sphere = compute_vol_unit_sphere(dim) / vol_env
    p_r_ball = measures_unit_sphere * (conn_rad ** dim)
    tail_val = k_neighbors / (m_samples - 1) / p_r_ball - 1

    # removing a degenerate case (if p_r_ball is too large, then
    # all points will land in it and mess everything up)
    # but preserving some monotonicity properties of the prob bound
    tail_val = np.maximum(tail_val, 0.0)
    log_p_fail = np.log(m_samples) - 2 * (m_samples - 1) * tail_val ** 2 / (p_r_ball ** 2)
    return log_p_fail


def compute_tail_knn_radius_log_prob_chernoff_generic(m_samples, k_neighbors, conn_rad, dim, vol_env):
    measures_unit_sphere = compute_vol_unit_sphere(dim) / vol_env
    p_r_ball = np.minimum(measures_unit_sphere * (conn_rad ** dim), 1.0)
    mu = (m_samples - 1) * p_r_ball
    tail_val = k_neighbors

    log_p_fail = np.log(m_samples) - mu + tail_val * (1 + np.log(mu) - np.log(tail_val))
    return log_p_fail


def compute_tail_knn_radius_log_prob_chernoff_kl(m_samples, k_neighbors, conn_rad, dim, vol_env,
                                                 enable_union_bound_factor=True):
    measures_unit_sphere = compute_vol_unit_sphere(dim) / vol_env
    p_r_ball = np.minimum(measures_unit_sphere * (conn_rad ** dim), 1.0)
    tail_val = np.maximum(k_neighbors / m_samples - p_r_ball, 0.0)

    ppt_dist = np.array([p_r_ball + tail_val, 1.0 - p_r_ball - tail_val])
    p_dist = np.array([p_r_ball, 1.0 - p_r_ball])

    if enable_union_bound_factor:
        log_p_fail = np.log(m_samples) - scipy.stats.entropy(ppt_dist, qk=p_dist) * (m_samples - 1)
    else:
        log_p_fail = - scipy.stats.entropy(ppt_dist, qk=p_dist) * (m_samples - 1)

    return log_p_fail


def halving_radius_search_over_log_prob_bound(radius_to_log_prob, rad_lb, rad_ub, success_prob, tol=1e-6):
    """
    Finds largest radius past the union bound hump.
    """
    assert rad_lb < rad_ub
    failure_prob = 1 - success_prob
    log_failure_prob = np.log(failure_prob)

    while True:
        if rad_ub - rad_lb <= tol:
            return rad_lb

        elif rad_ub < rad_lb:
            raise ArithmeticError('Something wrong happened.')

        test_rad = (rad_lb + rad_ub) / 2
        test_gamma = radius_to_log_prob(test_rad)
        next_test_gamma = radius_to_log_prob(test_rad - tol / 2)

        # ensure we are: decreasing as radius gets smaller and within the prob bound
        if next_test_gamma >= test_gamma or test_gamma > log_failure_prob:
            rad_ub = test_rad
        else:
            rad_lb = test_rad

def compute_eff_conn_rad_bound(m_samples, k_neighbors, dim, vol_env, failure_prob):
    measure_unit_sphere = np.minimum(compute_vol_unit_sphere(dim) / vol_env, 1.0)
    numer = k_neighbors - np.sqrt(2 * k_neighbors * (np.log(m_samples) - np.log(failure_prob)))
    denom = (m_samples - 1) * measure_unit_sphere
    return math.pow(numer / denom, 1 / dim)

def linear_radius_search_over_prob_bound(radius_to_prob, rad_lb, rad_ub, success_prob, tol=1e-6):
    # Sanity check search
    failure_prob = 1 - success_prob
    rad = rad_lb
    while True:
        if rad > rad_ub:
            raise ArithmeticError('Adjust tol, search must be finer-grained.')

        test_gamma = radius_to_prob(rad + tol)
        if test_gamma > failure_prob:
            return rad



## ZOMBIE CODE
# def doubling_radius_search_over_log2_prob_bound(radius_to_log_prob, rad_lb, success_prob, tol=1e-6):
#     """
#     Does not assume that there is a union bound hump. Finds
#     the smallest radius satisfying probability.
#     """
#     assert rad_lb > 0.0
#
#     failure_prob = 1 - success_prob
#     log_failure_prob = math.log2(failure_prob)
#
#     rad_ub = rad_lb
#     while True:
#         if rad_ub < 0 or rad_ub >= 10:
#             raise ArithmeticError('Overflow or output radius too large to be useful.')
#
#         if radius_to_log_prob(rad_ub) > log_failure_prob:
#             rad_ub *= 2
#         else:
#             break
#
#     while True:
#         if rad_ub - rad_lb <= tol:
#             return rad_ub
#         elif rad_ub < rad_lb:
#             raise ArithmeticError('Something wrong happened.')
#
#         test_rad = rad_ub + rad_lb / 2
#         if radius_to_log_prob > log_failure_prob:
#             # if the test failure probability is too large, then we need to grow radius for
#             # a larger net (less likely to fail)
#             rad_lb = test_rad
#         else:
#             # if the test failure probability is too small, then shrink the radius
#             # if the tst failure probability is too small,
#             rad_ub = test_rad
#
