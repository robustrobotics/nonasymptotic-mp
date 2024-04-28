import sys

import numpy as np
import scipy


def compute_vol_unit_sphere(_dim):
    return np.pi ** (_dim / 2) / scipy.special.gamma(_dim / 2 + 1)


def compute_epsilon_net_radius(clearance, tol):
    alpha = tol / np.sqrt(1 + tol ** 2) if tol is not None else 1.0
    return alpha * clearance


def compute_rho(clearance, tol, dim, vol_env):
    measures_unit_sphere = compute_vol_unit_sphere(dim) / vol_env
    return measures_unit_sphere * (compute_epsilon_net_radius(clearance, tol) ** dim)


def compute_sauer_shelah_bound(m_samples, rho, vc_dim):
    ss_comb_sum = np.sum(scipy.special.comb(2 * m_samples, np.arange(vc_dim + 1)))
    decay_rate = 2 * np.exp2(-rho * m_samples / 2)

    return ss_comb_sum * decay_rate


def doubling_search_over_sauer_shelah(rho, vc_dim, success_prob):
    # we're exploiting the monotonicity of the increase/decrease behavior of the probability bound.
    failure_prob = 1 - success_prob

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
        if m_samples_ub == m_samples_lb + 1:
            return m_samples_ub
        elif m_samples_ub <= m_samples_lb:
            raise ArithmeticError('Something wrong happened.')

        test_samples = int((m_samples_lb + m_samples_ub) / 2)
        # test if the tester is the sample count we're looking for
        cand_gamma = compute_sauer_shelah_bound(test_samples, rho, vc_dim)

        if cand_gamma > failure_prob:
            m_samples_lb = test_samples
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

    return doubling_search_over_sauer_shelah(rho, vc_dim, success_prob), conn_r


def compute_connection_radius(clearance, tol):
    return 2 * clearance * (tol + 1) / np.sqrt(1 + tol ** 2)
