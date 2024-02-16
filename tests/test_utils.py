from nonasymptotic.util import random_point_in_mpolygon, compute_sauer_shelah_bound, compute_numerical_bound, \
    compute_rho

from shapely.geometry import Polygon, MultiPolygon, MultiLineString, MultiPoint
from shapely.plotting import plot_polygon, plot_points

import matplotlib.pyplot as plt
import numpy as np
import scipy


class TestRandomSampling:
    rng = np.random.default_rng(seed=1)

    square = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    tri = Polygon([(2, 0), (2, 1), (1, 2)])
    tri_square = MultiPolygon([tri, square])

    def sample_membership_test(self, geom):
        samples = np.array(
            [random_point_in_mpolygon(geom, rng=self.rng, vis=False) for _ in range(1000)]
        )

        is_member = geom.covers(samples)

        errors = ''

        if not is_member.all():
            errors += "%s is/are not members of geom" % str(samples[is_member])

        # fig, ax = plt.subplots()
        # plot_polygon(geom, ax=ax, color='blue')
        # plot_points(MultiPoint(samples), ax=ax, color='green')
        # plt.show()

        assert not errors

    def test_sample_membership_tri(self):
        self.sample_membership_test(self.tri)

    def test_sample_membership_square(self):
        self.sample_membership_test(self.square)

    def test_sample_membership_tri_square(self):
        self.sample_membership_test(self.tri_square)


class TestSauerShelah:

    def test_sauer_shelah_combinatorial_computation(self):
        # a very simple system sanity checks to make sure
        # we have the sauer-shelah lemma bound down.
        rho = 0.0
        vc_dim = 3
        m_samples = 5
        comb_sum = (scipy.special.comb(2 * m_samples, 0) +
                    scipy.special.comb(2 * m_samples, 1) +
                    scipy.special.comb(2 * m_samples, 2) +
                    scipy.special.comb(2 * m_samples, 3))
        comb_sum = int(comb_sum)

        assert comb_sum == int(compute_sauer_shelah_bound(m_samples, rho, vc_dim) / 2)

    def test_sauer_shelah_decay_computation(self):
        rho = 1.0
        vc_dim = 0  # this cancels the ss_comb_sum
        m_samples = 1
        decay_factor = 2 * np.exp2(-rho * m_samples / 2)

        assert decay_factor == compute_sauer_shelah_bound(m_samples, rho, vc_dim)

    def test_numerical_search(self):
        # making sure we recover the samples of a specific probabilit

        delta = 0.5
        epsilon = 0.5
        dim = 2
        vol_env = 1.0
        rho = compute_rho(delta, epsilon, dim, vol_env)

        vc_dim = dim + 1
        m_samples = 350

        failure_prob_lb = compute_sauer_shelah_bound(m_samples, rho, vc_dim)
        failure_prob_ub = compute_sauer_shelah_bound(m_samples + 1, rho, vc_dim)

        failure_prob_to_find = (failure_prob_ub + failure_prob_lb) / 2
        recovered_m_samples = compute_numerical_bound(delta, 1.0 - failure_prob_to_find, vol_env, dim, epsilon)

        assert recovered_m_samples[0] == m_samples + 1
