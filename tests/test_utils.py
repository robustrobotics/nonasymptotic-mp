from nonasymptotic.util import random_point_in_mpolygon

from shapely.geometry import Polygon, MultiPolygon, MultiLineString, MultiPoint
from shapely.plotting import plot_polygon, plot_points

import matplotlib.pyplot as plt
import numpy as np


class TestRandomSampling:
    rng = np.random.default_rng(seed=1)

    square = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    tri = Polygon([(2, 0), (2, 1), (1, 2)])
    tri_square = MultiPolygon([tri, square])

    def sample_membership_test(self, geom):
        samples = np.array(
            [random_point_in_mpolygon(geom, rng=self.rng, vis=True) for _ in range(1)]
        )

        is_member = geom.covers(samples)

        errors = ''

        if not is_member.all():
            errors += "%s is/are not members of geom" % str(samples[is_member])

        fig, ax = plt.subplots()
        plot_polygon(geom, ax=ax, color='blue')
        plot_points(MultiPoint(samples), ax=ax, color='green')
        plt.show()

        assert not errors

    def test_sample_membership_tri(self):
        self.sample_membership_test(self.tri)

    def test_sample_membership_square(self):
        self.sample_membership_test(self.square)

    def test_sample_membership_tri_square(self):
        self.sample_membership_test(self.tri_square)

    def test_bad_call(self):
        import pickle
        import triangle as tr
        with open('bad_triangle_call.pkl', 'rb') as handle:
            data = pickle.load(handle)

        fig, ax = plt.subplots()
        plot_polygon(data['mpolygon'], ax=ax, color='blue')

        del data['mpolygon']
        triangle_out = tr.triangulate(data, 'p')
        triangles = triangle_out['vertices'][triangle_out['triangles']]

        for triangle in triangles:
            plot_polygon(Polygon(triangle), ax=ax, color='purple')
        plt.show()
