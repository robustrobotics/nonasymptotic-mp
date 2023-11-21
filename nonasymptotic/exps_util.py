# Implements any more general algorithms to be run in experiments
from nonasymptotic import ompl
from ompl import base as ob
from ompl import geometric as og
import numpy as np


# TODO: this may just be better to do in the experiment script
def prm_well_approximates_env_path(pdef, cspace, prm, curve_env, conn_rad, epsilon, adv_desc=10, adv_rad=1e-1):
    # Curve is a map from [0, 1] \to \mathbb R^d of the curve we are trying to approximate

    # Epsilon is the solution tolerance
    pass