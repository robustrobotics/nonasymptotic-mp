from __future__ import print_function

import numpy as np
from pybullet_tools.pr2_primitives import Conf, Trajectory, create_trajectory, Command
from pybullet_tools.utils import get_point, get_custom_limits, all_between, pairwise_collision, \
    plan_joint_motion, joints_from_names, set_pose, get_oobb,\
    remove_body, get_visual_data, get_pose, get_aabb_extent, aabb_from_extent_center,\
    wait_for_duration, create_body, visual_shape_from_data, LockRenderer, plan_nonholonomic_motion, \
    child_link_from_joint, Attachment, OOBB, get_oobb_vertices, Pose, AABB, get_aabb
from separating_axis import vec_separating_axis_theorem
from pddlstream.language.constants import Output
from nonasymptotic.envs import Environment
from nonasymptotic.prm import SimpleNearestNeighborRadiusPRM
from typing import List
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from tqdm import tqdm
import itertools
import random
import copy

VIS_RANGE = 2
COM_RANGE = 2*VIS_RANGE

BASE_JOINTS = ['x', 'y', 'theta']


# Function to reorder vertices to form a convex hull
def hull(vertices):
    # Compute the convex hull
    hull = ConvexHull(vertices)
    # Reorder vertices according to the hull vertices
    ordered_vertices = [vertices[index] for index in hull.vertices]
    return ordered_vertices

def get_base_joints(robot):
    return joints_from_names(robot, BASE_JOINTS)

class BagOfBoundingBoxes(Environment):

    def __init__(self, seed, robot_shape:AABB, obstacle_oobbs:List[OOBB], custom_limits):
        self.obstacles = obstacle_oobbs

        self.bounds = self.extents(itertools.chain(*[get_oobb_vertices(oobb) for oobb in self.obstacles]))
        self.robot_shape = robot_shape
        self.custom_limits = custom_limits
        self.rng = np.random.default_rng(seed)

        self.robot_verts = np.array(hull(self.oobb_flat_vertices(OOBB(self.robot_shape, Pose()))))
        self.obstacle_hulls = [np.array(hull(self.oobb_flat_vertices(obstacle))) for obstacle in self.obstacles]

    def extents(self, vertices):
        # Unpack the vertices into separate lists for x, y, and z dimensions
        x_coords, y_coords, z_coords = zip(*vertices)

        # Calculate min and max along each dimension
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)

        return ((min_x, max_x), (min_y, max_y), (min_z, max_z))

    def sample_from_env(self):
        return np.array([random.uniform(self.bounds[0][0], self.bounds[0][1]), 
                         random.uniform(self.bounds[1][0], self.bounds[1][1])])

    def arclength_to_curve_point(self, t_normed):
        raise NotImplementedError

    def oobb_flat_vertices(self, oobb):
        diff_thresh = 0.001
        verts = get_oobb_vertices(oobb)
        verts2d = []
        for vert in verts:
            unique = True
            for vert2d in verts2d:
                if (
                    np.linalg.norm(np.array(vert[:2]) - np.array(vert2d[:2]))
                    < diff_thresh
                ):
                    unique = False
            if unique:
                verts2d.append(vert[:2])
        assert len(verts2d) == 4
        return verts2d
    
    def is_motion_valid(self, start, goal):
        
        valids = np.ones(start.shape[0]).astype(bool)
        start_exp  = np.tile(np.expand_dims(start, axis=1), (1, self.robot_verts.shape[0], 1))
        goal_exp  = np.tile(np.expand_dims(goal, axis=1), (1, self.robot_verts.shape[0], 1))
        robot_start_verts = np.tile(np.expand_dims(self.robot_verts, axis=0), (start.shape[0], 1, 1)) + start_exp
        robot_goal_verts = np.tile(np.expand_dims(self.robot_verts, axis=0), (start.shape[0], 1, 1)) + goal_exp
        num_steps = int(np.max(np.min(np.abs(robot_start_verts-robot_goal_verts), axis=1))/0.15)+2
        alphas = np.linspace(0, 1, num_steps)
        for obstacle_hull, alpha in tqdm(list(itertools.product(self.obstacle_hulls, alphas))):
                interm_verts = robot_start_verts + alpha*(robot_goal_verts-robot_start_verts)
                valids = valids & ~vec_separating_axis_theorem(interm_verts, np.tile(obstacle_hull, (interm_verts.shape[0], 1, 1)))
        return valids.astype(bool)

    def is_prm_epsilon_delta_complete(self, prm, tol):
        raise NotImplementedError

    def distance_to_path(self, points):
        return np.zeros((points.shape[0], ))

    

def get_ik(problem):
    def ik(pose):
        conf = copy.deepcopy(problem.init_conf)
        conf.positions = pose[0][:2]
        return Output(conf)
    return ik

def get_anytime_motion_fn(problem, custom_limits={}, collisions=True, teleport=False, holonomic=False, reversible=False, algorithm="prm", start_samples=10, end_samples=100, factor=1.5, connect_radius=None, **kwargs):
    
    def test(rover, q1, q2, fluents=[]):
        start = np.array(q1.values[:2])
        goal = np.array(q2.values[:2])

        seed = 0
        
        obstacles = set(problem.obstacles)
        obstacle_oobbs = [get_oobb(obstacle) for obstacle in obstacles]
        robot_shape = aabb_from_extent_center(get_aabb_extent(get_aabb(rover)))
        prm_env_2d = BagOfBoundingBoxes(seed=seed, robot_shape=(robot_shape), obstacle_oobbs=obstacle_oobbs, custom_limits=custom_limits)
        prm = SimpleNearestNeighborRadiusPRM(32, 
                                     prm_env_2d.is_motion_valid, 
                                     prm_env_2d.sample_from_env, 
                                     prm_env_2d.distance_to_path, 
                                     seed=seed, verbose=False)
        
        assert prm_env_2d.is_motion_valid(np.expand_dims(start, axis=0), np.expand_dims(start, axis=0)), "Start in collision"
        assert prm_env_2d.is_motion_valid(np.expand_dims(goal, axis=0), np.expand_dims(goal, axis=0)), "Goal in collision"
        
        num_samples = start_samples
        while(num_samples<end_samples+1):
            print("Num samples: "+str(int(num_samples)))
            prm.grow_to_n_samples(int(num_samples))
            _, path = prm.query_best_solution(start, goal)
            if(len(path)>0):
                break
            num_samples = num_samples*factor
        
        print("Path: "+str(path))
        if(len(path) == 0):
            print("Max samples reached")
        else:
            print("Found solution in")
            print(num_samples)

        path = np.concatenate([np.expand_dims(start, axis=0), path, np.expand_dims(goal, axis=0)], axis=0).tolist()
        
        ht = create_trajectory(rover, q2.joints, path)
        return Output(ht)
        
    return test


def get_motion_fn(problem, custom_limits={}, collisions=True, teleport=False, holonomic=False, reversible=False, algorithm="prm", num_samples=10, connect_radius=None, **kwargs):
    def test(rover, q1, q2, fluents=[]):
        if teleport:
            ht = Trajectory([q1, q2])
            return Output(ht)

        base_link = child_link_from_joint(q1.joints[-1])
        q1.assign()
        attachments = []
        movable = set()
        for fluent in fluents:
            predicate, args = fluent[0], fluent[1:]
            if predicate == 'AtGrasp'.lower():
                r, b, g = args
                attachments.append(Attachment(rover, base_link, g.value, b))
            elif predicate == 'AtPose'.lower():
                b, p = args
                assert b not in movable
                p.assign()
                movable.add(b)
            # elif predicate == 'AtConf'.lower():
            #     continue
            else:
                raise NotImplementedError(predicate)

        obstacles = set(problem.fixed) | movable if collisions else []
        q1.assign()
        if holonomic:
            path = plan_joint_motion(rover, q1.joints, q2.values, custom_limits=custom_limits,
                                     attachments=attachments, obstacles=obstacles, self_collisions=False, 
                                     algorithm=algorithm, num_samples=num_samples, connect_distance=connect_radius, **kwargs)
        else:
            path = plan_nonholonomic_motion(rover, q1.joints, q2.values, reversible=reversible, custom_limits=custom_limits,
                                            attachments=attachments, obstacles=obstacles, self_collisions=False, 
                                            algorithm=algorithm, num_samples=num_samples, connect_distance=connect_radius, **kwargs)
        if path is None:
            return None
        
        ht = create_trajectory(rover, q2.joints, path)
        return Output(ht)
    return test
