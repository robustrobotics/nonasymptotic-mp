from __future__ import print_function

import numpy as np
from pybullet_tools.pr2_primitives import Conf, Trajectory, create_trajectory, Command
from pybullet_tools.utils import get_point, get_custom_limits, all_between, pairwise_collision, \
    plan_joint_motion, joints_from_names, set_pose, \
    remove_body, get_visual_data, get_pose, \
    wait_for_duration, create_body, visual_shape_from_data, LockRenderer, plan_nonholonomic_motion, \
    child_link_from_joint, Attachment, OOBB, get_oobb, get_oobb_vertices, Pose, AABB, get_aabb, Point
from pybullet_tools.separating_axis import separating_axis_theorem
from pddlstream.language.constants import Output

from nonasymptotic.envs import Environment
from nonasymptotic.prm import SimpleNearestNeighborRadiusPRM
from typing import Tuple, List
from dataclasses import dataclass
import random

VIS_RANGE = 2
COM_RANGE = 2*VIS_RANGE

BASE_JOINTS = ['x', 'y', 'theta']


def get_base_joints(robot):
    return joints_from_names(robot, BASE_JOINTS)

class Ray(Command):
    _duration = 1.0
    def __init__(self, body, start, end):
        self.body = body
        self.start = start
        self.end = end
        self.pose = get_pose(self.body)
        self.visual_data = get_visual_data(self.body)
    def apply(self, state, **kwargs):
        print(self.visual_data)
        with LockRenderer():
            visual_id = visual_shape_from_data(self.visual_data[0]) # TODO: TypeError: argument 5 must be str, not bytes
            cone = create_body(visual_id=visual_id)
            #cone = create_mesh(mesh, color=(0, 1, 0, 0.5))
            set_pose(cone, self.pose)
        wait_for_duration(self._duration)
        with LockRenderer():
            remove_body(cone)
            wait_for_duration(1e-2)
        wait_for_duration(self._duration)
        # TODO: set to transparent before removing
        yield
    def __repr__(self):
        return '{}->{}'.format(self.start, self.end)


def get_reachable_test(problem, iterations=10, **kwargs):
    initial_confs = {rover: Conf(rover, get_base_joints(rover))
                     for rover in problem.rovers}
    # TODO: restarts -> max_restarts
    motion_fn = get_motion_fn(problem, restarts=0, max_iterations=iterations, smooth=0, **kwargs)
    def test(rover, bq):
        bq0 = initial_confs[rover]
        result = motion_fn(rover, bq0, bq)
        return result is not None
    return test


def get_cfree_ray_test(problem, collisions=True):
    def test(ray, rover, conf):
        if not collisions or (rover == ray.start) or (rover == ray.end):
            return True
        conf.assign()
        collision = pairwise_collision(ray.body, rover)
        #if collision:
        #    wait_for_user()
        return not collision
    return test



def get_above_gen(problem, max_attempts=1, custom_limits={}, collisions=True, **kwargs):
    obstacles = problem.fixed if collisions else []
    reachable_test = get_reachable_test(problem, custom_limits=custom_limits, collisions=collisions, **kwargs)

    def gen(rover, rock):
        base_joints = get_base_joints(rover)
        x, y, _ = get_point(rock)
        lower_limits, upper_limits = get_custom_limits(rover, base_joints, custom_limits)
        while True:
            for _ in range(max_attempts):
                theta = np.random.uniform(-np.pi, np.pi)
                base_conf = [x, y, theta]
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(rover, base_joints, base_conf)
                bq.assign()
                if any(pairwise_collision(rover, b) for b in obstacles):
                    continue
                if not reachable_test(rover, bq):
                    continue
                yield Output(bq)
                break
            else:
                yield None
    return gen

#######################################################

class BagOfBoundingBoxes(Environment):

    def __init__(self, seed, robot_shape:AABB, obstacle_oobbs:List[OOBB], custom_limits):
        self.obstacles = obstacle_oobbs
        self.robot_shape = robot_shape
        self.custom_limits = custom_limits
        self.rng = np.random.default_rng(seed)

    def sample_from_env(self):
        return np.array([random.uniform(v[0], v[1]) for _, v in self.custom_limits.items()])

    def arclength_to_curve_point(self, t_normed):
        raise NotImplementedError

    def is_motion_valid(self, start, goal):
        print(start)
        print(goal)
        start_oobb = OOBB(self.robot_shape, Pose(Point(x=start[0], y=start[1], z=0)))
        goal_oobb = OOBB(self.robot_shape, Pose(Point(x=goal[0], y=goal[1], z=0)))
        
        for obstacle in self.obstacles:
            ov = get_oobb_vertices(obstacle)
            if not separating_axis_theorem(get_oobb_vertices(start_oobb), ov) \
                or not separating_axis_theorem(get_oobb_vertices(goal_oobb), ov):
                return False
        return True

    def is_prm_epsilon_delta_complete(self, prm, tol):
        raise NotImplementedError

    def distance_to_path(self, query_points):
        raise NotImplementedError
    
def get_nonasy_motion_fn(problem, custom_limits={}, collisions=True, teleport=False, holonomic=False, reversible=False, algorithm="prm", num_samples=10, connect_radius=None, **kwargs):
    
    def test(rover, q1, q2, fluents=[]):
        start = np.array(q1.values)
        goal = np.array(q2.values)
        
        seed = 0
        
        obstacles = set(problem.fixed)
        obstacle_oobbs = [get_oobb(obstacle) for obstacle in obstacles]
        prm_env_2d = BagOfBoundingBoxes(seed=seed, robot_shape=get_aabb(rover), obstacle_oobbs=obstacle_oobbs, custom_limits=custom_limits)
        prm = SimpleNearestNeighborRadiusPRM(32, 
                                     prm_env_2d.is_motion_valid, 
                                     prm_env_2d.sample_from_env, 
                                     prm_env_2d.distance_to_path, 
                                     seed=seed, verbose=False)
        prm.grow_to_n_samples(1000)
        d, path = prm.query_best_solution(start, goal)
        
        # if(path is None):
        #     return None
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
