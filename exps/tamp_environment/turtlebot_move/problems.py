import numpy as np

from collections import OrderedDict
from pybullet_tools.utils import set_point, Point, create_box, \
    stable_z, load_model, TURTLEBOT_URDF, joints_from_names, \
    set_joint_positions, get_joint_positions, remove_body, HideOutput, \
    GREY, TAN, get_bodies, pairwise_collision, sample_placement, get_aabb, wait_if_gui
import random

BOT_RADIUS = 0.179


def sample_placements(body_surfaces, obstacles=None, min_distances={}):
    if obstacles is None:
        obstacles = [body for body in get_bodies() if body not in body_surfaces]
    obstacles = list(obstacles)
    # TODO: max attempts here
    for body, surface in body_surfaces.items():
        min_distance = min_distances.get(body, 0.01)
        while True:
            pose = sample_placement(body, surface)
            if pose is None:
                return False
            if not any(pairwise_collision(body, obst, max_distance=min_distance)
                       for obst in obstacles if obst not in [body, surface]):
                obstacles.append(body)
                break
    return True

BASE_JOINTS = ['x', 'y', 'theta']

def get_base_joints(robot):
    return joints_from_names(robot, BASE_JOINTS)

def get_base_conf(robot):
    return get_joint_positions(robot, get_base_joints(robot))

def set_base_conf(robot, conf):
    set_joint_positions(robot, get_base_joints(robot), conf)

KINECT_FRAME = 'camera_rgb_optical_frame' # eyes
#KINECT_FRAME = 'eyes'

#######################################################

class RoversProblem(object):
    def __init__(self, rover=None, landers=[], objectives=[], rocks=[], soils=[], stores=[], limits=[], body_types=[], goal_conf=None):
        self.rover = rover
        self.goal_conf = goal_conf
        self.landers = landers
        self.objectives = objectives
        self.rocks = rocks
        self.soils = soils
        self.stores = stores
        self.limits = limits
        no_collisions = [self.rover] + self.rocks
        self.fixed = set(get_bodies()) - set(no_collisions)
        self.body_types = body_types
        self.costs = False

#######################################################
def hallway(robot_scale=1, dd=0.5):
    base_extent = 3.0
    base_limits = (-base_extent/2.*np.ones(2), base_extent/2.*np.ones(2))
    mound_height = 0.1

    floor = create_box(base_extent, base_extent, 0.001, color=TAN) # TODO: two rooms
    set_point(floor, Point(z=-0.001/2.))

    wall1 = create_box(base_extent + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall1, Point(y=base_extent/2., z=mound_height/2.))
    wall2 = create_box(base_extent + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall2, Point(y=-base_extent/2., z=mound_height/2.))
    wall3 = create_box(mound_height, base_extent + mound_height, mound_height, color=GREY)
    set_point(wall3, Point(x=base_extent/2., z=mound_height/2.))
    wall4 = create_box(mound_height, base_extent + mound_height, mound_height, color=GREY)
    set_point(wall4, Point(x=-base_extent/2., z=mound_height/2.))

    body_types = []
    initial_surfaces = OrderedDict()
    
    robot_width = BOT_RADIUS*robot_scale
    offset = robot_width + mound_height
    rover_conf = (base_extent/2.0-offset, -base_extent/2.0+offset, np.pi)
    goal_conf = (-base_extent/2.0+offset, base_extent/2.0-offset, 0)
    
    box_width = (base_extent/2.0-(mound_height+2*robot_width))*2-dd
    body = create_box(box_width, box_width, mound_height*4, color=GREY)
    initial_surfaces[body] = floor
    
    rover = load_model(TURTLEBOT_URDF, scale=robot_scale)
    rover2 = load_model(TURTLEBOT_URDF, scale=robot_scale)
    print(get_aabb(rover2))
        
    robot_z = stable_z(rover, floor)
    set_point(rover, Point(z=robot_z))
    set_base_conf(rover, rover_conf)
    
    robot2_z = stable_z(rover2, floor)
    set_point(rover2, Point(z=robot2_z))
    set_base_conf(rover2, goal_conf)

    # wait_if_gui()
    remove_body(rover2)
    return RoversProblem(rover, limits=base_limits, body_types=body_types, goal_conf=goal_conf)


def random_obstacles(n_obstacles=50, robot_scale=1):
    base_extent = 5.0
    base_limits = (-base_extent/2.*np.ones(2), base_extent/2.*np.ones(2))
    mound_height = 0.1

    floor = create_box(base_extent, base_extent, 0.001, color=TAN) # TODO: two rooms
    set_point(floor, Point(z=-0.001/2.))

    wall1 = create_box(base_extent + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall1, Point(y=base_extent/2., z=mound_height/2.))
    wall2 = create_box(base_extent + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall2, Point(y=-base_extent/2., z=mound_height/2.))
    wall3 = create_box(mound_height, base_extent + mound_height, mound_height, color=GREY)
    set_point(wall3, Point(x=base_extent/2., z=mound_height/2.))
    wall4 = create_box(mound_height, base_extent + mound_height, mound_height, color=GREY)
    set_point(wall4, Point(x=-base_extent/2., z=mound_height/2.))

    body_types = []
    initial_surfaces = OrderedDict()
    for _ in range(n_obstacles):
        body = create_box(mound_height, mound_height, 4*mound_height, color=GREY)
        initial_surfaces[body] = floor

    rover_conf = (+1.75, -1.75, np.pi)
    goal_conf = (-1.75, 1.75, 0)
    
    rover = load_model(TURTLEBOT_URDF, scale=robot_scale)
    rover2 = load_model(TURTLEBOT_URDF, scale=2.0)
    print(get_aabb(rover2))
        
    robot_z = stable_z(rover, floor)
    set_point(rover, Point(z=robot_z))
    set_base_conf(rover, rover_conf)
    
    robot2_z = stable_z(rover2, floor)
    set_point(rover2, Point(z=robot2_z))
    set_base_conf(rover2, goal_conf)

    
    sample_placements(initial_surfaces, obstacles=[rover, rover2])
    remove_body(rover2)
    return RoversProblem(rover, limits=base_limits, body_types=body_types, goal_conf=goal_conf)
