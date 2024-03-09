import numpy as np

from collections import OrderedDict
from pybullet_tools.utils import set_point, Point, create_box, \
    stable_z, load_model, TURTLEBOT_URDF, joints_from_names, \
    set_joint_positions, get_joint_positions, remove_body, \
    GREY, TAN, get_bodies, pairwise_collision, sample_placement, wait_if_gui
from pybullet_tools.pr2_primitives import Conf
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
    def __init__(self, rover=None, obstacles=[], limits=[], body_types=[], init_conf=None, goal_conf=None):
        self.rover = rover
        self.init_conf = init_conf
        self.goal_conf = goal_conf
        self.limits = limits
        self.obstacles=obstacles
        self.body_types = body_types
        self.costs = False

#######################################################
def hallway(robot_scale=0.2, dd=0.1):
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

    obstacles = [wall1, wall2, wall3, wall4]

    body_types = []
    initial_surfaces = OrderedDict()
    
    robot_width = BOT_RADIUS*robot_scale
    offset = robot_width + mound_height
    
    
    box_width = (base_extent/2.0-(mound_height+2*robot_width))*2-dd
    body = create_box(box_width, box_width, mound_height*4, color=GREY)
    initial_surfaces[body] = floor

    obstacles.append(body)
    
    rover = load_model(TURTLEBOT_URDF, scale=robot_scale)
    
    base_joints = get_base_joints(rover)
    init_conf = Conf(rover, base_joints[:2], (base_extent/2.0-offset, -base_extent/2.0+offset))
    goal_conf =  Conf(rover, base_joints[:2], (-base_extent/2.0+offset, base_extent/2.0-offset))
    robot_z = stable_z(rover, floor)
    set_point(rover, Point(z=robot_z))
    init_conf.assign()

    return RoversProblem(rover, limits=base_limits, body_types=body_types, init_conf=init_conf, goal_conf=goal_conf, obstacles=obstacles)

def hallway(robot_scale=0.2, dd=0.1):
    base_extent = 3.0
    base_limits = (-base_extent/2.*np.ones(2), base_extent/2.*np.ones(2))
    mound_height = 0.1

    room_length = 1  # The length of each side of the square rooms
    hallway_length = 3  # The length of the hallway
    wall_thickness = mound_height  # The thickness of the walls
    wall_height = mound_height  # The height of the walls
    hallway_width = robot_scale * BOT_RADIUS * 2 + wall_thickness + dd  # The width of the hallway

    # Walls for Room 1
    room1_front_wall = create_box(room_length+mound_height, wall_thickness, wall_height, color=GREY)
    set_point(room1_front_wall, Point(x=-hallway_length/2 - room_length/2, y=0, z=wall_height/2))

    room1_back_wall = create_box(room_length+mound_height, wall_thickness, wall_height, color=GREY)
    set_point(room1_back_wall, Point(x=-hallway_length/2 - room_length/2, y=room_length, z=wall_height/2))

    room1_left_wall = create_box(wall_thickness, room_length+mound_height, wall_height, color=GREY)
    set_point(room1_left_wall, Point(x=-hallway_length/2 - room_length - wall_thickness/2, y=room_length/2, z=wall_height/2))

    flap_width = (room_length - hallway_width) / 2

    # Room 1 Right Wall Upper Flap
    room1_flap1 = create_box(wall_thickness, flap_width+mound_height, wall_height, color=GREY)
    set_point(room1_flap1, Point(x=-hallway_length/2, y=room_length/2 + hallway_width/2 + flap_width/2, z=wall_height/2))

    # Room 1 Right Wall Lower Flap
    room1_flap2 = create_box(wall_thickness, flap_width+mound_height, wall_height, color=GREY)
    set_point(room1_flap2, Point(x=-hallway_length/2, y=room_length/2 - hallway_width/2 - flap_width/2, z=wall_height/2))

    # Walls for Room 2 (mirroring Room 1 with respect to the origin)
    room2_front_wall = create_box(room_length+mound_height, wall_thickness, wall_height, color=GREY)
    set_point(room2_front_wall, Point(x=hallway_length/2 + room_length/2, y=room_length, z=wall_height/2))

    room2_back_wall = create_box(room_length+mound_height, wall_thickness, wall_height, color=GREY)
    set_point(room2_back_wall, Point(x=hallway_length/2 + room_length/2, y=0, z=wall_height/2))

    # Flap 1 (Left side of the opening)
    room2_flap1 = create_box(wall_thickness, flap_width+mound_height, wall_height, color=GREY)
    set_point(room2_flap1, Point(x=hallway_length/2, y=flap_width/2, z=wall_height/2))

    # Flap 2 (Right side of the opening)
    room2_flap2 = create_box(wall_thickness, flap_width+mound_height, wall_height, color=GREY)
    set_point(room2_flap2, Point(x=hallway_length/2, y=room_length - flap_width/2, z=wall_height/2))


    room2_right_wall = create_box(wall_thickness, room_length+mound_height, wall_height, color=GREY)
    set_point(room2_right_wall, Point(x=hallway_length/2 + room_length + wall_thickness/2, y=room_length/2, z=wall_height/2))

    # Hallway Walls
    hallway_top_wall = create_box(hallway_length, wall_thickness, wall_height, color=GREY)
    set_point(hallway_top_wall, Point(y=room_length/2.0+hallway_width/2, z=wall_height/2))

    hallway_bottom_wall = create_box(hallway_length, wall_thickness, wall_height, color=GREY)
    set_point(hallway_bottom_wall, Point(y=room_length/2.0-hallway_width/2, z=wall_height/2))


    obstacles = [
        room1_front_wall, room1_back_wall, room1_left_wall, room1_flap1, room2_flap1,
        room2_front_wall, room2_back_wall, room2_right_wall, room1_flap2, room2_flap2, #
        hallway_top_wall, hallway_bottom_wall
    ]

    # Room 1 Floor
    room1_floor = create_box(room_length, room_length, 0.001, color=TAN)
    set_point(room1_floor, Point(x=-hallway_length/2 - room_length/2, y=room_length/2, z=-0.001/2))

    # Hallway Floor
    hallway_floor = create_box(hallway_length, hallway_width, 0.001, color=TAN)
    set_point(hallway_floor, Point(x=0, y=room_length/2.0, z=-0.001/2))

    # Room 2 Floor
    room2_floor = create_box(room_length, room_length, 0.001, color=TAN)
    set_point(room2_floor, Point(x=hallway_length/2 + room_length/2, y=room_length/2, z=-0.001/2))

    # Collecting the floors in a list for further operations if needed
    floors = [room1_floor, hallway_floor, room2_floor]

    body_types = []       
    rover = load_model(TURTLEBOT_URDF, scale=robot_scale)
    
    base_joints = get_base_joints(rover)

    

    init_conf = Conf(rover, base_joints[:2], (-hallway_length/2 - room_length/2, room_length/2))
    goal_conf =  Conf(rover, base_joints[:2], (hallway_length/2 + room_length/2, room_length/2))
    robot_z = stable_z(rover, room1_floor)
    set_point(rover, Point(z=robot_z))
    init_conf.assign()

    wait_if_gui()
    return RoversProblem(rover, limits=base_limits, body_types=body_types, init_conf=init_conf, goal_conf=goal_conf, obstacles=obstacles)


def corner(robot_scale=0.2, dd=0.1):
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

    obstacles = [wall1, wall2, wall3, wall4]

    body_types = []
    initial_surfaces = OrderedDict()
    
    robot_width = BOT_RADIUS*robot_scale
    offset = robot_width + mound_height
    
    
    box_width = (base_extent/2.0-(mound_height+2*robot_width))*2-dd
    body = create_box(box_width, box_width, mound_height*4, color=GREY)
    initial_surfaces[body] = floor

    obstacles.append(body)
    
    rover = load_model(TURTLEBOT_URDF, scale=robot_scale)
    
    base_joints = get_base_joints(rover)
    init_conf = Conf(rover, base_joints[:2], (base_extent/2.0-offset, -base_extent/2.0+offset))
    goal_conf =  Conf(rover, base_joints[:2], (-base_extent/2.0+offset, base_extent/2.0-offset))
    robot_z = stable_z(rover, floor)
    set_point(rover, Point(z=robot_z))
    init_conf.assign()

    return RoversProblem(rover, limits=base_limits, body_types=body_types, init_conf=init_conf, goal_conf=goal_conf, obstacles=obstacles)


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

    
    rover = load_model(TURTLEBOT_URDF, scale=robot_scale)
    rover2 = load_model(TURTLEBOT_URDF, scale=2.0)
    
    base_joints = get_base_joints(rover)
    init_conf = Conf(rover, base_joints, (+1.75, -1.75, np.pi))
    goal_conf =  Conf(rover, base_joints, (-1.75, 1.75, 0))
    
    
    robot_z = stable_z(rover, floor)
    set_point(rover, Point(z=robot_z))
    set_base_conf(rover, init_conf.values)
    
    robot2_z = stable_z(rover2, floor)
    set_point(rover2, Point(z=robot2_z))
    set_base_conf(rover2, goal_conf.values)

    sample_placements(initial_surfaces, obstacles=[rover, rover2])
    remove_body(rover2)
    return RoversProblem(rover, limits=base_limits, body_types=body_types, init_conf=init_conf, goal_conf=goal_conf)
