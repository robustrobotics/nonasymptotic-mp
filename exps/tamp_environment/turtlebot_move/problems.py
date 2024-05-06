import numpy as np

from collections import OrderedDict
from pybullet_tools.utils import set_point, Point, create_box, \
    stable_z, load_model, TURTLEBOT_URDF, joints_from_names, \
    set_joint_positions, get_joint_positions, remove_body, \
    GREY, TAN, YELLOW, get_bodies, pairwise_collision, \
    wait_if_gui, get_pose, Pose, create_cylinder, flatten_links, \
    get_moving_links, get_aabb, Euler, set_pose, AABB, multiply, \
    sample_aabb, get_point, get_center_extent, aabb_empty, unit_pose, \
    get_aabb_extent, aabb_from_extent_center
    
from typing import List, Tuple
from dataclasses import dataclass, field
import random

BOT_RADIUS = 0.179

@dataclass
class ObjectPose:
    pose: Pose

@dataclass
class Conf:
    
    body: int
    joints: List[int]
    values: List[float] = None
    init: bool = False
    
    def __post_init__(self):
        if(self.values is None):
            get_joint_positions(self.body, self.joints)

    @property
    def bodies(self):  # TODO: misnomer
        return flatten_links(self.body, get_moving_links(self.body, self.joints))

    def assign(self):
        set_joint_positions(self.body, self.joints, self.values)

    def iterate(self):
        yield self

    def __repr__(self):
        return "q{}".format(id(self) % 1000)


def sample_placement_on_aabb(
    top_body,
    bottom_aabb,
    top_pose=unit_pose(),
    percent=1.0,
    max_attempts=50,
    epsilon=1e-3,
    **kwargs
):
    # TODO: transform into the coordinate system of the bottom
    # TODO: maybe I should instead just require that already in correct frame
    for _ in range(max_attempts):
        theta = 0
        rotation = Euler(yaw=theta)
        set_pose(top_body, multiply(Pose(euler=rotation), top_pose), **kwargs)
        center, extent = get_center_extent(top_body, **kwargs)
        lower = (np.array(bottom_aabb[0]) + percent * extent / 2)[
            :2
        ]  # TODO: scale_aabb
        upper = (np.array(bottom_aabb[1]) - percent * extent / 2)[:2]
        aabb = AABB(lower, upper)
        if aabb_empty(aabb):
            continue
        x, y = sample_aabb(aabb)
        z = (bottom_aabb[1] + extent / 2.0)[2] + epsilon
        point = np.array([x, y, z]) + (get_point(top_body, **kwargs) - center)
        pose = multiply(Pose(point, rotation), top_pose)
        set_pose(top_body, pose, **kwargs)
        return pose
    return None


def sample_placement(top_body, bottom_body, bottom_link=None, **kwargs):
    bottom_aabb = get_aabb(bottom_body, link=bottom_link, **kwargs)
    return sample_placement_on_aabb(top_body, bottom_aabb, **kwargs)


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
BASE_LINK = 'base_link'

def get_base_joints(robot):
    return joints_from_names(robot, BASE_JOINTS)

def get_base_conf(robot):
    return get_joint_positions(robot, get_base_joints(robot))

def set_base_conf(robot, conf):
    set_joint_positions(robot, get_base_joints(robot), conf)

#######################################################

@dataclass
class RoversProblem():
    rover:int = None
    obstacles:List[int] = field(default_factory=lambda: [])
    limits:List[Tuple[float]] = field(default_factory=lambda: [])
    targets:List[int] = field(default_factory=lambda: []) 
    target_init_poses:List[ObjectPose] = field(default_factory=lambda: []) 
    target_goal_poses:List[ObjectPose] = field(default_factory=lambda: []) 
    target_sizes:List[float] = field(default_factory=lambda: [])
    init_conf:Conf = None
    goal_conf:Conf = None
    hallway_gap:float=0

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

    return RoversProblem(rover, limits=base_limits,init_conf=init_conf, goal_conf=goal_conf, obstacles=obstacles)


def hallway_manip(robot_scale=0.2, dd=0.1, num_target=1):
    base_extent = 3.0
    base_limits = (-base_extent/2.*np.ones(2), base_extent/2.*np.ones(2))
    mound_height = 0.1

    room_length = 1.5  # The length of each side of the square rooms
    hallway_length = 3  # The length of the hallway
    wall_thickness = mound_height  # The thickness of the walls
    wall_height = mound_height  # The height of the walls
    
    # Make each obstacle have a size between the robot size and the hallway witdh, 
    # with one of the objects being only dd smaller than the hallway
    assert num_target > 0
    max_target_diff = 0.1
    min_object_size = robot_scale * BOT_RADIUS

    max_object_size = min_object_size+max_target_diff
    targets = [create_cylinder(max_object_size, 0.1, color=YELLOW)]
    target_sizes = {targets[0]: max_object_size}
    
    for _ in range(num_target-1):
        u = np.random.uniform(0, 1)
        target_radius = min_object_size + (max_object_size - min_object_size) * (1 - (1 - u)**(1/4))
        target = create_cylinder(target_radius, 0.1, color=YELLOW)
        target_sizes[target] = target_radius
        targets.append(target)
    
    hallway_gap = max(list(target_sizes.values()))*2 + dd  
    hallway_width = hallway_gap + wall_thickness # The width of the hallway
    
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

    rover = load_model(TURTLEBOT_URDF, scale=robot_scale)
    
    base_joints = get_base_joints(rover)

    body_surfaces = {target: room1_floor for target in targets}
    sample_placements(body_surfaces, obstacles=obstacles)
    target_goal_poses = [ObjectPose(get_pose(target)) for target in targets]
        
    body_surfaces = {target: room2_floor for target in targets}
    sample_placements(body_surfaces, obstacles=obstacles)
    target_init_poses = [ObjectPose(get_pose(target)) for target in targets]

    for target in targets:
        print(target_sizes[target])
        print(aabb_from_extent_center(get_aabb_extent(get_aabb(target))))

    body_surfaces = {}
    init_conf = Conf(rover, base_joints[:2], (-hallway_length/2 - room_length/2, room_length/2))
    # goal_conf =  Conf(rover, base_joints[:2], (hallway_length/2 + room_length/2, room_length/2))
    robot_z = stable_z(rover, room1_floor)
    set_point(rover, Point(z=robot_z))
    init_conf.assign()

    wait_if_gui()
    return RoversProblem(rover, limits=base_limits, 
                         init_conf=init_conf, 
                         obstacles=obstacles, 
                         targets=targets, 
                         target_init_poses=target_init_poses,
                         target_goal_poses=target_goal_poses,
                         target_sizes=target_sizes,
                         hallway_gap=hallway_gap)

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
  
    rover = load_model(TURTLEBOT_URDF, scale=robot_scale)
    
    base_joints = get_base_joints(rover)

    
    init_conf = Conf(rover, base_joints[:2], (-hallway_length/2 - room_length/2, room_length/2))
    goal_conf =  Conf(rover, base_joints[:2], (hallway_length/2 + room_length/2, room_length/2))
    robot_z = stable_z(rover, room1_floor)
    set_point(rover, Point(z=robot_z))
    init_conf.assign()

    wait_if_gui()
    return RoversProblem(rover, limits=base_limits, init_conf=init_conf, goal_conf=goal_conf, obstacles=obstacles)


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

    return RoversProblem(rover, limits=base_limits, init_conf=init_conf, goal_conf=goal_conf, obstacles=obstacles)


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
    return RoversProblem(rover, limits=base_limits, init_conf=init_conf, goal_conf=goal_conf)
