
import sys 

sys.path.extend(["./pddlstream/examples/pybullet/utils"])
sys.path.extend(["./pddlstream"])

import time
import pybullet_tools.utils as pbu
from pybullet_tools.kuka_primitives import TOOL_FRAMES, BodyPose, BodyConf, get_grasp_gen, \
    get_stable_gen, get_ik_fn, get_free_motion_gen, \
    get_holding_motion_gen, Command, get_fixed_stable_gen
from pddlstream.algorithms.meta import solve
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.utils import read, get_file_path, negate_test
from pddlstream.language.constants import PDDLProblem
import os
import argparse
import pybullet as p
import string
import itertools
from collections import namedtuple
import numpy as np

DEFAULT_ARM_POS = (0, 0, 0, 0, 0, 0, 0)
Shape = namedtuple(
    "Shape", ["link", "index"]
)

def create_hollow_shapes(indices, width=0.30, length=0.4, height=0.15, thickness=0.01):
    assert len(indices) == 3
    dims = [width, length, height]
    center = [0.0, 0.0, height / 2.0]
    coordinates = string.ascii_lowercase[-len(dims) :]

    # TODO: no way to programmatically set the name of the geoms or links
    # TODO: rigid links version of this
    shapes = []
    for index, signs in enumerate(indices):
        link_dims = np.array(dims)
        link_dims[index] = thickness
        for sign in sorted(signs):
            # name = '{:+d}'.format(sign)
            name = "sink_{}".format(coordinates[index])
            geom = pbu.get_box_geometry(*link_dims)
            link_center = np.array(center)
            link_center[index] += sign * (dims[index] - thickness) / 2.0
            pose = pbu.Pose(point=link_center)
            shapes.append((name, geom, pose))
    return shapes

def get_fixed(robot, movable):
    rigid = [body for body in pbu.get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed

def get_tool_link(robot):
    return pbu.link_from_name(robot, TOOL_FRAMES[pbu.get_body_name(robot)])


def pddlstream_from_problem(robot, names = {}, placement_links=[], movable=[], fixed=[], teleport=False, grasp_name='top'):
    #assert (not are_colliding(tree, kin_cache))

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    print('Robot:', robot)
    pbu.set_joint_positions(robot, pbu.get_movable_joints(robot), DEFAULT_ARM_POS)
    start_conf = BodyConf(robot, pbu.get_configuration(robot))
    init = [('CanMove',),
            ('Conf', start_conf),
            ('AtConf', start_conf),
            ('HandEmpty',)]


    print('Movable:', movable)
    print('Fixed:', fixed)
    sinks = [surface for surface in fixed if 'sink' in names[surface]]
    grasp_gen = get_grasp_gen(robot, grasp_name)
    body_to_grasp = {}
    for body in movable:
        grasp, = next(grasp_gen(body))
        body_to_grasp[body] = grasp
        pose = BodyPose(body, pbu.get_pose(body))
        conf = get_ik_fn(robot, body, pose, grasp, fixed=fixed, teleport=teleport)
        init += [('Grasp', body, grasp),
                 ('Graspable', body),
                 ('Pose', body, pose),
                 ('AtPose', body, pose),
                 ('Conf', conf),
                 ('Kin', body, pose, grasp, conf)]
        
        for surface in fixed:
            name = names[surface]
            if 'sink' in name:
                init += [('Stackable', body, surface)]
                if pbu.is_placement(body, surface):
                    init += [('Supported', body, pose, surface)]
                

    assert len(movable) == len(sinks)
    for body, sink in zip(movable, sinks):
        stable_gen = get_fixed_stable_gen(fixed, placement_links)
        grasp = body_to_grasp[body]
        
        for (bp,) in stable_gen(body, sink):
            conf = get_ik_fn(robot, body, bp, grasp, fixed=fixed, teleport=teleport)
            if(conf is not None):
                init += [('Pose', body, bp), 
                         ("Supported", body, bp, sink),
                         ("Conf", conf),
                         ('Kin', body, bp, grasp, conf)]
                break
        assert conf is not None


    for body in fixed:
        name = names[body]
        if 'sink' in name:
            init += [('Sink', body)]
        if 'stove' in name:
            init += [('Stove', body)]

    goal = ['and',
            ('AtConf', start_conf),
    ]+[('Cleaned', body) for body in movable]
    # goal = ['not',
    #         ('HandEmpty', ),
    # ]


    stream_map = {
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
    }

    print("Init: "+str(init))

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

SHAPE_INDICES = {
    "tray": [{-1, +1}, {-1, +1}, {-1}],
    "bin": [{-1, +1}, {-1, +1}, {-1}],
    "cubby": [{-1}, {-1, +1}, {-1, +1}],
    "fence": [{-1, +1}, {-1, +1}, {}],
    "x_walls": [[-1, +1], [], [-1]],
    "y_walls": [[], [-1, +1], [-1]],
}

def create_link_body(geoms, poses, colors=None):
    """
    Create a body with multiple linked parts connected via revolute joints.
    :param geoms: List of half-extents for each part (geometries).
    :param poses: List of relative poses (positions/orientations) for each geometry.
    :param colors: List of colors for each geometry (optional, uses gray by default).
    :return: The ID of the base body.
    """
    # Ensure the input lists are valid
    assert len(geoms) == len(poses), "Number of geometries must match the number of poses."
    if colors is None:
        colors = [[0.5, 0.5, 0.5, 1.0]] * len(geoms)

    # Base properties (first geometry is the base)
    base_collision_shape = -1  # Indicating no collision shape
    base_visual_shape = -1  # Indicating no visual shape

    # Initialize the base position (logical root)
    # base_position = [0, 0, 0]
    # base_orientation = p.getQuaternionFromEuler([0, 0, 0])

    # Lists to hold properties for each link
    link_masses = []
    link_collision_shapes = []
    link_visual_shapes = []
    link_positions = []
    link_orientations = []
    link_joint_types = []
    link_joint_axes = []
    link_parent_indices = []

    # Fill out each link's properties
    for index in range(0, len(geoms)):
        link_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=geoms[index]['halfExtents'])
        link_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=geoms[index]['halfExtents'], rgbaColor=colors[index])
        link_collision_shapes.append(link_collision)
        link_visual_shapes.append(link_visual)
        link_positions.append(poses[index][0].tolist())  # Position of the link relative to its parent
        link_orientations.append(poses[index][1])  # Assuming default orientation
        link_masses.append(0.0)  # Set link mass
        link_joint_types.append(p.JOINT_FIXED)  # Joint type (e.g., revolute)
        link_joint_axes.append([0, 0, 1])  # Example axis, modify as needed
        link_parent_indices.append(0)  # Link parent index is the previous link

    # Create the entire articulated body with all links
    base_body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=base_collision_shape,
        baseVisualShapeIndex=base_visual_shape,
        basePosition = [0,0,0],  # Position of the base
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_shapes,
        linkVisualShapeIndices=link_visual_shapes,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=[[0, 0, 0]] * (len(geoms)),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * (len(geoms)),
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes
    )

    return base_body


def create_hollow(category, color=pbu.GREY, *args, **kwargs):
    indices = SHAPE_INDICES[category]
    shapes = create_hollow_shapes(indices, *args, **kwargs)
    print(shapes)
    name, geoms, poses = zip(*shapes)
    colors = len(shapes) * [color]
    body = create_link_body(geoms, poses)
    return body


def load_world():

    pbu.set_default_camera()
    # pbu.draw_global_system()
    num_radish = 3
    radishes = []
    
    with pbu.HideOutput():
        robot = pbu.load_model(pbu.DRAKE_IIWA_URDF, fixed_base=True)
        floor = pbu.load_model('models/short_floor.urdf')
        # sink = pbu.load_model(pbu.SINK_URDF, pose=pbu.Pose(pbu.Point(x=-0.5)))
        stove = pbu.load_model(pbu.STOVE_URDF, pose=pbu.Pose(pbu.Point(x=+0.5)))
        for _ in range(num_radish):
            radishes.append(pbu.load_model(pbu.BLOCK_URDF, fixed_base=False))

        container_width = 0.20
        container_length = 0.20
        container_height = 0.15
        num_grid_x = 1
        num_grid_y = 3
        bin_grid_x = [i*container_width for i in range(num_grid_x)]
        bin_grid_y = [i*container_length for i in range(num_grid_y)]
        bin_center = (-0.68, -container_width/2.0)
        bins = []
        for bin_pos in itertools.product(bin_grid_x, bin_grid_y):
            new_bin = create_hollow("bin", width=container_width, length=container_length, height = container_height)
            bins.append(new_bin)
            pbu.set_pose(new_bin, pbu.Pose(pbu.Point(x=bin_pos[0]+bin_center[0], y=bin_pos[1]+bin_center[1])))

    # pbu.draw_pose(pbu.Pose(), parent=robot, parent_link=get_tool_link(robot))

    body_names = {bini: 'sink_'+str(bini) for bini in bins} | {
        
        stove: 'stove',
        
        floor: 'floor'
    } | {radish: 'radish_'+str(radish) for radish in radishes}
    movable_bodies = radishes
    placement_links = {body: None if "sink" not in name else 4 for body, name in body_names.items()}

    # print(placement_links)
    fixed = [stove]+bins
    

    placed = []
    
    for radish in radishes:
        while(True):
            grasp_gen = get_grasp_gen(robot, "top")
            stable_gen = get_stable_gen(fixed, placement_links)
            pose, = next(stable_gen(radish, stove, obstacles=placed)) 
            pbu.set_pose(radish, pose.value)
            grasp, = next(grasp_gen(radish))
            conf =  get_ik_fn(robot, radish, pose, grasp, fixed=fixed, teleport=False)
            if(conf is not None):
                placed.append(radish)
                break
            




   
    return robot, body_names, movable_bodies, fixed, placement_links

def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        
        if name in ['move', 'move_free', 'move_holding']:
            paths += args[-1].body_paths
    return Command(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', default="./logs/debug", type=str, help='Directory to save planning results')
    args = parser.parse_args()
    
    pbu.connect(use_gui=True)
    teleport = False
    robot, names, movable, fixed, placement_links = load_world()
    print('Objects:', names)

    pbu.wait_if_gui()

    saver = pbu.WorldSaver()

    problem = pddlstream_from_problem(robot, names=names, placement_links=placement_links, fixed=fixed, movable=movable, teleport=teleport)

    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', pbu.str_from_object(set(stream_map)))
    save_dir = os.path.join(args.save_dir, str(time.time()))
    os.makedirs(os.path.join(save_dir, "pddl"))
    with pbu.Profiler():
        solution = solve(problem, algorithm="adaptive", unit_costs=False, success_cost=pbu.INF, temp_dir=os.path.join(save_dir, "pddl"))
        saver.restore()

    plan, cost, evaluations = solution
    if (plan is None) or not pbu.has_gui():
        pbu.disconnect()
        return

    pbu.set_joint_positions(robot, pbu.get_movable_joints(robot), DEFAULT_ARM_POS)
    command = postprocess_plan(plan)
    pbu.wait_for_user('Execute?')
    command.refine(num_steps=10).execute(time_step=0.001)
    pbu.wait_for_user('Finish?')
    pbu.disconnect()


if __name__ == '__main__':
    main()

