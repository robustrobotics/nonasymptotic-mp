
import sys 

sys.path.extend(["./pddlstream/examples/pybullet/utils"])
sys.path.extend(["./pddlstream"])

import time
import pybullet_tools.utils as pbu
from pybullet_tools.kuka_primitives import TOOL_FRAMES, BodyPose, BodyConf, get_grasp_gen, \
    get_stable_gen, get_ik_fn, get_free_motion_gen, \
    get_holding_motion_gen, get_movable_collision_test, get_tool_link, Command
from pddlstream.algorithms.meta import solve
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.utils import read, get_file_path, negate_test
from pddlstream.language.constants import PDDLProblem
from streams import get_cfree_pose_pose_test, \
    get_cfree_obj_approach_pose_test
import os
import argparse
import string
import itertools
from collections import namedtuple
import numpy as np

# DEFAULT_ARM_POS = (-2.74228529839567, -1.1180049615399599, 2.2107771723948684, -1.320262400722078, 0.9131962195838295, 1.1464814897121636, 0.009226410633654493)
Shape = namedtuple(
    "Shape", ["link", "index"]
)

def create_hollow_shapes(indices, width=0.30, length=0.4, height=0.2, thickness=0.01):
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
            name = "{}{}".format("-" if sign < 0 else "+", coordinates[index])
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


def pddlstream_from_problem(robot, movable=[], teleport=False, grasp_name='top'):
    #assert (not are_colliding(tree, kin_cache))

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    print('Robot:', robot)
    conf = BodyConf(robot, pbu.get_configuration(robot))
    init = [('CanMove',),
            ('Conf', conf),
            ('AtConf', conf),
            ('HandEmpty',)]

    fixed = get_fixed(robot, movable)
    print('Movable:', movable)
    print('Fixed:', fixed)
    for body in movable:
        pose = BodyPose(body, pbu.get_pose(body))
        init += [('Graspable', body),
                 ('Pose', body, pose),
                 ('AtPose', body, pose)]
        for surface in fixed:
            init += [('Stackable', body, surface)]
            if pbu.is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    for body in fixed:
        name = pbu.get_body_name(body)
        if 'sink' in name:
            init += [('Sink', body)]
        if 'stove' in name:
            init += [('Stove', body)]

    body = movable[0]
    goal = ('and',
            ('AtConf', conf),
            ('Cleaned', body),
    )

    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(fixed)),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
        'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test()),
        'test-cfree-approach-pose': from_test(get_cfree_obj_approach_pose_test()),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test())), #get_cfree_traj_pose_test()),

        'TrajCollision': get_movable_collision_test(),
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

SHAPE_INDICES = {
    "tray": [{-1, +1}, {-1, +1}, {-1}],
    "bin": [{-1, +1}, {-1, +1}, {-1}],
    "cubby": [{-1}, {-1, +1}, {-1, +1}],
    "fence": [{-1, +1}, {-1, +1}, {}],
    "x_walls": [[-1, +1], [], [-1]],
    "y_walls": [[], [-1, +1], [-1]],
}

def create_hollow(category, color=pbu.GREY, *args, **kwargs):
    indices = SHAPE_INDICES[category]
    shapes = create_hollow_shapes(indices, *args, **kwargs)
    _, geoms, poses = zip(*shapes)
    colors = len(shapes) * [color]
    collision_id, visual_id = pbu.create_shape_array(geoms, poses, colors)
    body = pbu.create_body(collision_id, visual_id, mass=pbu.STATIC_MASS)
    return body


def load_world():

    pbu.set_default_camera()
    pbu.draw_global_system()
    with pbu.HideOutput():
        robot = pbu.load_model(pbu.DRAKE_IIWA_URDF, fixed_base=True)
        # pbu.set_joint_positions(robot, pbu.get_movable_joints(robot), DEFAULT_ARM_POS)
        floor = pbu.load_model('models/short_floor.urdf')
        # sink = pbu.load_model(pbu.SINK_URDF, pose=pbu.Pose(pbu.Point(x=-0.5)))
        stove = pbu.load_model(pbu.STOVE_URDF, pose=pbu.Pose(pbu.Point(x=+0.5)))
        radish = pbu.load_model(pbu.BLOCK_URDF, fixed_base=False)
        container_width = 0.3
        container_length = 0.3
        grid_size_x = 4
        grid_size_y = 4
        bin_grid_x = list(np.linspace(0, container_width, grid_size_x))
        bin_grid_y = list(np.linspace(0, container_width, grid_size_y))
        bin_width = container_width/float(grid_size_x)+0.03
        bin_length = container_length/float(grid_size_y)+0.03
        bin_center = (-0.65, -0.15)
        bins = []
        for bin_pos in itertools.product(bin_grid_x, bin_grid_y):
            print(bin_pos)
            new_bin = create_hollow("bin", width=bin_width, length=bin_length)
            bins.append(new_bin)
            pbu.set_pose(new_bin, pbu.Pose(pbu.Point(x=bin_pos[0]+bin_center[0], y=bin_pos[1]+bin_center[1])))

    pbu.draw_pose(pbu.Pose(), parent=robot, parent_link=get_tool_link(robot))


    body_names = {
        new_bin: 'sink',
        stove: 'stove',
        radish: 'radish',
    }
    movable_bodies = [radish]

    fixed = get_fixed(robot, movable_bodies)
    stable_gen = get_stable_gen(fixed)
    pose, = next(stable_gen(radish, stove))
    pbu.set_pose(radish, pose.value)
    return robot, body_names, movable_bodies

def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name == 'place':
            paths += args[-1].reverse().body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick']:
            paths += args[-1].body_paths
    return Command(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', default="./logs/debug", type=str, help='Directory to save planning results')
    args = parser.parse_args()
    
    pbu.connect(use_gui=True)
    teleport = False
    robot, names, movable = load_world()
    print('Objects:', names)

    pbu.wait_if_gui()

    saver = pbu.WorldSaver()

    problem = pddlstream_from_problem(robot, movable=movable, teleport=teleport)
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

    command = postprocess_plan(plan)
    pbu.wait_for_user('Execute?')
    command.refine(num_steps=10).execute(time_step=0.001)
    pbu.wait_for_user('Finish?')
    pbu.disconnect()


if __name__ == '__main__':
    main()

