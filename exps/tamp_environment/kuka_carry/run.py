
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
            #('Holding', body),
            #('On', body, fixed[1]),
            #('On', body, fixed[2]),
            #('Cleaned', body),
            ('Cooked', body),
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



def load_world():
    # TODO: store internal world info here to be reloaded
    pbu.set_default_camera()
    pbu.draw_global_system()
    with pbu.HideOutput():
        #add_data_path()
        robot = pbu.load_model(pbu.DRAKE_IIWA_URDF, fixed_base=True) # DRAKE_IIWA_URDF | KUKA_IIWA_URDF
        floor = pbu.load_model('models/short_floor.urdf')
        sink = pbu.load_model(pbu.SINK_URDF, pose=pbu.Pose(pbu.Point(x=-0.5)))
        stove = pbu.load_model(pbu.STOVE_URDF, pose=pbu.Pose(pbu.Point(x=+0.5)))
        celery = pbu.load_model(pbu.BLOCK_URDF, fixed_base=False)
        radish = pbu.load_model(pbu.SMALL_BLOCK_URDF, fixed_base=False)
        #cup = load_model('models/dinnerware/cup/cup_small.urdf',
        # Pose(Point(x=+0.5, y=+0.5, z=0.5)), fixed_base=False)

    pbu.draw_pose(pbu.Pose(), parent=robot, parent_link=get_tool_link(robot)) # TODO: not working
    # dump_body(robot)
    # wait_for_user()

    body_names = {
        sink: 'sink',
        stove: 'stove',
        celery: 'celery',
        radish: 'radish',
    }
    movable_bodies = [celery, radish]

    pbu.set_pose(celery, pbu.Pose(pbu.Point(y=0.5, z=pbu.stable_z(celery, floor))))
    pbu.set_pose(radish, pbu.Pose(pbu.Point(y=-0.5, z=pbu.stable_z(radish, floor))))

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

