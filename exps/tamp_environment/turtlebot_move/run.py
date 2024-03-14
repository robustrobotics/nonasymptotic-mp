import sys 

sys.path.extend(["./pddlstream/examples/pybullet/utils"])
sys.path.extend(["./pddlstream"])

from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.language.constants import And, print_solution, PDDLProblem
from pddlstream.language.stream import StreamInfo
from pddlstream.language.generator import from_fn
from pddlstream.utils import read, INF, get_file_path
from pybullet_tools.pr2_primitives import control_commands, apply_commands, State, Attach, Detach
from pybullet_tools.utils import connect, disconnect, has_gui, LockRenderer, WorldSaver, wait_if_gui, joint_from_name, get_pose
from streams import get_anytime_motion_fn, get_ik
from problems import hallway, BOT_RADIUS, hallway_manip, BASE_LINK
import random
import time
import numpy as np
import logging
import os
import copy


class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup_logging(save_dir):
    log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.FileHandler(os.path.join(save_dir, f"{time.time()}.log"))],
    )

    logger = logging.getLogger()

    # Add StreamHandler to logger to output logs to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Redirect stdout and stderr
    sys.stdout = StreamToLogger(logger, log_level)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    

def get_custom_limits(robot, base_limits, yaw_limit=None):
    x_limits, y_limits = zip(*base_limits)
    custom_limits = {
        joint_from_name(robot, 'x'): x_limits,
        joint_from_name(robot, 'y'): y_limits,
    }
    if yaw_limit is not None:
        custom_limits.update({
            joint_from_name(robot, 'theta'): yaw_limit,
        })
    return custom_limits

# Without pick place
# def pddlstream_from_problem(problem, collisions=True, mp_alg=None, max_samples=None, connect_radius=None, **kwargs):
#     # TODO: push and attach to movable objects

#     domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
#     stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
#     constant_map = {}

#     init = []
#     goal_literals = []
    
#     q0 = problem.init_conf
#     goal_conf = problem.goal_conf
    
#     init += [('Rover', problem.rover), ('Conf', problem.rover, q0), ('AtConf', problem.rover, q0), ("Conf", problem.rover, goal_conf)]
#     goal_literals += [('Holding', problem.targets[0])]
#     goal_formula = And(*goal_literals)

#     custom_limits = {}
#     if problem.limits is not None:
#         custom_limits.update(get_custom_limits(problem.rover, problem.limits))

#     stream_map = {
#         'sample-motion': from_fn(get_anytime_motion_fn(problem, custom_limits=custom_limits,
#                                                collisions=collisions, algorithm=mp_alg, 
#                                                start_samples=100, 
#                                                end_samples=max_samples,
#                                                factor=1.5,
#                                                connect_radius=connect_radius,
#                                                **kwargs)),
#     }

#     return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)


def pddlstream_from_problem(problem, min_samples=None, max_samples=None, factor=1.0, adaptive_n=False, **kwargs):
    # TODO: push and attach to movable objects

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    init = [('HFree',)]
    goal_literals = []
    
    q0 = problem.init_conf
    goal_conf = problem.goal_conf
    confs = [('Conf', problem.rover, q0), ('AtConf', problem.rover, q0)]

    target_poses = []
    goal_literals = []

    for target, target_init_pose, target_goal_pose in zip(problem.targets, problem.target_init_poses, problem.target_goal_poses):        
        target_poses+=[('Target', target), ('Pose', target, target_goal_pose), ('Pose', target, target_init_pose), ("AtPose", target, target_init_pose)]
        goal_literals.append(('AtPose', target, target_goal_pose))

    init += [('Rover', problem.rover)]+confs+target_poses

    # goal_literals = [('Holding', problem.targets[0])]
    goal_formula = And(*goal_literals)

    custom_limits = {}
    if problem.limits is not None:
        custom_limits.update(get_custom_limits(problem.rover, problem.limits))

    stream_map = {
        'sample-motion': from_fn(get_anytime_motion_fn(problem, custom_limits=custom_limits, start_samples=min_samples, end_samples=max_samples, factor=factor, adaptive_n=adaptive_n, **kwargs)),
        'sample-motion-holding': from_fn(get_anytime_motion_fn(problem, custom_limits=custom_limits, start_samples=min_samples, end_samples=max_samples, factor = factor, adaptive_n=adaptive_n, holding=True, **kwargs)),
        'sample-ik': from_fn(get_ik(problem)),
    }

    print("Init: "+str(init))
    print("Goal: "+str(goal_formula))
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)


def post_process(problem, plan):
    if plan is None:
        return None
    commands = []
    attachments = {}
    for i, (name, args) in enumerate(plan):
        if name == 'pick':
            v, q1, t, q2, p, o = args
            attachments[v] = o
            new_commands = [t, Attach(v, arm=BASE_LINK, grasp=None, body=attachments[v])]
        elif name == 'place':
            v, q1, t, q2, p, o = args
            new_commands = [t, Detach(v, arm=BASE_LINK, body=attachments[v])]
        else:
            raise ValueError(name)
        print(i, name, args, new_commands)
        commands += new_commands
    return commands

def main():
    # Unoptimized code: 300 takes 1:15
    parser = create_parser()
    parser.add_argument('--cfree', action='store_true', help='Disables collisions')
    parser.add_argument('--deterministic', action='store_true', help='Uses a deterministic sampler')
    parser.add_argument('--optimal', action='store_true', help='Runs in an anytime mode')
    parser.add_argument('--min-samples', default=10, type=int, help='Max num samples for motion planning')
    parser.add_argument('--max-samples', default=2000, type=int, help='Max num samples for motion planning')
    parser.add_argument('--factor', default=1.2, type=int, help='The rate at which we geometrically expand from min-samples to max-samples')
    parser.add_argument('--delta', default=0.1, type=float, help='Difference between the hallway width and the largest object that needs to fit thorugh the hallway')
    parser.add_argument('--seed', default=-1, type=int, help='Seed for selection of robot size and collision placement')
    parser.add_argument('--save-dir', default="./logs/debug", type=str, help='Directory to save planning results')
    parser.add_argument('--vis', action='store_true', help='GUI during planning')
    parser.add_argument('--teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('--randomize-delta', action='store_true', help='Teleports between configurations')
    parser.add_argument('--num-targets', type=float, default=5, help='Number of objects to carry across the hallway')
    parser.add_argument('--adaptive-n', action='store_true', help='Teleports between configurations')
    parser.add_argument('--simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()

    print("Experiment arguments:")
    print(vars(args))
    connect(use_gui=args.vis)
    
    save_dir = os.path.join(args.save_dir, str(time.time()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setup_logging(save_dir=save_dir)
    
    os.makedirs(os.path.join(save_dir, "pddl"))
    print('Arguments:', args)
    
    if(args.seed >= 0):
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    robot_scale = 0.3
    if(args.randomize_delta):
        delta = random.uniform(0.01, 0.2)
    else:
        delta = args.delta

    rovers_problem = hallway_manip(robot_scale=robot_scale, dd=delta, num_target=args.num_targets)

    max_samples = args.max_samples
    min_samples = args.min_samples
    
    print("Delta: "+str(delta))
    print("Min samples: "+str(min_samples))
    print("Max samples: "+str(max_samples))
    
    pddlstream_problem = pddlstream_from_problem(rovers_problem, collisions=not args.cfree, teleport=args.teleport,
                                                 holonomic=True, reversible=True, use_aabb=True, min_samples=min_samples, 
                                                 max_samples=max_samples, factor=args.factor, adaptive_n=args.adaptive_n)
    print(pddlstream_problem)
    stream_info = {
        'sample-motion': StreamInfo(overhead=10),
        'sample-ik': StreamInfo(overhead=10),
    }
    search_sample_ratio = 2
    max_planner_time = 10
    planner = 'ff-wastar3'
    success_cost = 0 if args.optimal else INF
    saver = WorldSaver()
    
    st = time.time()
    with LockRenderer(lock=True):
        solution = solve(pddlstream_problem, algorithm=args.algorithm, stream_info=stream_info,
                         planner=planner, max_planner_time=max_planner_time, debug=False,
                         unit_costs=args.unit, success_cost=success_cost,
                         max_time=np.inf, verbose=True,
                         unit_efforts=True, effort_weight=1,
                         search_sample_ratio=search_sample_ratio, 
                         temp_dir=os.path.join(save_dir, "pddl"))
    
    print_solution(solution)
    print("Time: "+str(time.time()-st))
    plan, cost, evaluations = solution
    if (plan is None) or not has_gui():
        return

    # Maybe OpenRAVE didn't actually sample any joints...
    # http://openrave.org/docs/0.8.2/openravepy/examples.tutorial_iksolutions/
    with LockRenderer():
        commands = post_process(rovers_problem, plan)
        saver.restore()

    wait_if_gui('Begin?')
    if args.simulate:
        control_commands(commands)
    else:
        time_step = None if args.teleport else 0.01
        apply_commands(State(), commands, time_step)
    wait_if_gui('Finish?')
    disconnect()



if __name__ == '__main__':
    main()
