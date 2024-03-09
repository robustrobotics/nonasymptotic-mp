import sys 

sys.path.extend(["./pddlstream/examples/pybullet/utils"])
sys.path.extend(["./pddlstream"])

from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.language.constants import And, print_solution, PDDLProblem
from pddlstream.language.stream import StreamInfo
from pddlstream.language.generator import from_fn
from pddlstream.utils import read, INF, get_file_path
from pybullet_tools.pr2_primitives import control_commands, apply_commands, State
from pybullet_tools.utils import connect, disconnect, has_gui, LockRenderer, WorldSaver, wait_if_gui, joint_from_name
from streams import get_nonasy_motion_fn, get_anytime_motion_fn
from nonasymptotic.util import compute_numerical_bound
from problems import hallway, BOT_RADIUS
import random
import time
import numpy as np
import logging
import os


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

def pddlstream_from_problem(problem, collisions=True, mp_alg=None, max_samples=None, connect_radius=None, **kwargs):
    # TODO: push and attach to movable objects

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    init = []
    goal_literals = []
    
    q0 = problem.init_conf
    goal_conf = problem.goal_conf
    
    init += [('Rover', problem.rover), ('Conf', problem.rover, q0), ('AtConf', problem.rover, q0), ("Conf", problem.rover, goal_conf)]
    goal_literals += [('AtConf', problem.rover, goal_conf)]
    goal_formula = And(*goal_literals)

    custom_limits = {}
    if problem.limits is not None:
        custom_limits.update(get_custom_limits(problem.rover, problem.limits))

    stream_map = {
        'sample-motion': from_fn(get_anytime_motion_fn(problem, custom_limits=custom_limits,
                                               collisions=collisions, algorithm=mp_alg, 
                                               num_samples=max_samples,
                                               connect_radius=connect_radius,
                                               **kwargs)),
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)


def post_process(problem, plan):
    if plan is None:
        return None
    commands = []
    attachments = {}
    for i, (name, args) in enumerate(plan):
        if name == 'move':
            v, q1, t, q2 = args
            new_commands = [t]
        else:
            raise ValueError(name)
        print(i, name, args, new_commands)
        commands += new_commands
    return commands

def main():
    parser = create_parser()
    parser.add_argument('-cfree', action='store_true', help='Disables collisions')
    parser.add_argument('-deterministic', action='store_true', help='Uses a deterministic sampler')
    parser.add_argument('-optimal', action='store_true', help='Runs in an anytime mode')
    parser.add_argument('-t', '--max-time', default=240, type=int, help='The max time')
    parser.add_argument('-ms', '--max-samples', default=30000, type=int, help='Max num samples for motion planning')
    parser.add_argument('-d', '--delta', default=0.2, type=int, help='Max num samples for motion planning')
    parser.add_argument('-mp_alg', '--mp-alg', default="prm", type=str, help='Algorithm to use for motion planning')
    parser.add_argument('-seed', '--seed', default=-1, type=int, help='Seed for selection of robot size and collision placement')
    parser.add_argument('-sd', '--save-dir', default="./logs/debug", type=str, help='Directory to save planning results')
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('--vis', action='store_true', help='GUI during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('--randomize-delta', action='store_true', help='Teleports between configurations')
    parser.add_argument('--adaptive-n', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()

    connect(use_gui=args.vis)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    setup_logging(save_dir=args.save_dir)
    
    print('Arguments:', args)
    
    if(args.seed >= 0):
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    robot_scale = 0.5
    if(args.randomize_delta):
        delta = args.delta
    else:
        delta = random.uniform(0.01, 0.2)

    rovers_problem = hallway(robot_scale=robot_scale, dd=delta)

    max_samples = args.max_samples
    connect_radius = 0.1
    
    if(args.adaptive_n):
        max_samples, connect_radius = compute_numerical_bound(delta, 0.9, 4, 2, None)
    
    print("Delta: ")
    print(delta)
    print("Max samples: "+str(max_samples))
    print("Connection radius: "+str(connect_radius))
    
    pddlstream_problem = pddlstream_from_problem(rovers_problem, collisions=not args.cfree, teleport=args.teleport,
                                                 holonomic=True, reversible=True, use_aabb=True, max_samples=max_samples, 
                                                 connect_radius=connect_radius,
                                                 mp_alg=args.mp_alg)
    print(pddlstream_problem)
    stream_info = {
        'sample-motion': StreamInfo(overhead=10),
    }
    search_sample_ratio = 2
    max_planner_time = 10
    planner = 'ff-wastar3'
    success_cost = 0 if args.optimal else INF
    saver = WorldSaver()
    
    st = time.time()
    with LockRenderer(lock=not args.enable):
        solution = solve(pddlstream_problem, algorithm=args.algorithm, stream_info=stream_info,
                                planner=planner, max_planner_time=max_planner_time, debug=False,
                                unit_costs=args.unit, success_cost=success_cost,
                                max_time=args.max_time, verbose=True,
                                unit_efforts=True, effort_weight=1,
                                search_sample_ratio=search_sample_ratio)
    
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
