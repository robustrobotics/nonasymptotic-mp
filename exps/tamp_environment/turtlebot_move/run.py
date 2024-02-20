import sys 

sys.path.extend(["./pddlstream/examples/pybullet/utils"])
sys.path.extend(["./pddlstream"])

from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.language.constants import And, print_solution, PDDLProblem
from pddlstream.language.stream import StreamInfo
from pddlstream.language.generator import from_fn
from pddlstream.utils import read, INF, get_file_path
from pybullet_tools.pr2_primitives import Conf, control_commands, apply_commands, State
from pybullet_tools.utils import connect, disconnect, has_gui, LockRenderer, WorldSaver, wait_for_user, joint_from_name
from streams import get_motion_fn, get_base_joints
from problems import rovers1

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

def pddlstream_from_problem(problem, collisions=True, **kwargs):
    # TODO: push and attach to movable objects

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    init = []
    goal_literals = []
    base_joints = get_base_joints(problem.rover)
    
    q0 = Conf(problem.rover, base_joints)
    goal_conf = Conf(problem.rover, base_joints, values=problem.goal_conf)
    
    init += [('Rover', problem.rover), ('Conf', problem.rover, q0), ('AtConf', problem.rover, q0), ("Conf", problem.rover, goal_conf)]
    goal_literals += [('AtConf', problem.rover, goal_conf)]
    goal_formula = And(*goal_literals)

    custom_limits = {}
    if problem.limits is not None:
        custom_limits.update(get_custom_limits(problem.rover, problem.limits))

    stream_map = {
        'sample-motion': from_fn(get_motion_fn(problem, custom_limits=custom_limits,
                                               collisions=collisions, **kwargs)),
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
    parser.add_argument('-t', '--max_time', default=120, type=int, help='The max time')
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()
    connect(use_gui=True)

    print('Arguments:', args)
    rovers_problem = rovers1()
    pddlstream_problem = pddlstream_from_problem(rovers_problem, collisions=not args.cfree, teleport=args.teleport,
                                                holonomic=False, reversible=True, use_aabb=True)
    print(pddlstream_problem)
    stream_info = {
        'sample-motion': StreamInfo(overhead=10),
    }
    search_sample_ratio = 2
    max_planner_time = 10
    planner = 'ff-wastar3'
    success_cost = 0 if args.optimal else INF
    saver = WorldSaver()
    with LockRenderer(lock=not args.enable):
        solution = solve(pddlstream_problem, algorithm=args.algorithm, stream_info=stream_info,
                                planner=planner, max_planner_time=max_planner_time, debug=False,
                                unit_costs=args.unit, success_cost=success_cost,
                                max_time=args.max_time, verbose=True,
                                unit_efforts=True, effort_weight=1,
                                search_sample_ratio=search_sample_ratio)
    
    print_solution(solution)
    
    plan, cost, evaluations = solution
    if (plan is None) or not has_gui():
        disconnect()
        return

    # Maybe OpenRAVE didn't actually sample any joints...
    # http://openrave.org/docs/0.8.2/openravepy/examples.tutorial_iksolutions/
    with LockRenderer():
        commands = post_process(rovers_problem, plan)
        saver.restore()

    wait_for_user('Begin?')
    if args.simulate:
        control_commands(commands)
    else:
        time_step = None if args.teleport else 0.01
        apply_commands(State(), commands, time_step)
    wait_for_user('Finish?')
    disconnect()



if __name__ == '__main__':
    main()
