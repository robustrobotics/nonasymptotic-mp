
import sys 

sys.path.extend(["./pddlstream/examples/pybullet/utils"])
sys.path.extend(["./pddlstream"])

import time
import pybullet_tools.utils as pbu
from kuka_primitives import TOOL_FRAMES, BodyPose, BodyConf, get_grasp_gen, \
    get_stable_gen, get_ik_fn, get_free_motion_gen, \
    get_holding_motion_gen, Command, get_fixed_stable_gen, BodyPath
from pddlstream.algorithms.meta import solve
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.utils import read, get_file_path, negate_test
from separating_axis import OBB, separating_axis_theorem
from pddlstream.language.constants import PDDLProblem, print_solution
import random
from typing import List
from nonasymptotic.bound import compute_numerical_bound
from nonasymptotic.prm import SimpleNearestNeighborRadiusPRM, SimplePRM
from nonasymptotic.envs import Environment
import os
import argparse
import pybullet as p
import copy
import string
import itertools
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import logging

DEFAULT_ARM_POS = (0, 0, 0, 0, 0, 0, 0)
Shape = namedtuple(
    "Shape", ["link", "index"]
)


class BagOfBoundingBoxes(Environment):

    def __init__(self, seed, start_oobb:pbu.OOBB, end_oobb:pbu.OOBB, obstacle_oobbs:List[pbu.OOBB]):
        self.obstacle_oobbs = obstacle_oobbs
        self.aabb = start_oobb.aabb
        all_oobb_verts = [pbu.get_oobb_vertices(oobb) for oobb in self.obstacle_oobbs]+[[start_oobb[1][0]]]
        self.bounds = self.extents(itertools.chain(*all_oobb_verts))
        print("Bounds: "+str(self.bounds))
        self.rng = np.random.default_rng(seed)

    def extents(self, vertices):
        # Unpack the vertices into separate lists for x, y, and z dimensions
        x_coords, y_coords, z_coords = zip(*vertices)

        # Calculate min and max along each dimension
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)
        rote = np.pi/32.0
        return ((min_x, max_x), (min_y, max_y), (min_z, max_z), (-rote, rote), (-rote, rote), (-rote, rote))

    def sample_from_env(self):
        return np.array([random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(len(self.bounds))])

    def mc_cfree_est(self):
        samples = np.stack([self.sample_from_env() for _ in range(10000)], axis=0)
        return np.mean(self.is_motion_valid(samples, samples).astype(int))
    

    def arclength_to_curve_point(self, t_normed):
        raise NotImplementedError

    def print_pose(self, pose):
        return "{}, {}".format(pose[0], pbu.euler_from_quat(pose[1]))
    
    def is_motion_valid(self, start, goal):
        valids = np.ones(start.shape[0]).astype(bool)
        obb_interps = []
        # print(start.shape[0])
        num_steps = int(np.max(np.linalg.norm(start-goal, axis=1))/0.03)
        # st = time.time()
        for i in range(start.shape[0]):
            start_pose = pbu.Pose(pbu.Point(*start[i, :3]), pbu.Euler(*start[i, 3:]))
            end_pose = pbu.Pose(pbu.Point(*goal[i, :3]), pbu.Euler(*goal[i, 3:]))
            interpolated = pbu.interpolate_poses(start_pose, end_pose, num_steps=num_steps+2)
            obb_interp = []
            for interp_pose in interpolated:
                oobb = pbu.OOBB(self.aabb, interp_pose)
                obb_interp.append(OBB.from_oobb(oobb).to_vectorized())
            obb_interps.append(obb_interp)
        # print(time.time()-st)
        interp_stacks = [np.stack([obb_interps[i][interp_idx] for i in range(start.shape[0])], axis=0) for interp_idx in range(len(obb_interps[0]))]
        # st = time.time()
        for obstacle_oobb in self.obstacle_oobbs:
            obstacle_stack = np.tile(OBB.from_oobb(obstacle_oobb).to_vectorized(), (start.shape[0], 1))
            for interp_stack in interp_stacks:
                valids = valids & ~separating_axis_theorem(obstacle_stack, interp_stack)
        # print(time.time()-st)

        return valids.astype(bool)

    def is_prm_epsilon_delta_complete(self, prm, tol):
        raise NotImplementedError

    def distance_to_path(self, points):
        return np.zeros((points.shape[0], ))

    @property
    def volume(self):
        raise NotImplementedError
    
def to_flat(pose):
    return list(pose[0])+list(pbu.euler_from_quat(pose[1]))

def to_pose(flat):
    return pbu.Pose(pbu.Point(*flat[:3]), pbu.Euler(flat[3:]))

def get_insert_motion_gen(robot, 
                          body_aabb_map = {},
                          body_obstacle_map = {},
                          teleport=False, 
                          start_samples=100, 
                          end_samples=1000, 
                          factor=1.5,
                          adaptive_n=False):
    
    def fn(conf1, conf2, body, grasp, fluents=[]):
        
        conf1.assign()
        grasp.attachment().assign()
        start_oobb = pbu.get_oobb(body)
        conf2.assign()
        grasp.attachment().assign()
        end_oobb = pbu.get_oobb(body)

        seed = 0
        obstacle_oobbs = body_obstacle_map[body]
        # Subtract one wall from another
        WALL_LINK_1 = 0
        WALL_LINK_2 = 2
        hallway_size = obstacle_oobbs[WALL_LINK_2][1][0][0]+obstacle_oobbs[WALL_LINK_2][0].upper[0] - \
            obstacle_oobbs[WALL_LINK_1][1][0][0]+obstacle_oobbs[WALL_LINK_1][0].lower[0]

        prm_env_2d = BagOfBoundingBoxes(seed=seed, start_oobb = start_oobb, end_oobb = end_oobb, obstacle_oobbs = obstacle_oobbs)
        cspace_volume = 1
        for bound in prm_env_2d.bounds:
            cspace_volume *= (bound[1]-bound[0])
        cfree_volume = cspace_volume

        print("Total volume: "+str(cspace_volume))
        print("Hallway size: "+str(hallway_size))
        collision_buffer = 0.01
        delta = hallway_size - ((body_aabb_map[body].upper[0] - body_aabb_map[body].lower[0]) + collision_buffer)/2.0
        print("[Inside MP] delta: "+str(delta))
        bound = compute_numerical_bound(delta, 0.99, cfree_volume, 6, None)
        print("Bound prediction: "+str(bound))
        if(adaptive_n):
            max_samples, _ = bound
            min_samples = max_samples-1
        else:
            min_samples=start_samples
            max_samples=end_samples
        
        print("[Inside MP] min_samples: "+str(min_samples))
        print("[Inside MP] max_samples: "+str(max_samples))


        prm = SimpleNearestNeighborRadiusPRM(32, 
                                     prm_env_2d.is_motion_valid, 
                                     prm_env_2d.sample_from_env, 
                                     prm_env_2d.distance_to_path, 
                                     seed=seed, verbose=False)
        start_vec = np.expand_dims(to_flat(start_oobb.pose), axis=0)
        end_vec = np.expand_dims(to_flat(end_oobb.pose), axis=0)
        assert prm_env_2d.is_motion_valid(start_vec, start_vec), "Start in collision"
        assert prm_env_2d.is_motion_valid(end_vec, end_vec), "Goal in collision"
        

        num_samples = min_samples
        while(num_samples<max_samples+1):
            print("Num samples: "+str(int(num_samples)))
            prm.grow_to_n_samples(int(num_samples))

            
            print('N nodes: %i' % prm.num_vertices())
            print('N edges: %i' % prm.num_edges())

            print("Start: "+str(start_vec))
            print("End: "+str(end_vec))
            _, path = prm.query_best_solution(start_vec, end_vec)

            if(len(path)>0):
                break
            num_samples = num_samples*factor
        
        
        print("Path: "+str(path))
        if(len(path) == 0):
            print("Max samples reached")
            return None
        else:
            print("Found solution in")
            print(num_samples)

        
        print("Enumerating path:")
        whole_path = start_vec.tolist()+path.tolist()+end_vec.tolist()

        # import matplotlib.pyplot as plt
        # plt.figure()

        # # plot the existing prm
        # xdim = 0
        # ydim = 2
        # for u, v in prm.prm_graph.iterEdges():
        #     coords_u = prm.prm_samples[u]
        #     coords_v = prm.prm_samples[v]
        #     plt.plot([coords_u[xdim], coords_v[xdim]], [coords_u[ydim], coords_v[ydim]], 'ro-')

        # for i in range(len(whole_path)-1):
        #     plt.plot([whole_path[i][xdim], whole_path[i+1][xdim]], [whole_path[i][ydim], whole_path[i+1][ydim]], 'go-')

        # plt.show()
        

        conf_path = []
        for el in whole_path:
            pose = pbu.Pose(pbu.Point(*el[:3]), pbu.Euler(*el[3:]))
            pbu.set_pose(body, pose)
            body_pose = BodyPose(body, pose)
            conf = get_ik_fn(robot, body, body_pose, grasp, randomize=False, teleport=teleport)
            if(conf is not None):
                conf_path.append(conf.values)
                conf_joints = conf.joints
                grasp.assign()

        command = Command(
            [BodyPath(robot, conf_path, joints=conf_joints, attachments=[grasp])]
        )
        return (command,)

    return fn

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
    

def create_hollow_shapes(indices, width=0.30, length=0.4, height=0.15, thickness=0.01):
    assert len(indices) == 3
    dims = [width, length, height]
    center = [0.0, 0.0, height / 2.0]
    coordinates = string.ascii_lowercase[-len(dims) :]

    # TODO: no way to programmatically set the name of the geoms or links
    # TODO: rigid links version of this
    shapes = []
    obstacle_oobbs = []
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
            link_aabb = pbu.AABB(lower=-np.array(geom["halfExtents"]), upper=np.array(geom["halfExtents"]))
            obstacle_oobbs.append(pbu.OOBB(link_aabb, pose))
    return shapes, obstacle_oobbs

def get_fixed(robot, movable):
    rigid = [body for body in pbu.get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed

def get_tool_link(robot):
    return pbu.link_from_name(robot, TOOL_FRAMES[pbu.get_body_name(robot)])


def pddlstream_from_problem(robot, names = {}, 
                            placement_links=[], 
                            movable=[], 
                            sink_obstacle_oobbs=[], 
                            fixed=[], 
                            teleport=False, 
                            min_samples=None, 
                            max_samples=None, 
                            factor=1.0, 
                            adaptive_n=False, 
                            grasp_name='top'):
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
        if(conf is None):
            return None
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
                

    # assert len(movable) == len(sinks)
    for body, sink in zip(movable, sinks):
        stable_gen = get_fixed_stable_gen(fixed, placement_links)
        grasp = body_to_grasp[body]
        preinsert_conf = None
        postinsert_conf = None
        for (postinsert,) in stable_gen(body, sink):
            preinsert = copy.deepcopy(postinsert)
            preinsert.pose = pbu.multiply(pbu.Pose(pbu.Point(x=0.0, z=0.12)), preinsert.pose)
            preinsert_conf = get_ik_fn(robot, body, preinsert, grasp, fixed=fixed, teleport=teleport)
            postinsert_conf = get_ik_fn(robot, body, postinsert, grasp, fixed=fixed, teleport=teleport)

            if(preinsert_conf is None or postinsert_conf is None):
                return None
            if(preinsert_conf is not None and postinsert_conf is not None):
                init += [('Pose', body, postinsert), 
                         ('Pose', body, preinsert), 
                         ("Supported", body, postinsert, sink),
                         ("Conf", preinsert_conf),
                         ("Conf", postinsert_conf),
                         ('Kin', body, preinsert, grasp, preinsert_conf),
                         ('Kin', body, postinsert, grasp, postinsert_conf),
                         ('PreInsert', preinsert_conf),
                         ('PostInsert', postinsert_conf)]
                
                break
        assert preinsert_conf is not None and postinsert_conf is not None


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

    body_aabb_map = {body: pbu.get_aabb(body) for body in movable}
    body_obstacle_map = {body: sink_oobbs for body, sink_oobbs in zip(movable, sink_obstacle_oobbs)}

    stream_map = {
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
        'plan-insert-motion': from_fn(get_insert_motion_gen(robot, 
                                                            body_aabb_map=body_aabb_map, 
                                                            body_obstacle_map = body_obstacle_map,
                                                            start_samples=min_samples, 
                                                            end_samples=max_samples, 
                                                            factor=factor, 
                                                            adaptive_n=adaptive_n,
                                                            teleport=teleport)),
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
    shapes, obstacle_oobbs = create_hollow_shapes(indices, *args, **kwargs)
    print(shapes)
    name, geoms, poses = zip(*shapes)
    colors = len(shapes) * [color]
    body = create_link_body(geoms, poses)
    return body, obstacle_oobbs


def load_world(min_gap = 0.06):

    pbu.set_default_camera()
    num_radish = 8
    radishes = []
    min_obj_size = 0.01
    max_obj_width = 0.06
    robot = pbu.load_model(pbu.DRAKE_IIWA_URDF, fixed_base=True)
    floor = pbu.load_model('models/short_floor.urdf')
    stove = pbu.load_model(pbu.STOVE_URDF, pose=pbu.Pose(pbu.Point(x=+0.5)))
    for i in range(num_radish):
        if(i>0):
            u = np.random.uniform(0, 1)
            obj_width = min_obj_size + (max_obj_width - min_obj_size) * (1 - (1 - u)**(1/4))
        else:
            obj_width = max_obj_width
        radishes.append(pbu.create_box(obj_width, obj_width, 0.2, color=pbu.GREEN))

    container_width = 0.06 + min_gap
    container_length = 0.06 + min_gap
    container_height = 0.15
    num_grid_x = 2
    num_grid_y = 4
    bin_grid_x = [i*container_width for i in range(num_grid_x)]
    bin_grid_y = [i*container_length for i in range(num_grid_y)]
    bin_center = (-0.68, -(container_length*(num_grid_y-1))/2.0)
    bins = []
    sink_obstacle_oobbs = []
    for bin_pos in itertools.product(bin_grid_x, bin_grid_y):
        new_bin, local_obstacle_oobbs = create_hollow("bin", width=container_width, length=container_length, height = container_height)
        
        bins.append(new_bin)
        bin_pose = pbu.Pose(pbu.Point(x=bin_pos[0]+bin_center[0], y=bin_pos[1]+bin_center[1]))
        obstacle_oobbs = []
        for local_oobb in local_obstacle_oobbs:
            # print(local_oobb[1])
            new_pose = pbu.multiply(bin_pose, local_oobb[1])
            new_oobb = pbu.OOBB(local_oobb[0], new_pose)
            obstacle_oobbs.append(new_oobb)
        sink_obstacle_oobbs.append(obstacle_oobbs)
        pbu.set_pose(new_bin, bin_pose)

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
    return robot, body_names, movable_bodies, sink_obstacle_oobbs, fixed, placement_links

def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name in ['move', 'move_free', 'move_holding', 'move_insert']:
            paths += args[-1].body_paths
    return Command(paths)

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
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-samples', default=500, type=int, help='Max num samples for motion planning')
    parser.add_argument('--max-samples', default=30000, type=int, help='Max num samples for motion planning')
    parser.add_argument('--factor', default=1.1, type=int, help='The rate at which we geometrically expand from min-samples to max-samples')
    parser.add_argument('--delta', default=0.04, type=float, help='Difference between the hallway width and the largest object that needs to fit thorugh the hallway')
    parser.add_argument('--seed', default=-1, type=int, help='Seed for selection of robot size and collision placement')
    parser.add_argument('--save-dir', default="./logs/debug", type=str, help='Directory to save planning results')
    parser.add_argument('--vis', action='store_true', help='GUI during planning')
    parser.add_argument('--teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('--randomize-delta', action='store_true', help='Teleports between configurations')
    parser.add_argument('--num-targets', type=float, default=5, help='Number of objects to carry across the hallway')
    parser.add_argument('--adaptive-n', action='store_true', help='Teleports between configurations')

    args = parser.parse_args()

    pbu.connect(use_gui=args.vis)

    save_dir = os.path.join(args.save_dir, str(time.time()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setup_logging(save_dir=save_dir)

    os.makedirs(os.path.join(save_dir, "pddl"))

    print("Experiment arguments:")
    print(vars(args))
    

    while(True):
        teleport = False
        robot, names, movable, sink_obstacle_oobbs, fixed, placement_links = load_world(min_gap=args.delta)
        print('Objects:', names)

        pbu.wait_if_gui()

        saver = pbu.WorldSaver()

        max_samples = args.max_samples
        min_samples = args.min_samples
        
        if(args.adaptive_n):
            min_samples = max_samples-1
        else:
            min_samples = args.min_samples


        print("Delta: "+str(args.delta))
        print("Min samples: "+str(min_samples))
        print("Max samples: "+str(max_samples))
        
        problem = pddlstream_from_problem(robot, names=names, min_samples=min_samples, 
                                        max_samples=max_samples, 
                                        factor=args.factor, 
                                        adaptive_n=args.adaptive_n, 
                                        placement_links=placement_links, 
                                        fixed=fixed, 
                                        movable=movable,
                                
                                        sink_obstacle_oobbs=sink_obstacle_oobbs, 
                                        teleport=teleport)
        
        if problem is not None: 
            break
        else:
            for body in pbu.get_bodies():
                pbu.remove_body(body)


    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', pbu.str_from_object(set(stream_map)))
    st = time.time()
    with pbu.Profiler():
        solution = solve(problem, algorithm="adaptive", unit_costs=False, verbose=True, success_cost=pbu.INF, temp_dir=os.path.join(save_dir, "pddl"))
        saver.restore()

    print_solution(solution)
    print("Time: "+str(time.time()-st))
    plan, cost, evaluations = solution
    if (plan is None) or not pbu.has_gui():
        p.disconnect()
        return

    pbu.set_joint_positions(robot, pbu.get_movable_joints(robot), DEFAULT_ARM_POS)
    command = postprocess_plan(plan)
    pbu.wait_for_user('Execute?')
    command.refine(num_steps=10).execute(time_step=0.001)
    pbu.wait_for_user('Finish?')

if __name__ == '__main__':
    main()

