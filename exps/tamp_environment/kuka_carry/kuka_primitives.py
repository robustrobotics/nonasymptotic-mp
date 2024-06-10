import time
from itertools import count

from pybullet_tools.pr2_utils import get_top_grasps, TOOL_POSE, GRASP_LENGTH, MAX_GRASP_WIDTH
from pybullet_tools.utils import (
    INF,
    Attachment,
    GraspInfo,
    Point,
    Pose,
    add_fixed_constraint,
    approach_from_grasp,
    disable_real_time,
    dump_world,
    enable_gravity,
    enable_real_time,
    end_effector_from_body,
    flatten,
    get_body_name,
    get_joint_positions,
    get_movable_joints,
    get_pose,
    get_refine_fn,
    get_sample_fn,
    inverse_kinematics,
    joint_controller,
    link_from_name,
    pairwise_collision,
    plan_direct_joint_motion,
    plan_joint_motion,
    refine_path,
    remove_fixed_constraint,
    sample_placement,
    set_joint_positions,
    set_pose,
    step_simulation,
    wait_for_duration,
    wait_if_gui,
    get_center_extent,
    get_aabb,
    get_link_pose,
    quat_from_euler,
    Euler,
    multiply,
    point_from_pose,
    approximate_as_prism,
    unit_pose
)
import math
import numpy as np
import random
# TODO: deprecate

def get_top_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                   max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH):
    # TODO: rename the box grasps
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, h / 2 - grasp_length])
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    grasps = []
    if w <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, math.pi / 2 + i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    if l <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    return grasps

GRASP_INFO = {
    "top": GraspInfo(
        lambda body: get_top_grasps(
            body, under=True, tool_pose=Pose(), max_width=INF, grasp_length=0
        ),
        approach_pose=Pose(0.01 * Point(z=1)),
    ),
}

TOOL_FRAMES = {
    "iiwa14": "iiwa_link_ee_kuka",  # iiwa_link_ee | iiwa_link_ee_kuka
}

DEBUG_FAILURE = False

##################################################


class BodyPose(object):
    num = count()

    def __init__(self, body, pose=None):
        if pose is None:
            pose = get_pose(body)
        self.body = body
        self.pose = pose
        self.index = next(self.num)

    @property
    def value(self):
        return self.pose

    def assign(self):
        set_pose(self.body, self.pose)
        return self.pose

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "p{}".format(index)


class BodyGrasp(object):
    num = count()

    def __init__(self, body, grasp_pose, approach_pose, robot, link):
        self.body = body
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
        self.robot = robot
        self.link = link
        self.index = next(self.num)

    @property
    def value(self):
        return self.grasp_pose

    @property
    def approach(self):
        return self.approach_pose

    # def constraint(self):
    #    grasp_constraint()
    def attachment(self):
        return Attachment(self.robot, self.link, self.grasp_pose, self.body)

    def assign(self):
        return self.attachment().assign()

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "g{}".format(index)


class BodyConf(object):
    num = count()

    def __init__(self, body, configuration=None, joints=None):
        if joints is None:
            joints = get_movable_joints(body)
        if configuration is None:
            configuration = get_joint_positions(body, joints)
        self.body = body
        self.joints = joints
        self.configuration = configuration
        self.index = next(self.num)

    @property
    def values(self):
        return self.configuration

    def assign(self):
        set_joint_positions(self.body, self.joints, self.configuration)
        return self.configuration

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "q{}".format(index)


class BodyPath(object):
    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = get_movable_joints(body)
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments

    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])

    def iterator(self):
        # TODO: compute and cache these
        # TODO: compute bounding boxes as well
        for i, configuration in enumerate(self.path):
            set_joint_positions(self.body, self.joints, configuration)
            for grasp in self.attachments:
                grasp.assign()
            yield i

    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        for values in self.path:
            for _ in joint_controller(self.body, self.joints, values):
                enable_gravity()
                if not real_time:
                    step_simulation()
                time.sleep(dt)

    # def full_path(self, q0=None):
    #     # TODO: could produce sequence of savers
    def refine(self, num_steps=0):
        return self.__class__(
            self.body,
            refine_path(self.body, self.joints, self.path, num_steps),
            self.joints,
            self.attachments,
        )

    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.joints, self.attachments)

    def __repr__(self):
        return "{}({},{},{},{})".format(
            self.__class__.__name__,
            self.body,
            len(self.joints),
            len(self.path),
            len(self.attachments),
        )


##################################################


class ApplyForce(object):
    def __init__(self, body, robot, link):
        self.body = body
        self.robot = robot
        self.link = link

    def bodies(self):
        return {self.body, self.robot}

    def iterator(self, **kwargs):
        return []

    def refine(self, **kwargs):
        return self

    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self.robot, self.body)


class Attach(ApplyForce):
    def control(self, **kwargs):
        # TODO: store the constraint_id?
        add_fixed_constraint(self.body, self.robot, self.link)

    def reverse(self):
        return Detach(self.body, self.robot, self.link)


class Detach(ApplyForce):
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)

    def reverse(self):
        return Attach(self.body, self.robot, self.link)


class Command(object):
    num = count()

    def __init__(self, body_paths):
        self.body_paths = body_paths
        self.index = next(self.num)

    def bodies(self):
        return set(flatten(path.bodies() for path in self.body_paths))

    # def full_path(self, q0=None):
    #     if q0 is None:
    #         q0 = Conf(self.tree)
    #     new_path = [q0]
    #     for partial_path in self.body_paths:
    #         new_path += partial_path.full_path(new_path[-1])[1:]
    #     return new_path
    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = "{},{}) step?".format(i, j)
                wait_if_gui(msg)
                # print(msg)

    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                # time.sleep(time_step)
                wait_for_duration(time_step)

    def control(self, real_time=False, dt=0):  # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)

    def refine(self, **kwargs):
        return self.__class__(
            [body_path.refine(**kwargs) for body_path in self.body_paths]
        )

    def reverse(self):
        return self.__class__(
            [body_path.reverse() for body_path in reversed(self.body_paths)]
        )

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "c{}".format(index)


#######################################################


def get_tool_link(robot):
    return link_from_name(robot, TOOL_FRAMES[get_body_name(robot)])


def get_grasp_gen(robot, grasp_name="top"):
    grasp_info = GRASP_INFO[grasp_name]
    tool_link = get_tool_link(robot)

    def gen(body):
        grasp_poses = grasp_info.get_grasps(body)
        # TODO: continuous set of grasps
        for grasp_pose in grasp_poses:
            body_grasp = BodyGrasp(
                body, grasp_pose, grasp_info.approach_pose, robot, tool_link
            )
            yield (body_grasp,)

    return gen


def get_fixed_stable_gen(fixed=[], links={}):
    def gen(body, surface, obstacles=[]):
        for theta in [0, np.pi/2.0, np.pi, -np.pi/2.0]:
            pose = sample_placement(body, surface, bottom_link = links[surface], theta=theta, scale=0)
            if (pose is None) or any(pairwise_collision(body, b) for b in fixed+obstacles):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)

    return gen

def get_stable_gen(fixed=[], links={}):
    def gen(body, surface, obstacles=[]):
        while(True):
            theta = random.uniform(-np.pi, np.pi)
            pose = sample_placement(body, surface, bottom_link = links[surface], theta=theta)
            if (pose is None) or any(pairwise_collision(body, b) for b in fixed+obstacles):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)

    return gen


def get_ik_fn(robot, body, pose, grasp, fixed=[], randomize=True, teleport=False, num_attempts=10):
    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)
    obstacles = fixed
    gripper_pose = end_effector_from_body(pose.pose, grasp.grasp_pose)
    approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)
    for i in range(num_attempts):
        if(randomize):
            set_joint_positions(robot, movable_joints, sample_fn())  # Random seed
        q_approach = inverse_kinematics(robot, grasp.link, approach_pose)
        if (q_approach is None) or any(
            pairwise_collision(robot, b) for b in obstacles
        ):
            continue
        
        # q_grasp = inverse_kinematics(robot, grasp.link, gripper_pose)
        conf = BodyConf(robot, q_approach)
        # if (q_grasp is None) or any(
        #     pairwise_collision(robot, b) for b in obstacles
        # ):
        #     continue
        
        return conf
        # TODO: holding collisions
    return None




##################################################


def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == "atpose":
            o, p = args
            obstacles.append(o)
            p.assign()
        else:
            raise ValueError(name)
    return obstacles


def get_free_motion_gen(robot, fixed=[], teleport=False, self_collisions=True):
    def fn(conf1, conf2, fluents=[]):
        assert (conf1.body == conf2.body) and (conf1.joints == conf2.joints)
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            obstacles = fixed + assign_fluent_state(fluents)
            path = plan_joint_motion(
                robot,
                conf2.joints,
                conf2.configuration,
                obstacles=obstacles,
                self_collisions=self_collisions,
            )
            if path is None:
                if DEBUG_FAILURE:
                    wait_if_gui("Free motion failed")
                return None
        command = Command([BodyPath(robot, path, joints=conf2.joints)])
        return (command,)

    return fn


def get_holding_motion_gen(robot, fixed=[], teleport=False, self_collisions=True):
    def fn(conf1, conf2, body, grasp, fluents=[]):
        assert (conf1.body == conf2.body) and (conf1.joints == conf2.joints)
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            obstacles = fixed + assign_fluent_state(fluents)
            path = plan_joint_motion(
                robot,
                conf2.joints,
                conf2.configuration,
                obstacles=obstacles,
                attachments=[grasp.attachment()],
                self_collisions=self_collisions,
            )
            if path is None:
                if DEBUG_FAILURE:
                    wait_if_gui("Holding motion failed")
                return None
        command = Command(
            [BodyPath(robot, path, joints=conf2.joints, attachments=[grasp])]
        )
        return (command,)

    return fn


##################################################


def get_movable_collision_test():
    def test(command, body, pose):
        if body in command.bodies():
            return False
        pose.assign()
        for path in command.body_paths:
            moving = path.bodies()
            if body in moving:
                # TODO: cannot collide with itself
                continue
            for _ in path.iterator():
                # TODO: could shuffle this
                if any(pairwise_collision(mov, body) for mov in moving):
                    if DEBUG_FAILURE:
                        wait_if_gui("Movable collision")
                    return True
        return False

    return test