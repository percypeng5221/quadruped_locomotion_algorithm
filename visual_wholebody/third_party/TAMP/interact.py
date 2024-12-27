"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


# set random seed
np.random.seed(40)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.05

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "/home/percy/princeton/visual_wholebody/third_party/isaacgym/assets"

# create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create box asset
box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# load franka asset
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

# use position drive for all dofs
if controller == "ik":
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][:7].fill(400.0)
    franka_dof_props["damping"][:7].fill(40.0)
else:       # osc
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    franka_dof_props["stiffness"][:7].fill(0.0)
    franka_dof_props["damping"][:7].fill(0.0)
    
# grippers
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]

# grippers open
default_dof_pos[7:] = franka_upper_limits[7:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["panda_hand"]

# configure env grid
num_envs = 52 # first: target(blue)    second: current(red)    rest: planning(green)
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

box_pose = gymapi.Transform()

envs = []
box_idxs = []

hand_idxs = []
init_pos_list = []
init_rot_list = []
num_boxs = 3

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# target box positions and rotations
target_positions_rotations = [0.2, 0.1, -math.pi/4]
target_mass = 0.5
box_x = []
box_y = []
box_quat = []
mass_range = [[0.2, 1], [2.2, 5.0], [0.8, 3.8], ]


# create env
for env_i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    
    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", env_i, 0)

    # add box
    tmp_box_idxs = []
    
    
    # target env generation
    if env_i == 0:
        color = gymapi.Vec3(0, 0, 1)
        box_pose.p.x = table_pose.p.x + target_positions_rotations[0]
        box_pose.p.y = table_pose.p.y + target_positions_rotations[1]
        box_pose.p.z = table_dims.z + 0.5 * box_size
        box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), target_positions_rotations[2])
        box_handle = gym.create_actor(env, box_asset, box_pose, "box", env_i, 0)
        # get global index of box in rigid body state tensor
        box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
        
        # randomize mass
        body_props = gym.get_actor_rigid_body_properties(env, box_handle)
        for b in range(len(body_props)):
            body_props[b].mass = target_mass
        gym.set_actor_rigid_body_properties(env, box_handle, body_props)
        tmp_box_idxs.append(box_idx)
        
    # current real env generation
    elif env_i == 1:
        color = gymapi.Vec3(1, 0, 0)
        for box_i in range(num_boxs):
            box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
            box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
            box_pose.p.z = table_dims.z + 0.5 * box_size
            box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            box_handle = gym.create_actor(env, box_asset, box_pose, "box", env_i, 0)
            # get global index of box in rigid body state tensor
            box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
            
            box_x.append(box_pose.p.x)
            box_y.append(box_pose.p.y)
            box_quat.append(box_pose.r)
            
            # randomize mass
            body_props = gym.get_actor_rigid_body_properties(env, box_handle)

            body_props[0].mass = np.random.uniform(mass_range[box_i][0], mass_range[box_i][1])
            gym.set_actor_rigid_body_properties(env, box_handle, body_props)
            tmp_box_idxs.append(box_idx)
            
    # env for planning
    else:
        color = gymapi.Vec3(0, 1, 0)    
        for box_i in range(num_boxs):
            box_pose.p.x = box_x[box_i]
            box_pose.p.y = box_y[box_i]
            box_pose.p.z = table_dims.z + 0.5 * box_size
            box_pose.r = box_quat[box_i]
            box_handle = gym.create_actor(env, box_asset, box_pose, "box", env_i, 0)
            # get global index of box in rigid body state tensor
            box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
            
            # randomize mass
            body_props = gym.get_actor_rigid_body_properties(env, box_handle)
            body_props[0].mass = np.random.uniform(mass_range[box_i][0], mass_range[box_i][1])
            gym.set_actor_rigid_body_properties(env, box_handle, body_props)
            tmp_box_idxs.append(box_idx)
    
    box_idxs.append(tmp_box_idxs)
    
    # set color for different envs
    gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    
    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", env_i, 2)
    # gym.enable_actor_dof_force_sensors(env, franka_handle)
    
    # set dof properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[1]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare simulation =====
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(device)

# box corner coords, used to determine grasping yaw
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to franka hand
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)
dof_vel = dof_states[:, 1].view(num_envs, 9, 1)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

# initialize tensors
holding_object = torch.zeros(num_envs, dtype=torch.bool).to(device)  # Track whether each environment is holding an object
action_status = torch.zeros(num_envs).to(device) # 0: move to object, 1: pick, 2: move to place target, 3: place, 4: move to initial 

goal_pos = torch.empty((num_envs, 3), device=device)
goal_rot = torch.empty((num_envs, 4), device=device)
close_gripper = torch.zeros(num_envs, dtype=torch.bool).to(device)

success_rate = torch.zeros(num_envs, dtype=torch.bool).to(device)

# freeze_counter
freeze_counter = torch.zeros(num_envs).to(device)
freeze_time = int(0.5 / sim_params.dt)

# Generate random positions and rotations in tensor format
target_6d_pos = []
counter = -1

# which object to generate
choice_tensor = torch.randint(0, 3, (num_envs, ))
choice_tensor[0] = 0
best_choice = 0

# simulation loop
while not gym.query_viewer_has_closed(viewer):
    counter += 1
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    # # get box states
    # for sublist, choice in zip(box_idxs, choice_tensor): 
    #     print("sublist, choice.item()", sublist, choice.item())
    #     print("sublist[choice.item()]", sublist[choice.item()])
        
    box_pos = rb_states[[sublist[choice.item()] for sublist, choice in zip(box_idxs, choice_tensor)], :3]
    box_rot = rb_states[[sublist[choice.item()] for sublist, choice in zip(box_idxs, choice_tensor)], 3:7]

    box_pos[1] = rb_states[box_idxs[1][best_choice], :3]
    box_rot[1] = rb_states[box_idxs[1][best_choice], 3:7]
    
    target_box_pos = rb_states[[*box_idxs[0]], :3]
    current_box_pos = rb_states[[*box_idxs[1]], :3]
    target_hand_pos = target_box_pos + torch.tensor([[0, 0, 0.1]]).to(device)
    
    target_box_rot = rb_states[[*box_idxs[0]], 3:7]
    current_box_rot = rb_states[[*box_idxs[1]], 3:7]
    target_hand_rot = target_box_rot
    
    # print("target_box_pos, target_box_rot: ", target_box_pos, target_box_rot)
    # print("current_box_pos, current_box_rot: ", current_box_pos, current_box_rot)

    # while True: continue
    # get end effector states
    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]

    to_box = box_pos - hand_pos
    to_box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / to_box_dist
    box_dot = box_dir @ down_dir.view(3, 1)
    
    for env_i in range(num_envs):
        box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
        box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
        box_pose.p.z = table_dims.z + 0.5 * box_size
        box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))

    # how far the hand should be from box for grasping
    grasp_offset = 0.1 

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
    gripped = (gripper_sep < 0.045) & (to_box_dist < grasp_offset + 0.5 * box_size)

               
    yaw_q = cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = quat_axis(yaw_q, 0)
    hand_yaw_dir = quat_axis(hand_rot, 0)
    yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    to_init_dist = torch.norm(to_init, dim=-1)
    
    to_place_pos = target_hand_pos - hand_pos
    to_place_dist = torch.norm(to_place_pos, dim=-1)
    
    # object dist
    obj_to_place_pos = target_box_pos - box_pos
    obj_to_place_dist = torch.norm(obj_to_place_pos, dim=-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (to_box_dist < grasp_offset * 3)).squeeze(-1)
    grasp_pos = box_pos.clone()
    grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)

    # action status change
    for env_i in range(num_envs):
        if env_i == 0:
            action_status[env_i] = 4
            continue
        
        if action_status[env_i] == 0 and to_box_dist[env_i] < grasp_offset + 0.2 * box_size:
            if freeze_counter[env_i] <= 0:
                action_status[env_i] = 1
                freeze_counter[env_i] = freeze_time
            else: freeze_counter[env_i] -= 1
            
        elif action_status[env_i] == 1 and gripped[env_i]:
            if freeze_counter[env_i] <= 0:
                action_status[env_i] = 2
                freeze_counter[env_i] = freeze_time
            else: freeze_counter[env_i] -= 1
            
        elif action_status[env_i] == 2 and to_place_dist[env_i] < 0.05:
            if freeze_counter[env_i] <= 0:
                action_status[env_i] = 3
                freeze_counter[env_i] = freeze_time
            else: freeze_counter[env_i] -= 1
            
        elif action_status[env_i] == 3 and not gripped[env_i]:
            if freeze_counter[env_i] <= 0:
                action_status[env_i] = 4
                freeze_counter[env_i] = freeze_time
            else: freeze_counter[env_i] -= 1
            
        elif action_status[env_i] == 4 and to_init_dist[env_i] < 0.02:
            action_status[env_i] = 4
            

    
    for env_i in range(num_envs):
        if env_i == 1 and torch.mean(action_status[2:]) < 4:
            goal_pos[env_i] = init_pos[env_i]
            goal_rot[env_i] = init_rot[env_i]
            close_gripper[env_i] = False
            continue
        
        if action_status[env_i] == 0:
            goal_pos[env_i] = grasp_pos[env_i]
            goal_rot[env_i] = quat_mul(down_q, quat_conjugate(yaw_q[env_i]))
            close_gripper[env_i] = False
        
        elif action_status[env_i] == 1:
            close_gripper[env_i] = True
            
        elif action_status[env_i] == 2:
            goal_pos[env_i] = target_hand_pos[0]
            goal_rot[env_i] = quat_mul(down_q, quat_conjugate(target_hand_rot[0]))
            close_gripper[env_i] = True
        
        elif action_status[env_i] == 3:
            close_gripper[env_i] = False
            
        elif action_status[env_i] == 4:
            goal_pos[env_i] = init_pos[env_i]
            goal_rot[env_i] = init_rot[env_i]
            close_gripper[env_i] = False
            
    # print("action_status, to_place_dist, box_dist: ", action_status, to_place_dist, box_dist)
    
    # goal_pos = torch.where(return_to_start, init_pos, position)
    # goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(quat)))
    
    # print("position, goal_pos, quat, goal_rot", position, goal_pos, quat, goal_rot)
    # print("type(position), type(goal_pos), type(quat), type(goal_rot)", type(position), type(goal_pos), type(quat), type(goal_rot))
    # compute position and orientation error
    pos_err = (goal_pos - hand_pos) 
    # print("pos_err: ", pos_err)
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1) / 2

    # Calculate control
    joint_dpose = control_ik(dpose)
    pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + joint_dpose
    
    # Calculate success rate for each choice in choice_tensor
    success_rate = torch.where(success_rate, True, torch.where(obj_to_place_dist < 0.05, True, False))
    success_rates = {}
    for choice in range(3):
        # Mask for the current choice
        choice_mask = (choice_tensor[2:] == choice)
        
        # Filter success_rate by the mask for the current choice
        success_rate_for_choice = success_rate[2:][choice_mask]
        
        # Calculate the success rate percentage for the current choice
        if success_rate_for_choice.numel() > 0:  # Avoid division by zero
            total_success_rate_for_choice = success_rate_for_choice.float().mean().item() * 100
        else:
            total_success_rate_for_choice = 0  # No entries for this choice
            
        # Store the success rate in the dictionary
        success_rates[choice] = total_success_rate_for_choice
        print(f"Success rate for choice {choice}: {total_success_rate_for_choice:.2f}% (Total values: {success_rate_for_choice.numel()})")
    
    # Determine the choice with the highest success rate
    best_choice = max(success_rates, key=success_rates.get)
    highest_success_rate = success_rates[best_choice]
    print(f"Choice with the highest success rate: {best_choice} ({highest_success_rate:.2f}%)")
    # print(f"\rChoice with the highest success rate: {best_choice} ({highest_success_rate:.2f}%)", end="")

    grip_acts = torch.where(close_gripper.unsqueeze(1), torch.Tensor([[0., 0.]] * num_envs).to(device), torch.Tensor([[0.05, 0.05]] * num_envs).to(device))
    pos_action[:, 7:9] = grip_acts

    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
