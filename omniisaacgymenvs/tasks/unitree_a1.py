from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.unitree_a1 import UnitreeA1
from omniisaacgymenvs.robots.articulations.views.unitree_a1_view import UnitreeA1View
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from omniisaacgymenvs import RL4Robot_omnisim_DATA_DIR

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.utils.torch.rotations import *

import numpy as np
import torch
import math
import random
import os

import pprint
pp = pprint.PrettyPrinter()

class UnitreeA1StandTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # work mode
        self.is_sample_init_state = self._task_cfg["env"]["work_mode"]["is_sample_init_state"]
        self.dummy_action = self._task_cfg["env"]["work_mode"]["dummy_action"]

        # normalization
        self.clip_observations = self._task_cfg["env"]["normalization"]["clip_observations"]
        self.clip_actions = self._task_cfg["env"]["normalization"]["clip_actions"]
        self.lin_vel_scale = self._task_cfg["env"]["normalization"]["obs_scales"]["lin_vel"]
        self.ang_vel_scale = self._task_cfg["env"]["normalization"]["obs_scales"]["ang_vel"]
        self.dof_pos_scale = self._task_cfg["env"]["normalization"]["obs_scales"]["dof_pos"]
        self.dof_vel_scale = self._task_cfg["env"]["normalization"]["obs_scales"]["dof_vel"]

        self.dof_pos_limits = self._task_cfg["env"]["normalization"]["dof_pos_limits"]
        self.dof_vel_limits = self._task_cfg["env"]["normalization"]["dof_vel_limits"]
        self.dof_torque_limits = self._task_cfg["env"]["normalization"]["dof_torque_limits"]

        # noise
        self.add_noise = self._task_cfg["env"]["noise"]["add_noise"]
        self.noise_level = self._task_cfg["env"]["noise"]["noise_level"]
        self.lin_vel_noise_scale = self._task_cfg["env"]["noise"]["noise_scales"]["lin_vel"]
        self.ang_vel_noise_scale = self._task_cfg["env"]["noise"]["noise_scales"]["ang_vel"]
        self.dof_pos_noise_scale = self._task_cfg["env"]["noise"]["noise_scales"]["dof_pos"]
        self.dof_vel_noise_scale = self._task_cfg["env"]["noise"]["noise_scales"]["dof_vel"]
        self.gravity_noise_scale = self._task_cfg["env"]["noise"]["noise_scales"]["gravity"]

        # reward scales and limit
        self.rew_scales = {}
        for reward_name, reward_scale in self._task_cfg["env"]["rewards"]["scales"].items():
            self.rew_scales[reward_name] = reward_scale
        self.soft_dof_pos_limit = self._task_cfg["env"]["rewards"]["soft_dof_pos_limit"]
        self.soft_dof_vel_limit = self._task_cfg["env"]["rewards"]["soft_dof_vel_limit"]
        self.soft_torque_limit = self._task_cfg["env"]["rewards"]["soft_torque_limit"]
        self.stand_still_sigma = self._task_cfg["env"]["rewards"]["stand_still_sigma"]
        self.tracking_sigma = self._task_cfg["env"]["rewards"]["tracking_sigma"]

        # command ranges
        self.command_yaw_range = self._task_cfg["env"]["commands"]["ranges"]["ang_vel_yaw"]
        # self.command_heading_range = self._task_cfg["env"]["commands"]["ranges"]["heading"]
        self.command_x_range = self._task_cfg["env"]["commands"]["ranges"]["lin_vel_x"]
        self.command_y_range = self._task_cfg["env"]["commands"]["ranges"]["lin_vel_y"]

        # init state
        pos = self._task_cfg["env"]["init_state"]["pos"]
        rot = self._task_cfg["env"]["init_state"]["rot"]
        v_lin = self._task_cfg["env"]["init_state"]["lin_vel"]
        v_ang = self._task_cfg["env"]["init_state"]["ang_vel"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.default_dof_poses = self._task_cfg["env"]["init_state"]["default_dof_poses"] # default_joint_angles
        self.down_dof_angles = self._task_cfg["env"]["init_state"]["down_dof_angles"] # down_joint_angles
        self.init_from_prepared_state_data = self._task_cfg["env"]["init_state"]["init_from_prepared_state_data"]

        # control
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.action_scale = self._task_cfg["env"]["control"]["action_scale"]
        self.control_type = self._task_cfg["env"]["control"]["control_type"]

        # other
        self.dt = 1 / 60
        self.max_episode_length_s = self._task_cfg["env"]["episode_length_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self._sim_steps = 0
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["num_actions"]

        # domain rand
        self.push_interval_s = self._task_cfg["env"]["domain_rand"]["push_interval_s"]
        self.max_push_vel_xy = self._task_cfg["env"]["domain_rand"]["max_push_vel_xy"]
        self.push_robots = self._task_cfg["env"]["domain_rand"]["push_robots"]
        self.max_push_interval = np.ceil(self.push_interval_s / self.dt)
        self.push_interval = int(random.uniform(self.max_push_interval/3, self.max_push_interval))

        # reward solving
        self.rew_register_list = []
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt
            if self.rew_scales[key] != 0:
                self.rew_register_list.append(key)

        if self.is_sample_init_state:
            self.push_robots = False
            self.init_from_prepared_state_data = False
        
        if self.init_from_prepared_state_data:
            self.prepared_init_state_data = np.load(os.path.join(RL4Robot_omnisim_DATA_DIR, 'UnitreeA1Stand_init_state_samples.npy'))
            self.prepared_init_state_data_cnt = self.prepared_init_state_data.shape[0]

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_unitree_a1()
        super().set_up_scene(scene)
        self._unitree_a1s = UnitreeA1View(prim_paths_expr="/World/envs/.*/unitree_a1", name="UnitreeA1View", track_contact_forces=True, prepare_contact_sensors=True)
        scene.add(self._unitree_a1s)
        scene.add(self._unitree_a1s._thighs)
        scene.add(self._unitree_a1s._calfs)
        scene.add(self._unitree_a1s._feet)
        scene.add(self._unitree_a1s._trunk)

        return

    def get_unitree_a1(self):
        unitree_a1 = UnitreeA1(prim_path=self.default_zero_env_path + "/unitree_a1", name="unitree_a1", translation=torch.tensor(self.base_init_state[:3]), orientation=torch.tensor(self.base_init_state[3:7]))
        self._sim_config.apply_articulation_settings("unitree_a1", get_prim_at_path(unitree_a1.prim_path), self._sim_config.parse_actor_config("unitree_a1"))
        unitree_a1.set_unitree_a1_properties(self._stage, unitree_a1.prim)
        unitree_a1.prepare_contacts(self._stage, unitree_a1.prim)

        # Configure joint properties
        joint_paths = [
            # 'trunk/FL_hip_joint', 'FL_hip/FL_thigh_joint', 'FL_thigh/FL_calf_joint', 
            # 'trunk/FR_hip_joint', 'FR_hip/FR_thigh_joint', 'FR_thigh/FR_calf_joint', 
            # 'trunk/RL_hip_joint', 'RL_hip/RL_thigh_joint', 'RL_thigh/RL_calf_joint', 
            # 'trunk/RR_hip_joint', 'RR_hip/RR_thigh_joint', 'RR_thigh/RR_calf_joint', 
            'trunk/FL_hip_joint', 'trunk/FR_hip_joint', 'trunk/RL_hip_joint', 'trunk/RR_hip_joint', 
            'FL_hip/FL_thigh_joint', 'FR_hip/FR_thigh_joint', 'RL_hip/RL_thigh_joint', 'RR_hip/RR_thigh_joint', 
            'FL_thigh/FL_calf_joint', 'FR_thigh/FR_calf_joint', 'RL_thigh/RL_calf_joint', 'RR_thigh/RR_calf_joint', 
        ]
        if not self.is_sample_init_state:
            for joint_path in joint_paths:
                set_drive(
                    f"{unitree_a1.prim_path}/{joint_path}",
                    "angular",
                    "position",
                    self.default_dof_poses[joint_path.split('/')[1].split('_')[1]],
                    self.Kp,
                    self.Kd,
                    self.dof_torque_limits[joint_path.split('/')[1].split('_')[1]] * self.soft_torque_limit,
                )

        self.default_dof_pos = torch.zeros((12), dtype=torch.float, device=self.device, requires_grad=False)
        self.down_dof_pos = torch.zeros((12), dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos_limit = torch.zeros((12, 2), dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limit = torch.zeros((12), dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_torque_limit = torch.zeros((12), dtype=torch.float, device=self.device, requires_grad=False)
        for i, dof_name in enumerate(unitree_a1.dof_names):
            self.default_dof_pos[i] = self.default_dof_poses[dof_name.split('_')[1]]
            self.down_dof_pos[i] = self.down_dof_angles[dof_name.split('_')[1]]
            self.dof_pos_limit[i, 0] = self.dof_pos_limits[dof_name.split('_')[1]][0] * self.soft_dof_pos_limit
            self.dof_pos_limit[i, 1] = self.dof_pos_limits[dof_name.split('_')[1]][1] * self.soft_dof_pos_limit
            self.dof_vel_limit[i] = self.dof_vel_limits[dof_name.split('_')[1]]
            self.dof_torque_limit[i] = self.dof_torque_limits[dof_name.split('_')[1]]
        return

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return
        
        if self.dummy_action:
            actions = torch.zeros_like(actions, dtype=actions.dtype, device=actions.device)
            actions[:] = self.default_dof_pos[:]
        
        self._sim_steps += 1

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # # actions always be default_dof_position
        # actions = self.default_dof_pos.repeat(self._num_envs, 1)
        # self.push_robots = False
        
        self.las_actions[:] = self.actions[:]
        self.actions[:] = actions.clone().to(self._device)
        # current_targets = self.current_targets + self.action_scale * self.actions * self.dt
        current_targets = self.action_scale * self.actions
        self.current_targets[:] = tensor_clamp(current_targets, self.dof_pos_limit[:, 0], self.dof_pos_limit[:, 1])
        self._unitree_a1s.set_joint_position_targets(self.current_targets)
        
        if self.push_robots:
            self.push_interval -= 1
            if self.push_interval <= 0:
                self._push_robots()
                self.push_interval = int(random.uniform(self.max_push_interval/3, self.max_push_interval))
        return

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize init state
        # dof_pos = torch.zeros((num_resets, 12), dtype=torch.float, device=self._device, requires_grad=False)
        # dof_pos[:, :4] = torch_rand_float(self.dof_pos_limit[0, 0], self.dof_pos_limit[0, 1], (num_resets, 4), device=self._device)
        # dof_pos[:, 4:8] = torch_rand_float(self.dof_pos_limit[4, 0], self.dof_pos_limit[4, 1], (num_resets, 4), device=self._device)
        # dof_pos[:, 8:12] = torch_rand_float(self.dof_pos_limit[8, 0], self.dof_pos_limit[8, 1], (num_resets, 4), device=self._device)
        # dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._unitree_a1s.num_dof), device=self._device)
        # dof_pos = self.default_dof_pos.repeat(num_resets, 1)
        dof_pos = torch.zeros((num_resets, 12), dtype=torch.float, device=self._device, requires_grad=False)
        dof_pos[:] = self.default_dof_pos[:]
        if self.is_sample_init_state:
            dof_pos[:, :4] = torch_rand_float(self.dof_pos_limit[0, 0], self.dof_pos_limit[0, 1], (num_resets, 4), device=self.device)
            dof_pos[:, 4:8] = torch_rand_float(self.dof_pos_limit[4, 0], self.dof_pos_limit[4, 1], (num_resets, 4), device=self.device)
            dof_pos[:, 8:12] = torch_rand_float(self.dof_pos_limit[8, 0], self.dof_pos_limit[8, 1], (num_resets, 4), device=self.device)
        dof_vel = torch.zeros((num_resets, 12), dtype=torch.float, device=self._device, requires_grad=False)

        root_pos, root_rot = self.init_pos[env_ids, :], self.init_rot[env_ids, :]
        root_vel = torch.zeros((num_resets, 6), dtype=torch.float, device=self._device, requires_grad=False)
        
        if self.init_from_prepared_state_data:
            sample_idx = np.random.choice(self.prepared_init_state_data_cnt, num_resets)
            samples = self.prepared_init_state_data[sample_idx, :]
            dof_pos[:, :] = torch.tensor(samples[:, -12:], dtype=torch.float32, device=self.device, requires_grad=False)
            root_rot[:, :] = torch.tensor(samples[:, 1:5], dtype=torch.float32, device=self.device, requires_grad=False)
            root_pos[:, 2] = torch.tensor(samples[:, 0], dtype=torch.float32, device=self.device, requires_grad=False)

        self.current_targets[env_ids] = dof_pos[:]
        
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._unitree_a1s.set_joint_positions(dof_pos, indices)
        self._unitree_a1s.set_joint_velocities(dof_vel, indices)
        self._unitree_a1s.set_joint_position_targets(dof_pos, indices)

        self._unitree_a1s.set_world_poses(root_pos, root_rot, indices)
        self._unitree_a1s.set_velocities(root_vel, indices)

        # self.commands_x[env_ids] = torch_rand_float(
        #     self.command_x_range[0], self.command_x_range[1], (num_resets, 1), device=self._device
        # ).squeeze()
        # self.commands_y[env_ids] = torch_rand_float(
        #     self.command_y_range[0], self.command_y_range[1], (num_resets, 1), device=self._device
        # ).squeeze()
        # self.commands_yaw[env_ids] = torch_rand_float(
        #     self.command_yaw_range[0], self.command_yaw_range[1], (num_resets, 1), device=self._device
        # ).squeeze()

        print_rew_summary = {}
        for key, rew in self.rew_summary.items():
            print_rew_summary[key] = rew[env_ids].mean().item()
        print('')
        pp.pprint(print_rew_summary)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.las_actions[env_ids] = 0.
        self.las_dof_vel[env_ids] = 0.
        self.max_down_still_reward[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        for key in self.rew_summary.keys():
            self.rew_summary[key][env_ids] = 0.

        return

    def post_reset(self):
        self.init_pos, self.init_rot = self._unitree_a1s.get_world_poses()
        self.init_pos[:, 2] = self.base_init_state[2]
        
        # self.current_targets = self.default_dof_pos.repeat(self._num_envs, 1)
        self.current_targets = torch.zeros((self._num_envs, 12), dtype=torch.float, device=self._device, requires_grad=False)

        # dof_limits = self._unitree_a1s.get_dof_limits()
        # self.dof_pos_limit[:, 0] = dof_limits[0, :, 0].to(device=self._device)
        # self.dof_pos_limit[:, 1] = dof_limits[0, :, 1].to(device=self._device)

        self.commands = torch.zeros(self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self.commands_y = self.commands.view(self._num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self._num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self._num_envs, 3)[..., 2]

        # initialize some data used later on
        self.extras = {}
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat(
            (self._num_envs, 1)
        )
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.las_dof_vel = torch.zeros((self._num_envs, 12), dtype=torch.float, device=self._device, requires_grad=False)
        self.las_actions = torch.zeros(self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False)

        self.time_out_buf = torch.zeros_like(self.reset_buf)
        
        self.collision_contact_forces = torch.zeros((self._num_envs, 8, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.reset_contact_forces = torch.zeros((self._num_envs, 1, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.feet_contact_forces = torch.zeros((self._num_envs, 4, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.las_collision_contact_forces = torch.zeros((self._num_envs, 8, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.las_reset_contact_forces = torch.zeros((self._num_envs, 1, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.las_feet_contact_forces = torch.zeros((self._num_envs, 4, 3), dtype=torch.float, device=self._device, requires_grad=False)
        
        self.feet_air_time = torch.zeros((self._num_envs), dtype=torch.float, device=self._device, requires_grad=False)
        
        self.max_down_still_reward = torch.zeros(self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)

        self.rew_summary = {}
        for rew_name in self.rew_register_list:
            self.rew_summary[rew_name] = torch.zeros((self._num_envs,), dtype=torch.float, device=self._device, requires_grad=False)

        if self.is_sample_init_state:
            self.total_sample_num = 204800
            # self.total_sample_num = 16
            self.sample_buf = np.zeros((self.total_sample_num, 17), dtype=np.float32) # height_1, rot_4, dof_pos_12
            self.sample_idx = 0

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        return

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._unitree_a1s.get_world_poses(clone=False)
        root_velocities = self._unitree_a1s.get_velocities(clone=False)
        dof_pos = self._unitree_a1s.get_joint_positions(clone=False)
        dof_vel = self._unitree_a1s.get_joint_velocities(clone=False)
        torques = self._unitree_a1s.get_applied_joint_efforts(clone=False)
        base_lin_vel = quat_rotate_inverse(torso_rotation, root_velocities[:, 0:3])
        base_ang_vel = quat_rotate_inverse(torso_rotation, root_velocities[:, 3:6])
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        
        self.torso_position, self.torso_rotation, self.root_velocities, self.dof_pos, self.dof_vel, self.torques, self.base_lin_vel, self.base_ang_vel, self.projected_gravity = torso_position, torso_rotation, root_velocities, dof_pos, dof_vel, torques, base_lin_vel, base_ang_vel, projected_gravity
        
        self.las_collision_contact_forces[:] = self.collision_contact_forces[:]
        self.las_reset_contact_forces[:] = self.reset_contact_forces[:]
        self.las_feet_contact_forces[:] = self.feet_contact_forces[:]
        trunk_contact_forces = self._unitree_a1s._trunk.get_net_contact_forces(clone=False, dt=self.dt).view(self._num_envs, 1, 3)
        thighs_contact_forces = self._unitree_a1s._thighs.get_net_contact_forces(clone=False, dt=self.dt).view(self._num_envs, 4, 3)
        calfs_contact_forces = self._unitree_a1s._calfs.get_net_contact_forces(clone=False, dt=self.dt).view(self._num_envs, 4, 3)
        feet_contact_forces = self._unitree_a1s._feet.get_net_contact_forces(clone=False, dt=self.dt).view(self._num_envs, 4, 3)
        self.collision_contact_forces = torch.cat((thighs_contact_forces, calfs_contact_forces), dim=1)
        self.reset_contact_forces = trunk_contact_forces
        self.feet_contact_forces = feet_contact_forces
        
        self.las_dof_vel[:] = dof_vel[:]
        # self.fallen_over = torch.any(torch.norm(self.reset_contact_forces, dim=-1) > 1.0, dim=1)
        self.fallen_over = self.projected_gravity[:, 2] > -0.0
        
        if self.add_noise:
            noise_projected_gravity = projected_gravity + (2.0 * torch.rand_like(projected_gravity) - 1.0) * self.gravity_noise_scale * self.noise_level
            obs = torch.cat((
                    # base_lin_vel * self.lin_vel_scale,
                    # base_ang_vel * self.ang_vel_scale,
                    noise_projected_gravity,
                    self.commands * torch.tensor(
                        [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                        requires_grad=False,
                        device=self.commands.device,
                    ),
                    (torch.normal(dof_pos, self.dof_pos_noise_scale * self.noise_level) - self.default_dof_pos) * self.dof_pos_scale,
                    torch.normal(dof_vel, self.dof_vel_noise_scale * self.noise_level) * self.dof_vel_scale,
                    # self.actions,
                ),
                dim=-1,
            )
        else:
            obs = torch.cat((
                    # base_lin_vel * self.lin_vel_scale,
                    # base_ang_vel * self.ang_vel_scale,
                    projected_gravity,
                    self.commands * torch.tensor(
                        [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                        requires_grad=False,
                        device=self.commands.device,
                    ),
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    dof_vel * self.dof_vel_scale,
                    # self.actions,
                ),
                dim=-1,
            )
        self.obs_buf[:] = obs

        observations = {
            self._unitree_a1s.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def calculate_metrics(self) -> None:
        torso_position, torso_rotation = self._unitree_a1s.get_world_poses(clone=False)
        root_velocities = self._unitree_a1s.get_velocities(clone=False)
        dof_pos = self._unitree_a1s.get_joint_positions(clone=False)
        dof_vel = self._unitree_a1s.get_joint_velocities(clone=False)
        torques = self._unitree_a1s.get_applied_joint_efforts(clone=False)
        base_lin_vel = quat_rotate_inverse(torso_rotation, root_velocities[:, 0:3])
        base_ang_vel = quat_rotate_inverse(torso_rotation, root_velocities[:, 3:6])
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        
        self.torso_position, self.torso_rotation, self.root_velocities, self.dof_pos, self.dof_vel, self.torques, self.base_lin_vel, self.base_ang_vel, self.projected_gravity = torso_position, torso_rotation, root_velocities, dof_pos, dof_vel, torques, base_lin_vel, base_ang_vel, projected_gravity

        # reward calculation
        total_reward = torch.zeros((self._num_envs), dtype=torch.float, device=self.device, requires_grad=False)
        for rew_name in self.rew_register_list:
            this_reward = getattr(self, f'_reward_{rew_name}')() * self.rew_scales[rew_name]
            total_reward += this_reward
            self.rew_summary[rew_name] += this_reward

        self.rew_buf[:] = total_reward.detach()
        return

    def is_done(self) -> None:
        # reset agents
        if self.is_sample_init_state:
            self.time_out_buf = ((self.dof_vel == 0).all(dim=1) & (self.progress_buf >= 30)) | (self.progress_buf >= self.max_episode_length - 1)
        else:
            self.time_out_buf = (self.progress_buf >= self.max_episode_length - 1)
        self.reset_buf[:] = self.time_out_buf | self.fallen_over

        if self.is_sample_init_state and torch.sum(self.time_out_buf) > 0:
            time_out_idx = torch.nonzero(self.time_out_buf).reshape(-1)
            num_time_outs = time_out_idx.shape[0]
            samples = np.concatenate((
                self.torso_position[time_out_idx, [2]].cpu().numpy().reshape(num_time_outs, 1),
                self.torso_rotation[time_out_idx, :].cpu().numpy().reshape(num_time_outs, 4),
                self.dof_pos[time_out_idx, :].cpu().numpy().reshape(num_time_outs, 12),
            ), axis=-1)
            if self.sample_idx + num_time_outs > self.total_sample_num:
                samples = samples[:self.total_sample_num-self.sample_idx, :]
            self.sample_buf[self.sample_idx:self.sample_idx+num_time_outs, :] = samples[:]
            self.sample_idx = self.sample_idx + num_time_outs
            print(f'acquire sample num: {num_time_outs}, now sample num: {self.sample_idx}')
            if self.sample_idx == self.total_sample_num:
                np.save('./UnitreeA1Stand_init_state_samples.npy', self.sample_buf)
                exit(0)

        self.extras['time_outs'] = self.time_out_buf
        return

    def _push_robots(self):
        root_velocities = self._unitree_a1s.get_velocities(clone=False)
        pushed_velocities = torch.zeros((self._num_envs, 6), dtype=torch.float, device=self.device, requires_grad=False)
        pushed_velocities[:, 2:] = root_velocities[:, 2:]
        pushed_velocities[:, :2] = torch_rand_float(-self.max_push_vel_xy, self.max_push_vel_xy, (self._num_envs, 2), device=self.device)
        self._unitree_a1s.set_velocities(pushed_velocities)
        return

        # rew_torque_limits = ((self.torques/self.torque_limits).square() - self.cfg.rewards.soft_torque_limit).clip(min=0.0).sum(dim=1)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        # ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        # rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        # rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        # rew_joint_acc = torch.sum(torch.square(self.las_dof_vel - dof_vel), dim=1) * self.rew_scales["joint_acc"]
        # rew_cosmetic = torch.sum(torch.abs(dof_pos[:, 0:4] - self.default_dof_pos[0:4]), dim=1) * self.rew_scales["cosmetic"]

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.dof_vel[:, :] - self.las_dof_vel[:, :]) / self.dt), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.0 * (torch.norm(self.collision_contact_forces, dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # feet_name: ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        las_contact = (self.las_feet_contact_forces > 1.0)
        contact = (self.feet_contact_forces > 1.0)
        contact_filt = torch.logical_or(contact, las_contact)
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return 1 + self.projected_gravity[:, 2]

    def _reward_action_rate(self):
        # Penalize action rate
        action_rate = self.las_actions - self.actions
        # if self.add_noise:
        #     reward = (0.2*(self.noise_dof_pos - self.actions).square() + 0.8*action_rate.square()).sum(dim=1)
        # else:
        #     reward = (0.1*(self.dof_pos - self.actions).square() + 0.9*action_rate.square()).sum(dim=1)
        reward = (0.1*(self.dof_pos - self.actions).square() + 0.9*action_rate.square()).sum(dim=1)
        return reward

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:, :] - self.dof_pos_limit[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos[:, :] - self.dof_pos_limit[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.square(self.dof_vel/self.dof_vel_limit) - self.soft_dof_vel_limit).clip(min=0.), dim=1)

    def _reward_down_still(self):
        #reward sit down
        _gate = (-torch.sign(self.progress_buf - 0.5 * self.max_episode_length) + 1.0) / 2.0
        rewards = (
            1.0 - (self.dof_pos.view(self._num_envs, 3, 4) - self.down_dof_pos.view(3, 4))\
                .square().clip(min=0.02).max(dim=-1)[0].mean(dim=-1)
        ) * _gate
        torch.maximum(rewards, self.max_down_still_reward, out=self.max_down_still_reward)
        return self.max_down_still_reward * _gate

    def _reward_stand_still(self):
        # reward stand up
        _gate = (torch.sign(self.progress_buf - self.max_episode_length * 0.8) + 1.0) / 2.0
        rewards = (
            1.0 - (self.dof_pos.view(self._num_envs, 3, 4) - self.default_dof_pos.view(3, 4))\
                .square().clip(min=0.02).max(dim=-1)[0].mean(dim=-1)
        ) * _gate
        return rewards

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return ((self.torques/self.dof_torque_limit).square() - self.soft_torque_limit).clip(min=0.0).sum(dim=1)
