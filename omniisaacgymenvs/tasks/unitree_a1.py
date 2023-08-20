from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.unitree_a1 import UnitreeA1
from omniisaacgymenvs.robots.articulations.views.unitree_a1_view import UnitreeA1View
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.utils.torch.rotations import *

import numpy as np
import torch
import math
import random


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

        # control
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.action_scale = self._task_cfg["env"]["control"]["action_scale"]
        self.control_type = self._task_cfg["env"]["control"]["control_type"]

        # other
        self.dt = 1 / 60
        self.max_episode_length_s = self._task_cfg["env"]["episode_length_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        # domain rand
        self.push_interval_s = self._task_cfg["env"]["domain_rand"]["push_interval_s"]
        self.max_push_vel_xy = self._task_cfg["env"]["domain_rand"]["max_push_vel_xy"]
        self.push_robots = self._task_cfg["env"]["domain_rand"]["push_robots"]
        self.max_push_interval = np.ceil(self.push_interval_s / self.dt)
        self.push_interval = int(random.uniform(self.max_push_interval/3, self.max_push_interval))

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["num_actions"]

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
        for joint_path in joint_paths:
            set_drive(f"{unitree_a1.prim_path}/{joint_path}", "angular", "position", self.default_dof_poses[joint_path.split('/')[1].split('_')[1]], self.Kp, self.Kd, self.dof_torque_limits[joint_path.split('/')[1].split('_')[1]])

        self.default_dof_pos = torch.zeros((12), dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos_limit = torch.zeros((12, 2), dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limit = torch.zeros((12), dtype=torch.float, device=self.device, requires_grad=False)
        for i, dof_name in enumerate(unitree_a1.dof_names):
            self.default_dof_pos[i] = self.default_dof_poses[dof_name.split('_')[1]]
            self.dof_pos_limit[i, 0] = self.dof_pos_limits[dof_name.split('_')[1]][0]
            self.dof_pos_limit[i, 1] = self.dof_pos_limits[dof_name.split('_')[1]][1]
            self.dof_vel_limit[i] = self.dof_vel_limits[dof_name.split('_')[1]]
        return

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._unitree_a1s.get_world_poses(clone=False)
        print(f'position: {torso_position[0, :]}')
        root_velocities = self._unitree_a1s.get_velocities(clone=False)
        dof_pos = self._unitree_a1s.get_joint_positions(clone=False)
        dof_vel = self._unitree_a1s.get_joint_velocities(clone=False)

        lin_velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, lin_velocity) * self.lin_vel_scale
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity) * self.ang_vel_scale
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale
        dof_vel_scaled = dof_vel * self.dof_vel_scale

        commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            requires_grad=False,
            device=self.commands.device,
        )

        obs = torch.cat(
            (
                # base_lin_vel,
                # base_ang_vel,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel_scaled,
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

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # actions always be default_dof_position
        actions = self.default_dof_pos.repeat(self.num_envs, 1)

        self.actions[:] = actions.clone().to(self._device)
        # current_targets = self.current_targets + self.action_scale * self.actions * self.dt
        current_targets = self.action_scale * self.actions
        self.current_targets[:] = tensor_clamp(current_targets, self.unitree_a1_dof_lower_limits, self.unitree_a1_dof_upper_limits)
        self._unitree_a1s.set_joint_position_targets(self.current_targets)
        
        if self.push_robots:
            self.push_interval -= 1
            if self.push_interval <= 0:
                self._push_robots()
                self.push_interval = int(random.uniform(self.max_push_interval/3, self.max_push_interval))
        return

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize DOF velocities
        dof_pos = self.default_dof_pos.repeat(num_resets, 1)
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._unitree_a1s.num_dof), device=self._device)

        self.current_targets[env_ids] = dof_pos[:]

        root_pos, root_rot = self.init_pos[env_ids, :], self.init_rot[env_ids, :]
        root_vel = torch.zeros((num_resets, 6), dtype=torch.float, device=self._device, requires_grad=False)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._unitree_a1s.set_joint_positions(dof_pos, indices)
        self._unitree_a1s.set_joint_velocities(dof_vel, indices)

        self._unitree_a1s.set_world_poses(root_pos, root_rot, indices)
        self._unitree_a1s.set_velocities(root_vel, indices)

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (num_resets, 1), device=self._device
        ).squeeze()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.

        return

    def post_reset(self):
        self.init_pos, self.init_rot = self._unitree_a1s.get_world_poses()
        self.current_targets = self.default_dof_pos.repeat(self.num_envs, 1)

        dof_limits = self._unitree_a1s.get_dof_limits()
        self.unitree_a1_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.unitree_a1_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

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
        self.last_dof_vel = torch.zeros((self._num_envs, 12), dtype=torch.float, device=self._device, requires_grad=False)
        self.last_actions = torch.zeros(self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False)

        self.time_out_buf = torch.zeros_like(self.reset_buf)

        # randomize all envs
        indices = torch.arange(self._unitree_a1s.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        return

    def calculate_metrics(self) -> None:
        torso_position, torso_rotation = self._unitree_a1s.get_world_poses(clone=False)
        root_velocities = self._unitree_a1s.get_velocities(clone=False)
        dof_pos = self._unitree_a1s.get_joint_positions(clone=False)
        dof_vel = self._unitree_a1s.get_joint_velocities(clone=False)

        lin_velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, lin_velocity)
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity)
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)

        trunk_contact_forces = self._unitree_a1s._trunk.get_net_contact_forces(clone=False, dt=self.dt).view(self.num_envs, 1, 3)
        thighs_contact_forces = self._unitree_a1s._thighs.get_net_contact_forces(clone=False, dt=self.dt).view(self.num_envs, 4, 3)
        calfs_contact_forces = self._unitree_a1s._calfs.get_net_contact_forces(clone=False, dt=self.dt).view(self.num_envs, 4, 3)
        collision_contact_forces = torch.cat((thighs_contact_forces, calfs_contact_forces), dim=1)
        reset_contact_forces = trunk_contact_forces

        # reward calculation
        rew_names = ['action_rate', 'collision', 'termination', 'dof_pos_limits', 'dof_vel_limits', 'stand_still', 'lin_vel_z', 'orientation']
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew_collision = torch.sum(1.0 * (torch.norm(collision_contact_forces, dim=-1) > 0.1), dim=1)
        rew_termination = self.reset_buf * ~self.time_out_buf
        rew_dof_pos_limits = torch.sum((dof_pos[:, :] - self.dof_pos_limit[:, 1]).clip(min=0.) - (dof_pos[:, :] - self.dof_pos_limit[:, 0]).clip(max=0.), dim=1)
        rew_dof_vel_limits = torch.sum((torch.square(dof_vel/self.dof_vel_limit) - self.soft_dof_vel_limit).clip(min=0.), dim=1)
        rew_stand_still = 1.0 - (dof_pos.view(self.num_envs, 3, 4) - self.default_dof_pos.view(3, 4)).square().clip(min=0.02).max(dim=-1)[0].mean(dim=-1)
        rew_lin_vel_z = torch.square(base_lin_vel[:, 2])
        rew_orientation = 1 + projected_gravity[:, 2]

        # total_reward calculation
        total_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device, requires_grad=False)
        for rew_name in rew_names:
            total_reward += locals()[f'rew_{rew_name}'] * self.rew_scales[rew_name]

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = dof_vel[:]

        self.fallen_over = torch.any(torch.norm(reset_contact_forces, dim=-1) > 1.0, dim=1)
        # total_reward[torch.nonzero(self.fallen_over)] = -1
        self.rew_buf[:] = total_reward.detach()
        return

    def is_done(self) -> None:
        # reset agents
        self.time_out_buf = (self.progress_buf >= self.max_episode_length - 1)
        self.reset_buf[:] = self.time_out_buf | self.fallen_over

        self.extras['time_outs'] = self.time_out_buf
        return

    def _push_robots(self):
        root_velocities = self._unitree_a1s.get_velocities(clone=False)
        pushed_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device, requires_grad=False)
        pushed_velocities[:, 2:] = root_velocities[:, 2:]
        pushed_velocities[:, :2] = torch_rand_float(-self.max_push_vel_xy, self.max_push_vel_xy, (self.num_envs, 2), device=self.device)
        self._unitree_a1s.set_velocities(pushed_velocities)
        return

        # rew_torque_limits = ((self.torques/self.torque_limits).square() - self.cfg.rewards.soft_torque_limit).clip(min=0.0).sum(dim=1)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        # ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        # rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        # rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        # rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - dof_vel), dim=1) * self.rew_scales["joint_acc"]
        # rew_cosmetic = torch.sum(torch.abs(dof_pos[:, 0:4] - self.default_dof_pos[0:4]), dim=1) * self.rew_scales["cosmetic"]