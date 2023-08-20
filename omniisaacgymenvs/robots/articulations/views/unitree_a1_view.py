from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import torch

class UnitreeA1View(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "UnitreeA1View",
        track_contact_forces=False,
        prepare_contact_sensors=False
    ) -> None:

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False,
        )

        self._thighs = RigidPrimView(prim_paths_expr="/World/envs/.*/unitree_a1/.*_thigh", name="thighs_view", reset_xform_properties=False, track_contact_forces=track_contact_forces, prepare_contact_sensors=prepare_contact_sensors)
        self._calfs = RigidPrimView(prim_paths_expr="/World/envs/.*/unitree_a1/.*_calf", name="calfs_view", reset_xform_properties=False, track_contact_forces=track_contact_forces, prepare_contact_sensors=prepare_contact_sensors)
        self._feet = RigidPrimView(prim_paths_expr="/World/envs/.*/unitree_a1/.*_foot", name="feet_view", reset_xform_properties=False, track_contact_forces=track_contact_forces, prepare_contact_sensors=prepare_contact_sensors)
        self._trunk = RigidPrimView(prim_paths_expr="/World/envs/.*/unitree_a1/trunk", name="trunk_view", reset_xform_properties=False, track_contact_forces=track_contact_forces, prepare_contact_sensors=prepare_contact_sensors)

        return

    # def is_thigh_below_threshold(self, threshold, ground_heights=None):
    #     knee_pos, _ = self._knees.get_world_poses()
    #     knee_heights = knee_pos.view((-1, 4, 3))[:, :, 2]
    #     if ground_heights is not None:
    #         knee_heights -= ground_heights
    #     return (knee_heights[:, 0] < threshold) | (knee_heights[:, 1] < threshold) | (knee_heights[:, 2] < threshold) | (knee_heights[:, 3] < threshold)

    # def is_base_below_threshold(self, threshold, ground_heights):
    #     base_pos, _ = self.get_world_poses()
    #     base_heights = base_pos[:, 2]
    #     base_heights -= ground_heights
    #     return (base_heights[:] < threshold)

    # def initialize(self, physics_sim_view):
    #     super().initialize(physics_sim_view)

    #     self.actuated_joint_names = [
    #         'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 
    #         'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
    #         'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 
    #         'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 
    #     ]
    #     self._actuated_dof_indices = list()
    #     for joint_name in self.actuated_joint_names:
    #         self._actuated_dof_indices.append(self.get_dof_index(joint_name))
    #     self._actuated_dof_indices.sort()

    #     limit_stiffness = torch.tensor([30.0] * self.num_fixed_tendons, device=self._device)
    #     damping = torch.tensor([0.1] * self.num_fixed_tendons, device=self._device)
    #     self.set_fixed_tendon_properties(dampings=damping, limit_stiffnesses=limit_stiffness)
