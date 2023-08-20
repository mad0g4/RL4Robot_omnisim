from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

import carb
from pxr import Usd, UsdGeom, Sdf, Gf, PhysxSchema, UsdPhysics


class UnitreeA1(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "unitree_a1",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = assets_root_path + "/Isaac/Robots/Unitree/a1.usd"

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        self._dof_names = [
            # 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 
            # 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
            # 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 
            # 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 
            'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
            'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
            'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint', 
        ]

        return

    @property
    def dof_names(self):
        return self._dof_names

    def set_unitree_a1_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64/np.pi*180)
        return

    def prepare_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                link_prim_path = str(link_prim.GetPrimPath())
                if "trunk" in link_prim_path or \
                        "thigh" in link_prim_path or \
                        "calf" in link_prim_path or \
                        "foot" in link_prim_path:
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)
        return

    # def set_motor_control_mode(self, stage, unitree_a1_path):
    #     # max_force is lost, default to 1.0
    #     joints_config = {
    #         "FL_hip_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'trunk/FL_hip_joint'},
    #         "FL_thigh_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'FL_hip/FL_thigh_joint'},
    #         "FL_calf_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'FL_thigh/FL_calf_joint'},
    #         "FR_hip_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'trunk/FR_hip_joint'},
    #         "FR_thigh_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'FR_hip/FR_thigh_joint'},
    #         "FR_calf_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'FR_thigh/FR_calf_joint'},
    #         "RL_hip_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'trunk/RL_hip_joint'},
    #         "RL_thigh_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'RL_hip/RL_thigh_joint'},
    #         "RL_calf_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'RL_thigh/RL_calf_joint'},
    #         "RR_hip_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'trunk/RR_hip_joint'},
    #         "RR_thigh_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'RR_hip/RR_thigh_joint'},
    #         "RR_calf_joint": {"stiffness": 40.0, "damping": 0.1, "max_force": 1.0, 'relative_prim_path': 'RR_thigh/RR_calf_joint'},
    #     }

    #     for joint_name, config in joints_config.items():
    #         set_drive(
    #             f"{self.prim_path}/{config['relative_prim_path']}",
    #             "angular",
    #             "position",
    #             0.0, 
    #             config["stiffness"]*np.pi/180, 
    #             config["damping"]*np.pi/180,
    #             config["max_force"],
    #         )