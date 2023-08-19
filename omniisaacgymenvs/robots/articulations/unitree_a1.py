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

        self._position = torch.tensor([0.0, 0.0, 0.4]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation
            
        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

    def set_shadow_hand_properties(self, stage, shadow_hand_prim):
        for link_prim in shadow_hand_prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(True)

    def set_motor_control_mode(self, stage, shadow_hand_path):
        joints_config = {
                         "robot0_WRJ1": {"stiffness": 5, "damping": 0.5, "max_force": 4.785},
                         "robot0_WRJ0": {"stiffness": 5, "damping": 0.5, "max_force": 2.175},
                         "robot0_FFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
                         "robot0_FFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
                         "robot0_FFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
                         "robot0_MFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
                         "robot0_MFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
                         "robot0_MFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
                         "robot0_RFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
                         "robot0_RFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
                         "robot0_RFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
                         "robot0_LFJ4": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
                         "robot0_LFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
                         "robot0_LFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
                         "robot0_LFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
                         "robot0_THJ4": {"stiffness": 1, "damping": 0.1, "max_force": 2.3722},
                         "robot0_THJ3": {"stiffness": 1, "damping": 0.1, "max_force": 1.45},
                         "robot0_THJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.99},
                         "robot0_THJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.99},
                         "robot0_THJ0": {"stiffness": 1, "damping": 0.1, "max_force": 0.81},
                        }

        for joint_name, config in joints_config.items():
            set_drive(
                f"{self.prim_path}/joints/{joint_name}", 
                "angular", 
                "position", 
                0.0, 
                config["stiffness"]*np.pi/180, 
                config["damping"]*np.pi/180, 
                config["max_force"]
            )