# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of LRHControlEnvs and distributed under the General Public License version 2 license.
# 
# LRHControlEnvs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# LRHControlEnvs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with LRHControlEnvs.  If not, see <http://www.gnu.org/licenses/>.
# 
import torch 

from omni.isaac.core.articulations.articulation_view import ArticulationView

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from lrhc_control.utils.jnt_imp_control_base import JntImpCntrlBase
        
class OmniJntImpCntrl(JntImpCntrlBase):

    def __init__(self, 
        articulation: ArticulationView,
        default_pgain: float = 300.0, 
        default_vgain: float = 10.0, 
        device: torch.device = torch.device("cpu"), 
        filter_BW = 50.0, # [Hz]
        filter_dt = None, # should correspond to the dt between samples
        dtype = torch.double,
        enable_safety = True,
        urdf_path: str = None,
        config_path: str = None,
        enable_profiling: bool = False,
        debug_checks: bool = False,
        override_art_controller = False): # [s]
        
        self._articulation_view = articulation # used to actually apply control
        # signals to the robot
        if not self._articulation_view.initialized:
            exception = f"the provided articulation_view is not initialized properly!"
            Journal.log(self.__class__.__name__,
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

        n_envs=self._articulation_view.count
        n_jnts = self._articulation_view.num_dof
        jnts_names = self._articulation_view.dof_names

        super().__init__(num_envs=n_envs,
            n_jnts=n_jnts,
            jnt_names=jnts_names,
            default_pgain=default_pgain,
            default_vgain=default_vgain,
            device=device,
            filter_BW=filter_BW,
            filter_dt=filter_dt,
            dtype=dtype,
            enable_safety=enable_safety,
            urdf_path=urdf_path,
            config_path=config_path,
            enable_profiling=enable_profiling,
            debug_checks=debug_checks,
            override_low_lev_controller=override_art_controller
        )

    def _set_gains(self, 
        kps: torch.Tensor = None, 
        kds: torch.Tensor = None):
        self._articulation_view.set_gains(kps=kps, 
            kds=kds)
    
    def _set_pos_ref(self, pos: torch.Tensor):
        self._articulation_view.set_joint_position_targets(pos)

    def _set_vel_ref(self, vel: torch.Tensor):
        self._articulation_view.set_joint_velocity_targets(vel)

    def _set_joint_efforts(self, effort: torch.Tensor):
        self._articulation_view.set_joint_efforts(effort)
                    