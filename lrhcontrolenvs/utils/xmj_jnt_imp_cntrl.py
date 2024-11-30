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


from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from lrhc_control.utils.jnt_imp_control_base import JntImpCntrlBase
from adarl_ros.adapters.XbotMjAdapter import XbotMjAdapter

class XMjJntImpCntrl(JntImpCntrlBase):

    def __init__(self, 
        xbot_adapter: XbotMjAdapter,
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
        
        self._xbot_adapter = xbot_adapter # used to actually apply control
        # signals to the robot
        n_envs=1 # multiple envs not supported
        controlled_joints=self._xbot_adapter.get_impedance_controlled_joints()
        jnts_names=[]

        i=0
        self._model_name=controlled_joints[0][0]
        for joint in controlled_joints:
            if not self._model_name==joint[0]:
                Journal.log(self.__class__.__name__,
                    "__init__",
                    f"Only one model name is currently supported. Read {joint[0]}, while prev. {self._model_name}",
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            jnts_names.append(joint[1])
            i+=1
        n_jnts=len(controlled_joints)
        
        super().__init__(num_envs=n_envs,
            n_jnts=n_jnts,
            jnt_names=jnts_names,
            default_pgain=self._xbot_adapter.fallback_striffness(),
            default_vgain=self._xbot_adapter.fallback_damping(),
            device=device,
            filter_BW=filter_BW,
            filter_dt=filter_dt,
            dtype=dtype,
            enable_safety=enable_safety,
            urdf_path=urdf_path,
            config_path=config_path,
            enable_profiling=enable_profiling,
            debug_checks=debug_checks,
            override_low_lev_controller=override_art_controller)
        
        self._pvesd_adapter=torch.full((n_jnts, 5), fill_value=0.0,
            device=torch.device("cpu"), 
            dtype=self._torch_dtype)

    def get_pvesd(self):
        return self._pvesd_adapter
    
    def _set_gains(self, 
        kps: torch.Tensor = None, 
        kds: torch.Tensor = None):

        if kps is not None:
            kps_cpu=kps.cpu()
            self._pvesd_adapter[:, 3]=kps_cpu
        if kds is not None:
            kds_cpu=kds.cpu()
            self._pvesd_adapter[:, 4]=kds_cpu

    def _set_pos_ref(self, pos: torch.Tensor):
        pos_cpu=pos.cpu()
        self._pvesd_adapter[:, 0]=pos_cpu
        
    def _set_vel_ref(self, vel: torch.Tensor):
        vel_cpu=vel.cpu()
        self._pvesd_adapter[:, 1]=vel_cpu

    def _set_joint_efforts(self, effort: torch.Tensor):
        effort_cpu=effort.cpu()
        self._pvesd_adapter[:, 2]=effort_cpu
                    