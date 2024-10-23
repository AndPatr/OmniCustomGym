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
import numpy as np

from typing import Union, Tuple, Dict, List

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from lrhcontrolenvs.utils.math_utils import quat_to_omega, quaternion_difference, rel_vel

from lrhc_control.envs.lrhc_remote_env_base import LRhcEnvBase
from lrhcontrolenvs.utils.xmj_jnt_imp_cntrl import XMjJntImpCntrl
from adarl_ros.adapters.XbotMjAdapter import XbotMjAdapter
from xbot2_mujoco.PyXbotMjSimEnv import LoadingUtils
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame,world2base_frame3D

class XMjSimEnv(LRhcEnvBase):

    def __init__(self,
        robot_names: List[str],
        robot_urdf_paths: List[str],
        robot_srdf_paths: List[str],
        jnt_imp_config_paths: List[str],
        n_contacts: List[int],
        cluster_dt: List[float],
        use_remote_stepping: List[bool],
        name: str = "IsaacSimEnv",
        num_envs: int = 1,
        debug = False,
        verbose: bool = False,
        vlevel: VLevel = VLevel.V1,
        n_init_step: int = 0,
        timeout_ms: int = 60000,
        env_opts: Dict = None,
        use_gpu: bool = False,
        dtype: torch.dtype = torch.float32,
        override_low_lev_controller: bool = False):
        
        if not len(robot_names)==1:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Multi-robot simulation is not supported yet!",
            LogType.EXCEP,
            throw_when_excep = True)

        if not num_envs==1:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Parallel simulation is not supported yet!",
            LogType.EXCEP,
            throw_when_excep = True)
            
        if use_gpu:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Only CPU simulation is supported!",
            LogType.EXCEP,
            throw_when_excep = True)

        self._xmj_adapter_init_tsteps=n_init_step
        super().__init__(name=name,
            robot_names=robot_names,
            robot_urdf_paths=robot_urdf_paths,
            robot_srdf_paths=robot_srdf_paths,
            jnt_imp_config_paths=jnt_imp_config_paths,
            n_contacts=n_contacts,
            cluster_dt=cluster_dt,
            use_remote_stepping=use_remote_stepping,
            num_envs=num_envs,
            debug=debug,
            verbose=verbose,
            vlevel=vlevel,
            n_init_step=0, # adapter will handle init steppingy
            timeout_ms=timeout_ms,
            env_opts=env_opts,
            use_gpu=use_gpu,
            dtype=dtype,
            override_low_lev_controller=override_low_lev_controller)
        # BaseTask.__init__(self,name=self._name,offset=None)

    def _sim_is_running(self):
        return self._xmj_adapter.sim_is_running()
    
    def _pre_setup(self):
        
        self._render = (not self._env_opts["headless"])

    def _parse_env_opts(self):
        xmj_opts={}
        xmj_opts["use_gpu"]=False
        xmj_opts["state_from_xbot"]=False
        xmj_opts["device"]="cpu"
        xmj_opts["sim_device"]="cpu" if xmj_opts["use_gpu"] else "cpu"
        xmj_opts["physics_dt"]=1e-3
        xmj_opts["rendering_dt"]=xmj_opts["physics_dt"]
        xmj_opts["substeps"]=1 # number of physics steps to be taken for for each rendering step
        xmj_opts["gravity"] = np.array([0.0, 0.0, -9.81])
        xmj_opts["use_diff_vels"] = False

        xmj_opts["headless"] = False
        xmj_opts["xmj_files_dir"]=None
        xmj_opts["xmj_timeout"]=1000

        xmj_opts.update(self._env_opts) # update defaults with provided opts
        xmj_opts["rendering_dt"]=xmj_opts["physics_dt"]
        
        if not xmj_opts["use_gpu"]: # don't use GPU at all
            xmj_opts["use_gpu_pipeline"]=False
            xmj_opts["device"]="cpu"
            xmj_opts["sim_device"]="cpu"
        else: # use GPU
            Journal.log(self.__class__.__name__,
            "_parse_env_opts",
            "GPU not supported yet for XMjSimEnv!!",
            LogType.EXCEP,
            throw_when_excep = True)        
        # overwrite env opts in case some sim params were missing
        self._env_opts=xmj_opts

        # update device flag based on sim opts
        self._device=xmj_opts["device"]
        self._use_gpu=xmj_opts["use_gpu"]

    def _init_world(self):
    
        info = "Using sim device: " + str(self._env_opts["sim_device"])
        Journal.log(self.__class__.__name__,
            "__init__",
            info,
            LogType.STAT,
            throw_when_excep = True)
                         
        big_info = "[World] Creating Mujoco-xbot2 simulation " + self._name + "\n" + \
            "use_gpu_pipeline: " + str(self._env_opts["use_gpu_pipeline"]) + "\n" + \
            "device: " + str(self._env_opts["sim_device"]) + "\n" +\
            "integration_dt: " + str(self._env_opts["physics_dt"]) + "\n" + \
            "rendering_dt: " + str(self._env_opts["rendering_dt"]) + "\n" 
        Journal.log(self.__class__.__name__,
            "_init_world",
            big_info,
            LogType.STAT,
            throw_when_excep = True)
    
        self._configure_scene()

        # if "enable_viewport" in sim_params:
        #     self._render = sim_params["enable_viewport"]

    def _configure_scene(self):

        # environment 
        self._fix_base = [False] * len(self._robot_names)
        self._self_collide = [False] * len(self._robot_names)
        self._merge_fixed = [True] * len(self._robot_names)
        
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            urdf_path = self._robot_urdf_paths[robot_name]
            srdf_path = self._robot_srdf_paths[robot_name]
            fix_base = self._fix_base[i]
            self_collide = self._self_collide[i]
            merge_fixed = self._merge_fixed[i]
            self._generate_rob_descriptions(robot_name=robot_name, 
                                    urdf_path=urdf_path,
                                    srdf_path=srdf_path)
            
            self._xmj_helper = LoadingUtils(self._name)
            xmj_files_dir=self._env_opts["xmj_files_dir"]
            if xmj_files_dir is None:
                Journal.log(self.__class__.__name__,
                    "_configure_scene",
                    "xmj_files_dir is None. It should be a valid path to where sim_opt.xml, world.xml and sites.xml files are.",
                    LogType.EXCEP,
                    throw_when_excep = True)
            self._xmj_helper.set_simopt_path(xmj_files_dir+"/sim_opt.xml")
            self._xmj_helper.set_world_path(xmj_files_dir+"/world.xml")
            self._xmj_helper.set_sites_path(xmj_files_dir+"/sites.xml")
            
            self._xmj_helper.set_urdf_path(self._urdf_dump_paths[self._robot_names[0]])
            self._xmj_helper.set_srdf_path(self._srdf_dump_paths[self._robot_names[0]])
            self._xmj_helper.set_xbot_config_path(self._jnt_imp_config_paths[self._robot_names[0]])
            self._xmj_helper.generate()
            self._mj_xml_path = self._xmj_helper.xml_path()

            self._xmj_adapter=XbotMjAdapter(model_fpath=self._mj_xml_path,
                model_name=self._robot_names[0],
                xbot2_config_path=self._jnt_imp_config_paths[self._robot_names[0]],
                stepLength_sec=self._env_opts["physics_dt"],
                headless=self._env_opts["headless"],
                init_steps=self._xmj_adapter_init_tsteps,
                timeout_ms=self._env_opts["xmj_timeout"],
                forced_ros_master_uri= None,
                maxObsDelay=float("+inf"),
                blocking_observation=False,
                is_floating_base=True,
                reference_frame="world",
                torch_device=torch.device(self._device),
                fallback_cmd_stiffness=200.0,
                fallback_cmd_damping=100.0,
                allow_fallback=True,
                enable_filters=True)
            self._xmj_adapter.startup()
            
            to_monitor=[]
            jnt_names_sim=self._robot_jnt_names(robot_name=robot_name)
            for jnt in range(len(jnt_names_sim)):
                to_monitor.append((self._robot_names[i],jnt_names_sim[jnt]))
            self._xmj_adapter.set_monitored_joints(to_monitor)
            self._xmj_adapter.set_impedance_controlled_joints(to_monitor)

            Journal.log(self.__class__.__name__,
                        "set_up_scene",
                        "finishing sim pre-setup...",
                        LogType.STAT,
                        throw_when_excep = True)
     
            self._reset_sim()
            self._fill_robot_info_from_world() 
            # initializes robot state data
            self._init_robots_state()
            # update solver options 
            self._print_envs_info() # debug print

            self.scene_setup_completed = True
    
    def _xrdf_cmds(self, robot_name:str):
        cmds=super()._xrdf_cmds(robot_name=robot_name)
        for i, s in enumerate(cmds):
            if "floating_joint:=" in s: # mujoco needs a floating joint
                cmds[i] = "floating_joint:=true" 
        return cmds

    def _render_sim(self, mode="human"):
        pass

    def _close(self):
        pass
    
    def _apply_cmds_to_jnt_imp_control(self, robot_name:str):
        super()._apply_cmds_to_jnt_imp_control(robot_name=robot_name)
        self._xmj_adapter.setJointsImpedanceCommand(self._jnt_imp_controllers[self._robot_names[0]].get_pvesd())

    def _step_sim(self): 
        time_elapsed=self._xmj_adapter.step()
        if not (abs(time_elapsed-self.physics_dt())<1e-6):
            Journal.log(self.__class__.__name__,
                "_step_sim",
                f"simulation stepped of {time_elapsed} [s], while expected one should be {self.physics_dt()} [s]",
                LogType.EXCEP,
                throw_when_excep = True)
        
    def _generate_jnt_imp_control(self, robot_name: str):
        
        jnt_imp_controller = XMjJntImpCntrl(xbot_adapter=self._xmj_adapter,
            device=self._device,
            dtype=self._dtype,
            enable_safety=True,
            urdf_path=self._urdf_dump_paths[robot_name],
            config_path=self._jnt_imp_config_paths[robot_name],
            enable_profiling=False,
            debug_checks=self._debug,
            override_art_controller=self._override_low_lev_controller)
        
        return jnt_imp_controller

    def _reset_sim(self):
        self._xmj_adapter.resetWorld()
    
    def _reset_state(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            randomize: bool = False):

        if randomize:
            self._randomize_yaw(robot_name=robot_name,env_indxs=None)
            self._set_root_to_defconfig(robot_name=robot_name)
        
        # we update the robots state 
        self._read_root_state_from_robot(env_indxs=env_indxs, 
            robot_name=robot_name)

    def _read_root_state_from_robot(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            ):
        
        if self._env_opts["use_diff_vels"]:
            self._get_root_state(dt=self.physics_dt(),
                env_indxs=env_indxs,
                robot_name=robot_name) # updates robot states
            # but velocities are obtained via num. differentiation
        else:
            self._get_root_state(env_indxs=env_indxs,
                robot_name=robot_name) # velocities directly from simulator (can 
            # introduce relevant artifacts, making them unrealistic)

    def _read_jnts_state_from_robot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):            
        
        if (not self._env_opts["state_from_xbot"]) and (not self._env_opts["use_diff_vels"]):
            self._get_robots_jnt_state(env_indxs=env_indxs,
                            robot_name=robot_name)
        elif self._env_opts["state_from_xbot"] and (not self._env_opts["use_diff_vels"]):
            self._get_robots_jnt_state_xbot(env_indxs=env_indxs,
                            robot_name=robot_name)
        elif (not self._env_opts["state_from_xbot"]) and self._env_opts["use_diff_vels"]:
            self._get_robots_jnt_state(dt=self.physics_dt(),
                            env_indxs=env_indxs,
                            robot_name=robot_name) 
        else:
            self._get_robots_jnt_state_xbot(dt=self.physics_dt(),
                            env_indxs=env_indxs,
                            robot_name=robot_name) 
    def _get_root_state(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        dt: float = None, 
        base_loc: bool = True):
        
        self._root_p[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().p).reshape(self._num_envs, -1).to(self._dtype)
        self._root_q[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().q).reshape(self._num_envs, -1).to(self._dtype)

        if dt is None:
            # we get velocities from the simulation. This is not good since 
            # these can actually represent artifacts which do not have physical meaning.
            # It's better to obtain them by differentiation to avoid issues with controllers, etc...
            self._root_v[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().twist[0:3]).reshape(self._num_envs, -1).to(self._dtype)             
            self._root_omega[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().twist[3:6]).reshape(self._num_envs, -1).to(self._dtype)        
        else:
            # differentiate numerically
            self._root_v[robot_name][:, :] = (self._root_p[robot_name] - \
                                            self._root_p_prev[robot_name]) / dt 
            self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q[robot_name], 
                                                        self._root_q_prev[robot_name], 
                                                        dt)
    
            # update "previous" data for numerical differentiation
            self._root_p_prev[robot_name][:, :] = self._root_p[robot_name]
            self._root_q_prev[robot_name][:, :] = self._root_q[robot_name]

        if base_loc:
            # rotate robot twist in base local
            twist_w=torch.cat((self._root_v[robot_name], 
                self._root_omega[robot_name]), 
                dim=1)
            twist_base_loc=torch.cat((self._root_v_base_loc[robot_name], 
                self._root_omega_base_loc[robot_name]), 
                dim=1)
            world2base_frame(t_w=twist_w,q_b=self._root_q[robot_name],t_out=twist_base_loc)
            self._root_v_base_loc[robot_name]=twist_base_loc[:, 0:3]
            self._root_omega_base_loc[robot_name]=twist_base_loc[:, 3:6]

            world2base_frame3D(v_w=self._gravity_normalized[robot_name],q_b=self._root_q[robot_name],
                v_out=self._gravity_normalized_base_loc[robot_name])

    def _get_robots_jnt_state(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        dt: float = None):

        self._jnts_q[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_q).reshape(self._num_envs, -1).to(self._dtype)

        if dt is None:
            self._jnts_v[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_v).reshape(self._num_envs, -1).to(self._dtype)     
        else: 
            self._jnts_v[robot_name][:, :] = self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name] - \
                self._jnts_q_prev[robot_name]) / dt
            
            self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name]

        self._jnts_eff[robot_name][env_indxs, :] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_eff).reshape(self._num_envs, -1).to(self._dtype) 

    def _get_robots_jnt_state_xbot(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        dt: float = None):

        jnt_state_from_xbot=self._xmj_adapter.getJointsState().T # [3(p, v, e)x n_jnts]

        self._jnts_q[robot_name][:, :] = jnt_state_from_xbot[0,:]

        if dt is None:
            self._jnts_v[robot_name][:, :] = jnt_state_from_xbot[1,:]
        else: 
            self._jnts_v[robot_name][:, :] = self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name] - \
                self._jnts_q_prev[robot_name]) / dt
            
            self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name]

        self._jnts_eff[robot_name][env_indxs, :] = jnt_state_from_xbot[2,:]

    def _set_jnts_homing(self, robot_name: str):
        self._xmj_adapter.xmj_env().move_to_homing_now()
                
    def _set_root_to_defconfig(self, robot_name: str):

        self._xmj_adapter.xmj_env().set_pi(self._root_p_default[robot_name].numpy())
        self._xmj_adapter.xmj_env().set_qi(self._root_q_default[robot_name].numpy())

        self._reset_sim()

    def _get_solver_info(self):
        raise NotImplementedError()

    def _print_envs_info(self):
        pass
    
    def _fill_robot_info_from_world(self):
        pass
    
    def _set_initial_camera_params(self, 
                                camera_position=[10, 10, 3], 
                                camera_target=[0, 0, 0]):
        raise NotImplementedError()
    
    def _init_contact_sensors(self):
        raise NotImplementedError()

    def _init_robots_state(self):

        for i in range(0, len(self._robot_names)):

            robot_name = self._robot_names[i]
        
            # root p (measured, previous, default)
            self._root_p[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().p.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._root_p_prev[robot_name] = self._root_p[robot_name].clone()
            # print(self._root_p_default[robot_name].device)
            self._root_p_default[robot_name] = self._root_p[robot_name].clone()
            # root q (measured, previous, default)
            self._root_q[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().q.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._root_q_prev[robot_name] = self._root_q[robot_name].clone()
            self._root_q_default[robot_name] = self._root_q[robot_name].clone()
            # jnt q (measured, previous, default)
            self._jnts_q[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_q.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._jnts_q_prev[robot_name] = self._jnts_q[robot_name].clone()
            self._jnts_q_default[robot_name] = self._jnts_q[robot_name].clone()
            
            # root v (measured, default)
            self._root_v[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().twist.copy()[0:3]).reshape(self._num_envs, -1).to(self._dtype)
            self._root_v_base_loc[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_v_default[robot_name] = self._root_v[robot_name].clone()

            # root omega (measured, default)
            self._root_omega[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().twist.copy()[3:6]).reshape(self._num_envs, -1).to(self._dtype)
            self._root_omega_base_loc[robot_name] = torch.full_like(self._root_omega[robot_name], fill_value=0.0)
            self._root_omega_default[robot_name] = self._root_omega[robot_name].clone()

            # joints v (measured, default)
            self._jnts_v[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_v.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._jnts_v_default[robot_name] = self._jnts_v[robot_name].clone()
            
            # joints efforts (measured, default)
            self._jnts_eff[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_eff.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._jnts_eff_default[robot_name] = self._jnts_eff[robot_name].clone()

            self._root_pos_offsets[robot_name] = torch.zeros((self._num_envs, 3), 
                                device=self._device) # reference position offses
            
            self._root_q_offsets[robot_name] = torch.zeros((self._num_envs, 4), 
                                device=self._device)
            self._root_q_offsets[robot_name][:, 0] = 1.0 # init to valid identity quaternion

            # self._update_root_offsets(robot_name)

    def current_tstep(self):
        return self._xmj_adapter.xmj_env().step_counter
    
    def current_time(self):
        return self._xmj_adapter.getEnvTimeFromReset()
    
    def physics_dt(self):
        return self._xmj_adapter.xmj_env().physics_dt
    
    def rendering_dt(self):
        return self._xmj_adapter.xmj_env().physics_dt
    
    def set_physics_dt(self, physics_dt:float):
        raise NotImplementedError()
    
    def set_rendering_dt(self, rendering_dt:float):
        raise NotImplementedError()
    
    def _robot_jnt_names(self, robot_name: str):
        return self._xmj_adapter.xmj_env().jnt_names()
