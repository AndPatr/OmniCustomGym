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

import time

from typing import Union, Tuple, Dict, List

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from lrhcontrolenvs.utils.math_utils import quat_to_omega, quaternion_difference, rel_vel
from lrhcontrolenvs.utils.xmj_jnt_imp_cntrl import XMjJntImpCntrl
from lrhc_control.envs.lrhc_remote_env_base import LRhcEnvBase
from adarl_ros.adapters.XbotMjAdapter import RosXbotAdapter
from xbot2_mujoco.PyXbotMjSim import LoadingUtils
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame,world2base_frame3D

class RtDeploymentEnv(LRhcEnvBase):

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
            "Multiple robots are not supported!",
            LogType.EXCEP,
            throw_when_excep = True)

        if not num_envs==1:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Parallel deployment is not supported!",
            LogType.EXCEP,
            throw_when_excep = True)
            
        if use_gpu:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Remote deployment env should run on CPU!",
            LogType.EXCEP,
            throw_when_excep = True)

        self._ros_xbot_adapter_init_tsteps=n_init_step
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
            override_low_lev_controller=False)
        # BaseTask.__init__(self,name=self._name,offset=None)
    
    def _pre_setup(self):
        self._render = False

    def _parse_env_opts(self):
        xmj_opts={}
        xmj_opts["use_gpu"]=False
        xmj_opts["state_from_xbot"]=True
        xmj_opts["device"]="cpu"
        xmj_opts["gravity"] = np.array([0.0, 0.0, -9.81])
        xmj_opts["use_diff_vels"] = False

        xmj_opts["xmj_files_dir"]=None

        xmj_opts["rt_safety_perf_coeff"]=1.0

        xmj_opts["xbot2_filter_prof"]="safe"

        xmj_opts.update(self._env_opts) # update defaults with provided opts
        
        xmj_opts["use_gpu_pipeline"]=False
        xmj_opts["device"]="cpu"
        xmj_opts["sim_device"]="cpu"
        # overwrite env opts in case some sim params were missing
        self._env_opts=xmj_opts

        # update device flag based on sim opts
        self._device=xmj_opts["device"]
        self._use_gpu=xmj_opts["use_gpu"]

    def _init_world(self):
    
        info = "Initializing deployment environment"
        Journal.log(self.__class__.__name__,
            "__init__",
            info,
            LogType.STAT,
            throw_when_excep = True)
    
        self._configure_scene()

    def _configure_scene(self):
        
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            urdf_path = self._robot_urdf_paths[robot_name]
            srdf_path = self._robot_srdf_paths[robot_name]
            self._generate_rob_descriptions(robot_name=robot_name, 
                                    urdf_path=urdf_path,
                                    srdf_path=srdf_path)

            self._ros_xbot_adapter=RosXbotAdapter(model_name=robot_name,
                stepLength_sec=self._cluster_dt[robot_name],
                forced_ros_master_uri= None,
                blocking_observation=False,
                is_floating_base=True,
                reference_frame="world",
                torch_device=torch.device(self._device),
                fallback_cmd_stiffness=200.0,
                fallback_cmd_damping=60.0,
                allow_fallback=True,
                enable_filters=True)
            # self._ros_xbot_adapter.build_scenario()
            self._ros_xbot_adapter.startup()
            self._ros_xbot_adapter.set_filters(set_enabled=True, 
                profile_name=self._env_opts["xbot2_filter_prof"])

            self._time_for_pre_step=0.0

            to_monitor=[]
            self._robot_iface_enabled_jnts=self._ros_xbot_adapter.get_robot_interface().getEnabledJointNames()
            
            for jnt in range(len(self._robot_iface_enabled_jnts)):
                to_monitor.append((self._robot_names[i],self._robot_iface_enabled_jnts[jnt]))

            self._ros_xbot_adapter.set_monitored_joints(to_monitor)
            self._ros_xbot_adapter.set_impedance_controlled_joints(to_monitor)

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
        self._ros_xbot_adapter.setJointsImpedanceCommand(self._jnt_imp_controllers[self._robot_names[0]].get_pvesd())
    
    def _pre_step_db(self): 
        
        start=time.perf_counter()
        super()._pre_step_db()
        self._time_for_pre_step=time.perf_counter()-start

    def _pre_step(self): 
        start=time.perf_counter()
        super()._pre_step()
        self._time_for_pre_step=time.perf_counter()-start

    def _step_world(self): 
        walltime_to_sleep=self.physics_dt()-self._time_for_pre_step
        if walltime_to_sleep<0:
            Journal.log(self.__class__.__name__,
                "_step_world",
                f"RT performance violated of {walltime_to_sleep} s.",
                LogType.WARN,
            throw_when_excep = True)
            walltime_to_sleep=0
            
        self._ros_xbot_adapter.run(duration_sec=self._env_opts["rt_safety_perf_coeff"]*walltime_to_sleep)

    def _generate_jnt_imp_control(self, robot_name: str):
        
        jnt_imp_controller = XMjJntImpCntrl(xbot_adapter=self._ros_xbot_adapter,
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
        self._ros_xbot_adapter.resetWorld()
        
    def _set_startup_jnt_imp_gains(self,
            robot_name:str, 
            env_indxs: torch.Tensor = None):
        super()._set_startup_jnt_imp_gains(robot_name=robot_name,env_indxs=env_indxs)
        # apply jnt imp cmds to xbot immediately
        self._ros_xbot_adapter.apply_joint_impedances(self._jnt_imp_controllers[self._robot_names[0]].get_pvesd())
        self._ros_xbot_adapter.run(duration_sec=1.0)

    def _reset_state(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            randomize: bool = False):

        if randomize:
            self._randomize_yaw(robot_name=robot_name,env_indxs=None)
            self._set_root_to_defconfig(robot_name=robot_name)
        
        self._reset_sim()
        
        # we update the robots state 
        self._read_root_state_from_robot(env_indxs=env_indxs, 
            robot_name=robot_name)
        self._read_jnts_state_from_robot(env_indxs=env_indxs,
            robot_name=robot_name)
        
    def _read_root_state_from_robot(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            ):
        
        if (not self._env_opts["state_from_xbot"]):
            self._get_root_state(numerical_diff=self._env_opts["use_diff_vels"],
                    env_indxs=env_indxs,
                    robot_name=robot_name)
        else:
            self._get_root_state_xbot(numerical_diff=self._env_opts["use_diff_vels"],
                    env_indxs=env_indxs,
                    robot_name=robot_name)
            
    def _read_jnts_state_from_robot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):            
        
        if (not self._env_opts["state_from_xbot"]):
            self._get_robots_jnt_state(
                numerical_diff=self._env_opts["use_diff_vels"],
                env_indxs=env_indxs,
                robot_name=robot_name)
        else:
            self._get_robots_jnt_state_xbot(
                numerical_diff=self._env_opts["use_diff_vels"],
                env_indxs=env_indxs,
                robot_name=robot_name) 
            
    def _get_root_state(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False,
        transf_to_base_loc: bool = True):
        
        raise NotImplementedError()

    def _get_root_state_xbot(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False,
        transf_to_base_loc: bool = True):
        
        frame_id, q, omega, linacc = self._ros_xbot_adapter.get_imu_data()

        # in sim we get pos from sim
        # self._root_p[robot_name][:, :] = torch.from_numpy(self._ros_xbot_adapter.xmj_env().p).reshape(self._num_envs, -1).to(self._dtype)

        self._root_q[robot_name][:, :] = torch.from_numpy(q).reshape(self._num_envs, -1).to(self._dtype)

        dt=self._cluster_dt[robot_name] # getting diff state always at cluster rate

        if not numerical_diff:
            # we get velocities from the simulation. This is not good since 
            # these can actually represent artifacts which do not have physical meaning.
            # It's better to obtain them by differentiation to avoid issues with controllers, etc...
            # self._root_v[robot_name][:, :] = torch.from_numpy(self._ros_xbot_adapter.xmj_env().twist[0:3]).reshape(self._num_envs, -1).to(self._dtype)   
            self._root_omega[robot_name][:, :] = torch.from_numpy(omega).reshape(self._num_envs, -1).to(self._dtype)        
            
            self._root_a[robot_name][env_indxs, :] = torch.from_numpy(linacc).reshape(self._num_envs, -1).to(self._dtype)  

            self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
                                            self._root_omega_prev[robot_name][env_indxs, :]) / dt 
            
            # self._root_v_prev[robot_name][env_indxs, :] = self._root_v[robot_name][env_indxs, :] 
            self._root_omega_prev[robot_name][env_indxs, :] = self._root_omega[robot_name][env_indxs, :]

        else:
            # differentiate numerically
            # self._root_v[robot_name][:, :] = (self._root_p[robot_name] - \
            #                                 self._root_p_prev[robot_name]) / dt 
            self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q_prev[robot_name], 
                                                        self._root_q[robot_name], 
                                                        dt)

            Journal.log(self.__class__.__name__,
                "_get_root_state_xbot",
                "Reading root state with differentiation not supported yet!!",
                LogType.EXCEP,
                throw_when_excep = True)
            
            # self._root_a[robot_name][env_indxs, :] = (self._root_v[robot_name][env_indxs, :] - \
            #                                     self._root_v_prev[robot_name][env_indxs, :]) / dt 
            self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
                                            self._root_omega_prev[robot_name][env_indxs, :]) / dt 
            
            # update "previous" data for numerical differentiation
            # self._root_p_prev[robot_name][:, :] = self._root_p[robot_name]
            self._root_q_prev[robot_name][:, :] = self._root_q[robot_name]
            # self._root_v_prev[robot_name][env_indxs, :] = self._root_v[robot_name][env_indxs, :] 
            self._root_omega_prev[robot_name][env_indxs, :] = self._root_omega[robot_name][env_indxs, :]

        world2base_frame3D(v_w=self._gravity_normalized[robot_name],q_b=self._root_q[robot_name],
                v_out=self._gravity_normalized_base_loc[robot_name])

        # if transf_to_base_loc:
        #     # rotate robot twist in base local
        #     twist_w=torch.cat((self._root_v[robot_name], 
        #         self._root_omega[robot_name]), 
        #         dim=1)
        #     twist_base_loc=torch.cat((self._root_v_base_loc[robot_name], 
        #         self._root_omega_base_loc[robot_name]), 
        #         dim=1)
        #     world2base_frame(t_w=twist_w,q_b=self._root_q[robot_name],t_out=twist_base_loc)
        #     self._root_v_base_loc[robot_name]=twist_base_loc[:, 0:3]
        #     self._root_omega_base_loc[robot_name]=twist_base_loc[:, 3:6]

        #     # rotate robot a in base local
        #     a_w=torch.cat((self._root_a[robot_name], 
        #         self._root_alpha[robot_name]), 
        #         dim=1)
        #     a_base_loc=torch.cat((self._root_a_base_loc[robot_name], 
        #         self._root_alpha_base_loc[robot_name]), 
        #         dim=1)
        #     world2base_frame(t_w=a_w,q_b=self._root_q[robot_name],t_out=a_base_loc)
        #     self._root_a_base_loc[robot_name]=a_base_loc[:, 0:3]
        #     self._root_alpha_base_loc[robot_name]=a_base_loc[:, 3:6]

        #     world2base_frame3D(v_w=self._gravity_normalized[robot_name],q_b=self._root_q[robot_name],
        #         v_out=self._gravity_normalized_base_loc[robot_name])
            
    def _get_robots_jnt_state(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False):
        
        raise NotImplementedError()

    def _get_robots_jnt_state_xbot(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False):

        jnt_state_from_xbot=self._ros_xbot_adapter.getJointsState().T # [3(p, v, e)x n_jnts]

        self._jnts_q[robot_name][:, :] = jnt_state_from_xbot[0,:]

        dt= self.physics_dt() if self._override_low_lev_controller else self._cluster_dt[robot_name]

        if dt is None:
            self._jnts_v[robot_name][:, :] = jnt_state_from_xbot[1,:]
        else: 
            self._jnts_v[robot_name][:, :] = self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name] - \
                self._jnts_q_prev[robot_name]) / dt
            
            self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name]

        self._jnts_eff[robot_name][env_indxs, :] = jnt_state_from_xbot[2,:]

    def _set_jnts_homing(self, robot_name: str):
        self._ros_xbot_adapter.trigger_homing() # blocking, moves the robot using plugins
                
    def _set_root_to_defconfig(self, robot_name: str):
        msg="Cannot teleport robot in real world! Please ensure the robot is in the desired reset configuration"
        Journal.log(self.__class__.__name__,
            "_set_root_to_defconfig",
            msg,
            LogType.WARN,
            throw_when_excep = True)
        time.sleep(5.0)

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

    def _get_contact_f(self, 
        robot_name: str, 
        contact_link: str,
        env_indxs: torch.Tensor) -> torch.Tensor:
        return None
    
    def _init_robots_state(self):

        for i in range(0, len(self._robot_names)):

            robot_name = self._robot_names[i]
        
            # root p (measured, previous, default)
            self._root_p[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype)
            self._root_p_prev[robot_name] = self._root_p[robot_name].clone()
            # print(self._root_p_default[robot_name].device)
            self._root_p_default[robot_name] = self._root_p[robot_name].clone()
            # root q (measured, previous, default)
            self._root_q[robot_name] = torch.zeros((self._num_envs, 4), dtype=self._dtype)
            self._root_q[robot_name][:, 0]=1
            self._root_q_prev[robot_name] = self._root_q[robot_name].clone()
            self._root_q_default[robot_name] = self._root_q[robot_name].clone()
            # jnt q (measured, previous, default)
            n_jnts=len(self._robot_iface_enabled_jnts)
            self._jnts_q[robot_name] = torch.zeros((self._num_envs, n_jnts), dtype=self._dtype)
            self._jnts_q_prev[robot_name] = self._jnts_q[robot_name].clone()
            self._jnts_q_default[robot_name] = self._jnts_q[robot_name].clone()
            
            # root v (measured, default)
            self._root_v[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype)
            self._root_v_base_loc[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_v_prev[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_v_default[robot_name] = self._root_v[robot_name].clone()

            # root omega (measured, default)
            self._root_omega[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype)
            self._root_omega_prev[robot_name] = torch.full_like(self._root_omega[robot_name], fill_value=0.0)
            self._root_omega_base_loc[robot_name] = torch.full_like(self._root_omega[robot_name], fill_value=0.0)
            self._root_omega_default[robot_name] = self._root_omega[robot_name].clone()

            # root a (measured,)
            self._root_a[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_a_base_loc[robot_name] = torch.full_like(self._root_a[robot_name], fill_value=0.0)
            self._root_alpha[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_alpha_base_loc[robot_name] = torch.full_like(self._root_alpha[robot_name], fill_value=0.0)

            # joints v (measured, default)
            self._jnts_v[robot_name] = torch.zeros((self._num_envs, n_jnts), dtype=self._dtype)
            self._jnts_v_default[robot_name] = self._jnts_v[robot_name].clone()
            
            # joints efforts (measured, default)
            self._jnts_eff[robot_name] = torch.zeros((self._num_envs, n_jnts), dtype=self._dtype)
            self._jnts_eff_default[robot_name] = self._jnts_eff[robot_name].clone()

            self._root_pos_offsets[robot_name] = torch.zeros((self._num_envs, 3), 
                                device=self._device) # reference position offses
            
            self._root_q_offsets[robot_name] = torch.zeros((self._num_envs, 4), 
                                device=self._device)
            self._root_q_offsets[robot_name][:, 0] = 1.0 # init to valid identity quaternion

            # self._update_root_offsets(robot_name)

    def current_tstep(self):
        return self._ros_xbot_adapter.xmj_env().step_counter
    
    def current_time(self):
        return self._ros_xbot_adapter.getEnvTimeFromReset()
    
    def physics_dt(self):
        robot_name = self._robot_names[0]
        return self._cluster_dt[robot_name]
    
    def rendering_dt(self):
        robot_name = self._robot_names[0]
        return self._cluster_dt[robot_name]
    
    def set_physics_dt(self, physics_dt:float):
        raise NotImplementedError()
    
    def set_rendering_dt(self, rendering_dt:float):
        raise NotImplementedError()
    
    def _robot_jnt_names(self, robot_name: str):
        return self._robot_iface_enabled_jnts
    
    def _is_running(self):
        return self._ros_xbot_adapter.is_ros_control_running()
