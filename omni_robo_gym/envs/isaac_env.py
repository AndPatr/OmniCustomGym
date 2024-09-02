# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of OmniRoboGym and distributed under the General Public License version 2 license.
# 
# OmniRoboGym is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# OmniRoboGym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with OmniRoboGym.  If not, see <http://www.gnu.org/licenses/>.
# 
from isaacsim import SimulationApp

import carb

import os
import signal

import torch
import numpy as np

from typing import Union, Tuple, Dict, List

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from omni_robo_gym.utils.math_utils import quat_to_omega, quaternion_difference, rel_vel

from lrhc_control.envs.lrhc_sim_env_base import LRhcEnvBase

class IsaacSimEnv(LRhcEnvBase):

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
        sim_opts: Dict = None,
        use_gpu: bool = True,
        dtype: torch.dtype = torch.float32):

        self._backend="torch"
        enable_livestream = sim_opts["enable_livestream"]
        enable_viewport = sim_opts["enable_viewport"]
        sim_device = 0
        base_isaac_exp = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.omnirobogym.kit'
        base_isaac_exp_headless = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.omnirobogym.headless.kit'

        experience=base_isaac_exp
        if sim_opts["headless"]:
            info = f"Will run in headless mode."
            Journal.log(self.__class__.__name__,
                "__init__",
                info,
                LogType.STAT,
                throw_when_excep = True)
            if enable_livestream:
                experience = ""
            elif enable_viewport:
                exception = f"Using viewport is not supported yet."
                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            else:
                experience=base_isaac_exp_headless

        self._simulation_app = SimulationApp({"headless": sim_opts["headless"],
                                            "physics_gpu": sim_device}, 
                                            experience=experience)
        # all imports depending on isaac sim kits have to be done after simulationapp
        # from omni.isaac.core.tasks.base_task import BaseTask
        self._import_isaac_pkgs()
        info = "Using IsaacSim experience file @ " + experience
        Journal.log(self.__class__.__name__,
            "__init__",
            info,
            LogType.STAT,
            throw_when_excep = True)
        # carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)

        if enable_livestream:
            info = "Livestream enabled"
            Journal.log(self.__class__.__name__,
                "__init__",
                info,
                LogType.STAT,
                throw_when_excep = True)
            
            self._simulation_app.set_setting("/app/livestream/enabled", True)
            self._simulation_app.set_setting("/app/window/drawMouse", True)
            self._simulation_app.set_setting("/app/livestream/proto", "ws")
            self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
            self._simulation_app.set_setting("/ngx/enabled", False)
            enable_extension("omni.kit.livestream.native")
            enable_extension("omni.services.streaming.manager")
        self._render = (not sim_opts["headless"]) or enable_livestream or enable_viewport
        self._record = False
        self._world = None
        self._physics_context = None
        self._scene = None
        self._task = None
        self._metadata = None    

        self._robots_art_views = {}
        self._robots_articulations = {}
        self._robots_geom_prim_views = {}
        self.omni_contact_sensors = {}

        self._use_diff_velocities = False

        self._solver_position_iteration_count=sim_opts["solver_position_iteration_count"]
        self._solver_velocity_iteration_count=sim_opts["solver_velocity_iteration_count"]
        self._solver_stabilization_thresh=sim_opts["stabilization_threshold"]
        self._solver_position_iteration_counts={}
        self._solver_velocity_iteration_counts={}
        self._solver_stabilization_threshs={}
        self._robot_bodynames={}
        self._robot_n_links={}
        self._robot_n_dofs={}
        self._robot_dof_names={}
        self._distr_offset={} # decribed how robots within each env are distributed
        self._use_flat_ground=sim_opts["use_flat_ground"]
        self._spawning_radius=sim_opts["spawning_radius"] # [m] -> default distance between roots of robots in a single 

        self._env_ns = "/World/envs"
        self._env_spacing=sim_opts["env_spacing"]
        self._template_env_ns = self._env_ns + "/env_0"
        self._ground_plane_prim_path = "/World/terrain"

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
            n_init_step=n_init_step,
            timeout_ms=timeout_ms,
            custom_opts=sim_opts,
            use_gpu=use_gpu,
            dtype=dtype)
        # BaseTask.__init__(self,name=self._name,offset=None)

    def _import_isaac_pkgs(self):
        # we use global, so that we can create the simulation app inside (and so
        # access Isaac's kit) and also expose to all methods the imports
        global World, omni_kit, get_context, UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools
        global enable_extension, set_camera_view, _urdf, move_prim, GridCloner, prim_utils
        global get_current_stage, Scene, ArticulationView, rep
        global OmniContactSensors, RlTerrains,OmniJntImpCntrl
        from omni.isaac.core.world import World
        from omni.usd import get_context
        from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.viewports import set_camera_view
        import omni.kit as omni_kit
        from omni.importer.urdf import _urdf
        from omni.isaac.core.utils.prims import move_prim
        from omni.isaac.cloner import GridCloner
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.scenes.scene import Scene
        from omni.isaac.core.articulations import ArticulationView
        import omni.replicator.core as rep

        from omni_robo_gym.utils.contact_sensor import OmniContactSensors
        from omni_robo_gym.utils.jnt_imp_cntrl import OmniJntImpCntrl
        from omni_robo_gym.utils.terrains import RlTerrains

    def _calc_robot_distrib(self):

        import math
        # we distribute robots in a single env. along the 
        # circumference of a circle of given radius
        n_robots = len(self._robot_names)
        offset_baseangle = 2 * math.pi / n_robots
        for i in range(n_robots):
            offset_angle = offset_baseangle * (i + 1) 
            robot_offset_wrt_center = torch.tensor([self._spawning_radius * math.cos(offset_angle), 
                                            self._spawning_radius * math.sin(offset_angle), 0], 
                    device=self._device, 
                    dtype=self._dtype)
            # list with n references to the original tensor
            tensor_list = [robot_offset_wrt_center] * self._num_envs
            self._distr_offset[self._robot_names[i]] = torch.stack(tensor_list, dim=0)

    def _init_world(self):

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self._env_ns)
        prim_utils.define_prim(self._template_env_ns)
        self._envs_prim_paths = self._cloner.generate_paths(self._env_ns + "/env", 
                                                self._num_envs)
        self._cloning_offset=self._custom_opts["cloning_offset"]
        if self._cloning_offset is None:
            self._cloning_offset = np.array([[0, 0, 0]] * self._num_envs)
        self._replicate_physics=self._custom_opts["replicate_physics"] 
        
        self.gpu_pipeline_enabled=False
        
        sim_params=self._custom_opts
        # parse device based on sim_param settings
        if "sim_device" in sim_params:
            device = sim_params["sim_device"]
        else:
            device = "cpu"
            physics_device_id = carb.settings.get_settings().get_as_int("/physics/cudaDevice")
            gpu_id = 0 if physics_device_id < 0 else physics_device_id
            if "use_gpu_pipeline" in sim_params:
                # GPU pipeline must use GPU simulation
                if sim_params["use_gpu_pipeline"]:
                    device = "cuda:" + str(gpu_id)
            elif "use_gpu" in sim_params:
                if sim_params["use_gpu"]:
                    device = "cuda:" + str(gpu_id)
        
        info = "Using device: " + str(device)
        Journal.log(self.__class__.__name__,
            "__init__",
            info,
            LogType.STAT,
            throw_when_excep = True)

        self.gpu_pipeline_enabled = sim_params["use_gpu_pipeline"]

        # defaults for integration and rendering dt
        if not("physics_dt" in sim_params):
            sim_params["physics_dt"] = 1.0/1000.0
            dt = sim_params["physics_dt"]
            info = f"Using default integration_dt of {dt} s."
            Journal.log(self.__class__.__name__,
                "set_task",
                info,
                LogType.STAT,
                throw_when_excep = True)
                        
        if not("rendering_dt" in sim_params):
            sim_params["rendering_dt"] = sim_params["physics_dt"]
            dt = sim_params["rendering_dt"]
            info = f"Using default rendering_dt of {dt} s."
            Journal.log(self.__class__.__name__,
                "set_task",
                info,
                LogType.STAT,
                throw_when_excep = True)
                
        self._world = World(
            stage_units_in_meters=1.0, 
            physics_dt=sim_params["physics_dt"], 
            rendering_dt=sim_params["rendering_dt"], # dt between rendering steps. Note: rendering means rendering a frame of 
            # the current application and not only rendering a frame to the viewports/ cameras. 
            # So UI elements of Isaac Sim will be refereshed with this dt as well if running non-headless
            backend=self._backend,
            device=str(device),
            physics_prim_path="/physicsScene", 
            set_defaults = False, # set to True to use the defaults settings [physics_dt = 1.0/ 60.0, 
            # stage units in meters = 0.01 (i.e in cms), rendering_dt = 1.0 / 60.0, gravity = -9.81 m / s 
            # ccd_enabled, stabilization_enabled, gpu dynamics turned off, 
            # broadcast type is MBP, solver type is TGS]
            sim_params=sim_params
        )

        big_info = "[World] Creating Isaac env " + self._name + "\n" + \
            "use_gpu_pipeline: " + str(sim_params["use_gpu_pipeline"]) + "\n" + \
            "device: " + str(device) + "\n" +\
            "backend: " + str(self._backend) + "\n" +\
            "integration_dt: " + str(sim_params["physics_dt"]) + "\n" + \
            "rendering_dt: " + str(sim_params["rendering_dt"]) + "\n" 
        Journal.log(self.__class__.__name__,
            "set_task",
            big_info,
            LogType.STAT,
            throw_when_excep = True)
        
        # we get the physics context to expose additional low-level ##
        # settings of the simulation
        self._physics_context = self._world.get_physics_context() 
        self._physics_scene_path = self._physics_context.prim_path
        self._physics_context.enable_gpu_dynamics(True)
        self._physics_context.enable_stablization(True)
        self._physics_scene_prim = self._physics_context.get_current_physics_scene_prim()
        self._solver_type = self._physics_context.get_solver_type()

        if "gpu_max_rigid_contact_count" in sim_params:
                self._physics_context.set_gpu_max_rigid_contact_count(sim_params["gpu_max_rigid_contact_count"])
        if "gpu_max_rigid_patch_count" in sim_params:
                self._physics_context.set_gpu_max_rigid_patch_count(sim_params["gpu_max_rigid_patch_count"])
        if "gpu_found_lost_pairs_capacity" in sim_params:
                self._physics_context.set_gpu_found_lost_pairs_capacity(sim_params["gpu_found_lost_pairs_capacity"])
        if "gpu_found_lost_aggregate_pairs_capacity" in sim_params:
                self._physics_context.set_gpu_found_lost_aggregate_pairs_capacity(sim_params["gpu_found_lost_aggregate_pairs_capacity"])
        if "gpu_total_aggregate_pairs_capacity" in sim_params:
                self._physics_context.set_gpu_total_aggregate_pairs_capacity(sim_params["gpu_total_aggregate_pairs_capacity"])
        if "gpu_max_soft_body_contacts" in sim_params:
                self._physics_context.set_gpu_max_soft_body_contacts(sim_params["gpu_max_soft_body_contacts"])
        if "gpu_max_particle_contacts" in sim_params:
                self._physics_context.set_gpu_max_particle_contacts(sim_params["gpu_max_particle_contacts"])
        if "gpu_heap_capacity" in sim_params:
                self._physics_context.set_gpu_heap_capacity(sim_params["gpu_heap_capacity"])
        if "gpu_temp_buffer_capacity" in sim_params:
                self._physics_context.set_gpu_temp_buffer_capacity(sim_params["gpu_temp_buffer_capacity"])
        if "gpu_max_num_partitions" in sim_params:
                self._physics_context.set_gpu_max_num_partitions(sim_params["gpu_max_num_partitions"])

        # overwriting defaults
        # self._physics_context.set_gpu_max_rigid_contact_count(2 * self._physics_context.get_gpu_max_rigid_contact_count())
        # self._physics_context.set_gpu_max_rigid_patch_count(2 * self._physics_context.get_gpu_max_rigid_patch_count())
        # self._physics_context.set_gpu_found_lost_pairs_capacity(2 * self._physics_context.get_gpu_found_lost_pairs_capacity())
        # self._physics_context.set_gpu_found_lost_aggregate_pairs_capacity(20 * self._physics_context.get_gpu_found_lost_aggregate_pairs_capacity())
        # self._physics_context.set_gpu_total_aggregate_pairs_capacity(20 * self._physics_context.get_gpu_total_aggregate_pairs_capacity())
        # self._physics_context.set_gpu_heap_capacity(2 * self._physics_context.get_gpu_heap_capacity())
        # self._physics_context.set_gpu_temp_buffer_capacity(20 * self._physics_context.get_gpu_heap_capacity())
        # self._physics_context.set_gpu_max_num_partitions(20 * self._physics_context.get_gpu_temp_buffer_capacity())

        # GPU buffers
        self._gpu_max_rigid_contact_count = self._physics_context.get_gpu_max_rigid_contact_count()
        self._gpu_max_rigid_patch_count = self._physics_context.get_gpu_max_rigid_patch_count()
        self._gpu_found_lost_pairs_capacity = self._physics_context.get_gpu_found_lost_pairs_capacity()
        self._gpu_found_lost_aggregate_pairs_capacity = self._physics_context.get_gpu_found_lost_aggregate_pairs_capacity()
        self._gpu_total_aggregate_pairs_capacity = self._physics_context.get_gpu_total_aggregate_pairs_capacity()
        self._gpu_max_soft_body_contacts = self._physics_context.get_gpu_max_soft_body_contacts()
        self._gpu_max_particle_contacts = self._physics_context.get_gpu_max_particle_contacts()
        self._gpu_heap_capacity = self._physics_context.get_gpu_heap_capacity()
        self._gpu_temp_buffer_capacity = self._physics_context.get_gpu_temp_buffer_capacity()
        # self._gpu_max_num_partitions = physics_context.get_gpu_max_num_partitions() # BROKEN->method does not exist

        big_info2 = "[physics context]:" + "\n" + \
            "gpu_max_rigid_contact_count: " + str(self._gpu_max_rigid_contact_count) + "\n" + \
            "gpu_max_rigid_patch_count: " + str(self._gpu_max_rigid_patch_count) + "\n" + \
            "gpu_found_lost_pairs_capacity: " + str(self._gpu_found_lost_pairs_capacity) + "\n" + \
            "gpu_found_lost_aggregate_pairs_capacity: " + str(self._gpu_found_lost_aggregate_pairs_capacity) + "\n" + \
            "gpu_total_aggregate_pairs_capacity: " + str(self._gpu_total_aggregate_pairs_capacity) + "\n" + \
            "gpu_max_soft_body_contacts: " + str(self._gpu_max_soft_body_contacts) + "\n" + \
            "gpu_max_particle_contacts: " + str(self._gpu_max_particle_contacts) + "\n" + \
            "gpu_heap_capacity: " + str(self._gpu_heap_capacity) + "\n" + \
            "gpu_temp_buffer_capacity: " + str(self._gpu_temp_buffer_capacity)
        
        Journal.log(self.__class__.__name__,
            "set_task",
            big_info2,
            LogType.STAT,
            throw_when_excep = True)

        self._scene = self._world.scene
        self._physics_context = self._world.get_physics_context()

        self._stage = get_context().get_stage()

        distantLight = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/World/DistantLight"))
        distantLight.CreateIntensityAttr(500)

        self._configure_scene()

        # if "enable_viewport" in sim_params:
        #     self._render = sim_params["enable_viewport"]

    def _configure_scene(self):

        for robot_name in self._robot_names:
            self.omni_contact_sensors[robot_name]=None
        self._contact_prims=self._custom_opts["contact_prims"]
        for robot_name in self._contact_prims:
            if not (self._contact_prims[robot_name] is None):
                self.omni_contact_sensors[robot_name]=OmniContactSensors(
                    name=robot_name, 
                    n_envs=self._num_envs, 
                    contact_prims=self._contact_prims, 
                    contact_offsets=self._custom_opts["contact_offsets"], 
                    sensor_radii=self._custom_opts["sensor_radii"], 
                    device=self._device, 
                    dtype=self.dtype,
                    enable_debug=self._debug)
            
        # environment 
        self._fix_base = [False] * len(self._robot_names)
        self._self_collide = [False] * len(self._robot_names)
        self._merge_fixed = [False] * len(self._robot_names)
        
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
            self._import_urdf(robot_name, 
                            fix_base=fix_base, 
                            self_collide=self_collide, 
                            merge_fixed=merge_fixed)
            Journal.log(self.__class__.__name__,
                        "_configure_scene",
                        "cloning environments...",
                        LogType.STAT,
                        throw_when_excep = True)
            self._cloner.clone(
                source_prim_path=self._template_env_ns,
                prim_paths=self._envs_prim_paths,
                replicate_physics=self._replicate_physics,
                position_offsets = self._cloning_offset
            ) # we can clone the environment in which all the robos are
            Journal.log(self.__class__.__name__,
                        "set_up_scene",
                        "finishing scene setup...",
                        LogType.STAT,
                        throw_when_excep = True)
            for i in range(len(self._robot_names)):
                robot_name = self._robot_names[i]
                self._robots_art_views[robot_name] = ArticulationView(name = robot_name + "ArtView",
                                                            prim_paths_expr = self._env_ns + "/env_.*"+ "/" + robot_name + "/base_link", 
                                                            reset_xform_properties=False)
                self._robots_articulations[robot_name] = self._scene.add(self._robots_art_views[robot_name])
                # self._robots_geom_prim_views[robot_name] = GeometryPrimView(name = robot_name + "GeomView",
                #                                                 prim_paths_expr = self._env_ns + "/env*"+ "/" + robot_name,
                #                                                 # prepare_contact_sensors = True
                #                                             )
                # self._robots_geom_prim_views[robot_name].apply_collision_apis() # to be able to apply contact sensors
            
            if self._use_flat_ground:
                self._scene.add_default_ground_plane(z_position=0, 
                            name="terrain", 
                            prim_path= self._ground_plane_prim_path, 
                            static_friction=1.5, 
                            dynamic_friction=1.5, 
                            restitution=0.0)
            else:
                
                self.terrains = RlTerrains(get_current_stage())
                self.terrains.get_obstacles_terrain(terrain_size=40, 
                                            num_obs=100, 
                                            max_height=0.4, 
                                            min_size=0.5,
                                            max_size=5.0)
            # delete_prim(self._ground_plane_prim_path + "/SphereLight") # we remove the default spherical light
            
            # set default camera viewport position and target
            self._set_initial_camera_params()
            self.apply_collision_filters(self._physics_context.prim_path, 
                                "/World/collisions")
            # init contact sensors
            self._init_contact_sensors() # IMPORTANT: this has to be called
            # after calling the clone() method and initializing articulation views!!! 
            self._reset_sim()
            self._fill_robot_info_from_world() 
            # initializes robot state data
            self._init_robots_state()
            # update solver options 
            self._update_art_solver_options() 
            self._get_solver_info() # get again solver option before printing everything
            self._print_envs_info() # debug print

            self.scene_setup_completed = True

    def _render_sim(self, mode="human"):

        if mode == "human":
            self._world.render()
            return None
        elif mode == "rgb_array":
            # check if viewport is enabled -- if not, then complain because we won't get any data
            if not self._render or not self._record:
                exception = f"Cannot render '{mode}' when rendering is not enabled. Please check the provided" + \
                    "arguments to the environment class at initialization."
                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            return rgb_data[:, :, :3]
        else:    
            return None

    def _create_viewport_render_product(self, resolution=(1280, 720)):
        """Create a render product of the viewport for rendering."""

        try:

            # create render product
            self._render_product = rep.create.render_product("/OmniverseKit_Persp", resolution)
            # create rgb annotator -- used to read data from the render product
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
            self._record = True
        except Exception as e:
            carb.log_info("omni.replicator.core could not be imported. Skipping creation of render product.")
            carb.log_info(str(e))

    def _close(self):
        if self._simulation_app.is_running():
            self._simulation_app.close()
    
    def _step_sim(self): 
        self._world.step(render=self._render)

    def _generate_jnt_imp_control(self, robot_name: str):
        
        jnt_imp_controller = OmniJntImpCntrl(articulation=self._robots_articulations[robot_name],
            device=self._device,
            dtype=self._dtype,
            enable_safety=True,
            urdf_path=self._urdf_dump_paths[robot_name],
            config_path=self._jnt_imp_config_paths[robot_name],
            enable_profiling=False,
            debug_checks=self._debug,
            override_art_controller=self._override_low_lev_controller)
        
        return jnt_imp_controller

    def _reset(self,
        env_indxs: torch.Tensor = None,
        robot_names: List[str] =None,
        randomize: bool = False):

        self._reset_state(env_indxs=env_indxs,
            robot_names=robot_names,
            randomize=randomize)

        for i in range(len(robot_names)):
            self._reset_jnt_imp_control(robot_name=robot_names[i],
                                env_indxs=env_indxs)


    def _reset_sim(self):
        self._world.reset(soft=False)
    
    def _reset_state(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str] =None,
            randomize: bool = False):

        rob_names = robot_names if (robot_names is not None) else self._robot_names
        if env_indxs is not None:
            if self._debug:
                if self._use_gpu:
                    if not env_indxs.device.type == "cuda":
                            error = "Provided env_indxs should be on GPU!"
                            Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)
                else:
                    if not env_indxs.device.type == "cpu":
                        error = "Provided env_indxs should be on CPU!"
                        Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)
            for i in range(len(rob_names)):
                robot_name = rob_names[i]
                if randomize:
                    self._randomize_yaw(robot_name=robot_name,env_indxs=env_indxs)

                # root q
                self._robots_art_views[robot_name].set_world_poses(positions = self._root_p_default[robot_name][env_indxs, :],
                                                    orientations=self._root_q_default[robot_name][env_indxs, :],
                                                    indices = env_indxs)
                # jnts q
                self._robots_art_views[robot_name].set_joint_positions(positions = self._jnts_q_default[robot_name][env_indxs, :],
                                                        indices = env_indxs)
                # root v and omega
                self._robots_art_views[robot_name].set_joint_velocities(velocities = self._jnts_v_default[robot_name][env_indxs, :],
                                                        indices = env_indxs)
                # jnts v
                concatenated_vel = torch.cat((self._root_v_default[robot_name][env_indxs, :], 
                                                self._root_omega_default[robot_name][env_indxs, :]), dim=1)
                self._robots_art_views[robot_name].set_velocities(velocities = concatenated_vel,
                                                        indices = env_indxs)
                # jnts eff
                self._robots_art_views[robot_name].set_joint_efforts(efforts = self._jnts_eff_default[robot_name][env_indxs, :],
                                                        indices = env_indxs)
        else:

            for i in range(len(rob_names)):
                robot_name = rob_names[i]

                if randomize:
                    self.randomize_yaw(robot_name=robot_name,env_indxs=None)

                # root q
                self._robots_art_views[robot_name].set_world_poses(positions = self._root_p_default[robot_name][:, :],
                                                    orientations=self._root_q_default[robot_name][:, :],
                                                    indices = None)
                # jnts q
                self._robots_art_views[robot_name].set_joint_positions(positions = self._jnts_q_default[robot_name][:, :],
                                                        indices = None)
                # root v and omega
                self._robots_art_views[robot_name].set_joint_velocities(velocities = self._jnts_v_default[robot_name][:, :],
                                                        indices = None)
                # jnts v
                concatenated_vel = torch.cat((self._root_v_default[robot_name][:, :], 
                                                self._root_omega_default[robot_name][:, :]), dim=1)
                self._robots_art_views[robot_name].set_velocities(velocities = concatenated_vel,
                                                        indices = None)
                # jnts eff
                self._robots_art_views[robot_name].set_joint_efforts(efforts = self._jnts_eff_default[robot_name][:, :],
                                                        indices = None)

        # we update the robots state 
        self._update_state_from_sim(env_indxs=env_indxs, 
                        robot_names=rob_names)
        
    def _import_urdf(self, 
        robot_name: str,
        fix_base = False, 
        self_collide = False, 
        merge_fixed = True):
        
        import_config=_urdf.ImportConfig()
        # status,import_config=omni_kit.commands.execute("URDFCreateImportConfig")

        Journal.log(self.__class__.__name__,
            "update_root_offsets",
            "importing robot URDF",
            LogType.STAT,
            throw_when_excep = True)
        _urdf.acquire_urdf_interface()  
        # we overwrite some settings which are bound to be fixed
        import_config.merge_fixed_joints = merge_fixed # makes sim more stable
        # in case of fixed joints with light objects
        import_config.import_inertia_tensor = True
        # import_config.convex_decomp = False
        import_config.fix_base = fix_base
        import_config.self_collision = self_collide
        # import_config.distance_scale = 1
        # import_config.make_default_prim = True
        # import_config.create_physics_scene = True
        # import_config.default_drive_strength = 1047.19751
        # import_config.default_position_drive_damping = 52.35988
        # import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        # import URDF
        success, robot_prim_path_default = omni_kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self._urdf_dump_paths[robot_name],
            import_config=import_config, 
            # get_articulation_root=True,
        )

        robot_base_prim_path = self._template_env_ns + "/" + robot_name
        # moving default prim to base prim path for cloning
        move_prim(robot_prim_path_default, # from
                robot_base_prim_path) # to
        
        return success

    def apply_collision_filters(self, 
                                physicscene_path: str, 
                                coll_root_path: str):

        self._cloner.filter_collisions(physicsscene_path = physicscene_path,
                                collision_root_path = coll_root_path, 
                                prim_paths=self._envs_prim_paths, 
                                global_paths=[self._ground_plane_prim_path] # can collide with these prims
                                )

    def _update_state_from_sim(self,
                env_indxs: torch.Tensor = None,
                robot_names: List[str] = None):
        
        if self._use_diff_velocities:
            self._get_robots_state(dt = self.integration_dt(),
                            env_indxs = env_indxs,
                            robot_names = robot_names) # updates robot states
            # but velocities are obtained via num. differentiation
        else:
            self._get_robots_state(env_indxs = env_indxs,
                            robot_names = robot_names) # velocities directly from simulator (can 
            # introduce relevant artifacts, making them unrealistic)

    def _get_robots_state(self, 
        env_indxs: torch.Tensor = None,
        robot_names: List[str] = None,
        dt: float = None, 
        reset: bool = False):
         
        rob_names = robot_names if (robot_names is not None) else self._robot_names
        if env_indxs is not None:
            for i in range(0, len(rob_names)):
                robot_name = rob_names[i]
                pose = self._robots_art_views[robot_name].get_world_poses( 
                                                clone = True,
                                                indices=env_indxs) # tuple: (pos, quat)
                
                self._root_p[robot_name][env_indxs, :] = pose[0] 
                self._root_q[robot_name][env_indxs, :] = pose[1] # root orientation
                self._jnts_q[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_joint_positions(
                                                clone = True,
                                                indices=env_indxs) # joint positions 
                if dt is None:
                    # we get velocities from the simulation. This is not good since 
                    # these can actually represent artifacts which do not have physical meaning.
                    # It's better to obtain them by differentiation to avoid issues with controllers, etc...
                    self._root_v[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_linear_velocities(
                                                clone = True,
                                                indices=env_indxs) # root lin. velocity               
                    self._root_omega[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_angular_velocities(
                                                clone = True,
                                                indices=env_indxs) # root ang. velocity
                    self._jnts_v[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_joint_velocities( 
                                                clone = True,
                                                indices=env_indxs) # joint velocities
                else:
                    # differentiate numerically
                    if not reset:                    
                        self._root_v[robot_name][env_indxs, :] = (self._root_p[robot_name][env_indxs, :] - \
                                                        self._root_p_prev[robot_name][env_indxs, :]) / dt 
                        self._root_omega[robot_name][env_indxs, :] = quat_to_omega(self._root_q[robot_name][env_indxs, :], 
                                                                    self._root_q_prev[robot_name][env_indxs, :], 
                                                                    dt)
                        self._jnts_v[robot_name][env_indxs, :] = (self._jnts_q[robot_name][env_indxs, :] - \
                                                        self._jnts_q_prev[robot_name][env_indxs, :]) / dt
                    else:
                        # to avoid issues when differentiating numerically
                        self._root_v[robot_name][env_indxs, :].zero_()
                        self._root_omega[robot_name][env_indxs, :].zero_()
                        self._jnts_v[robot_name][env_indxs, :].zero_()
                    # update "previous" data for numerical differentiation
                    self._root_p_prev[robot_name][env_indxs, :] = self._root_p[robot_name][env_indxs, :] 
                    self._root_q_prev[robot_name][env_indxs, :] = self._root_q[robot_name][env_indxs, :]
                    self._jnts_q_prev[robot_name][env_indxs, :] = self._jnts_q[robot_name][env_indxs, :]

                self._jnts_eff[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_measured_joint_efforts( 
                                                clone = True,
                                                joint_indices=None,
                                                indices=env_indxs) # measured joint efforts (computed by joint force solver)
        else:
            # updating data for all environments
            for i in range(0, len(rob_names)):
                robot_name = rob_names[i]
                pose = self._robots_art_views[robot_name].get_world_poses( 
                                                clone = True) # tuple: (pos, quat)
                self._root_p[robot_name][:, :] = pose[0]  
                self._root_q[robot_name][:, :] = pose[1] # root orientation
                self._jnts_q[robot_name][:, :] = self._robots_art_views[robot_name].get_joint_positions(
                                                clone = True) # joint positions 
                if dt is None:
                    # we get velocities from the simulation. This is not good since 
                    # these can actually represent artifacts which do not have physical meaning.
                    # It's better to obtain them by differentiation to avoid issues with controllers, etc...
                    self._root_v[robot_name][:, :] = self._robots_art_views[robot_name].get_linear_velocities(
                                                clone = True) # root lin. velocity 
                    self._root_omega[robot_name][:, :] = self._robots_art_views[robot_name].get_angular_velocities(
                                                    clone = True) # root ang. velocity
                    self._jnts_v[robot_name][:, :] = self._robots_art_views[robot_name].get_joint_velocities( 
                                                    clone = True) # joint velocities
                else: 
                    # differentiate numerically
                    if not reset:        
                        self._root_v[robot_name][:, :] = (self._root_p[robot_name][:, :] - \
                                                        self._root_p_prev[robot_name][:, :]) / dt 
                        self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q[robot_name][:, :], 
                                                                    self._root_q_prev[robot_name][:, :], 
                                                                    dt)
                        self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name][:, :] - \
                                                        self._jnts_q_prev[robot_name][:, :]) / dt
                        # self._jnts_v[robot_name][:, :].zero_()
                    else:
                        # to avoid issues when differentiating numerically
                        self._root_v[robot_name][:, :].zero_()
                        self._root_omega[robot_name][:, :].zero_()
                        self._jnts_v[robot_name][:, :].zero_()
                    # update "previous" data for numerical differentiation
                    self._root_p_prev[robot_name][:, :] = self._root_p[robot_name][:, :] 
                    self._root_q_prev[robot_name][:, :] = self._root_q[robot_name][:, :]
                    self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name][:, :]
                
                self._jnts_eff[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_measured_joint_efforts( 
                                                clone = True) # measured joint efforts (computed by joint force solver)
    
    def _move_jnts_to_homing(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._robots_art_views[robot_name].set_joints_default_state(positions=self._homing, 
                velocities = torch.zeros((self._homing.shape[0], self._homing.shape[1]), \
                                    dtype=self._dtype, device=self._device), 
                efforts = torch.zeros((self._homing.shape[0], self._homing.shape[1]), \
                                    dtype=self._dtype, device=self._device))
                
    def _move_root_to_defconfig(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._robots_art_views[robot_name].set_default_state(positions=self._root_p_default[robot_name], 
                orientations=self._root_q_default[robot_name])
            
    def _get_solver_info(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._solver_position_iteration_counts[robot_name] = self._robots_art_views[robot_name].get_solver_position_iteration_counts()
            self._solver_velocity_iteration_counts[robot_name] = self._robots_art_views[robot_name].get_solver_velocity_iteration_counts()
            self._solver_stabilization_threshs[robot_name] = self._robots_art_views[robot_name].get_stabilization_thresholds()
    
    def _update_art_solver_options(self):
        
        # sets new solver iteration options for specifc articulations
        self._get_solver_info() # gets current solver info for the articulations of the 
        # environments, so that dictionaries are filled properly
        
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            # increase by a factor
            self._solver_position_iteration_counts[robot_name] = torch.full((self._num_envs,), self._solver_position_iteration_count)
            self._solver_velocity_iteration_counts[robot_name] = torch.full((self._num_envs,), self._solver_velocity_iteration_count)
            self._solver_stabilization_threshs[robot_name] = torch.full((self._num_envs,), self._solver_stabilization_thresh)
            self._robots_art_views[robot_name].set_solver_position_iteration_counts(self._solver_position_iteration_counts[robot_name])
            self._robots_art_views[robot_name].set_solver_velocity_iteration_counts(self._solver_velocity_iteration_counts[robot_name])
            self._robots_art_views[robot_name].set_stabilization_thresholds(self._solver_stabilization_threshs[robot_name])
            self._get_solver_info() # gets again solver info for articulation, so that it's possible to debug if
            # the operation was successful

    def _print_envs_info(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            task_info = f"[{robot_name}]" + "\n" + \
                "bodies: " + str(self._robots_art_views[robot_name].body_names) + "\n" + \
                "n. prims: " + str(self._robots_art_views[robot_name].count) + "\n" + \
                "prims names: " + str(self._robots_art_views[robot_name].prim_paths) + "\n" + \
                "n. bodies: " + str(self._robots_art_views[robot_name].num_bodies) + "\n" + \
                "n. dofs: " + str(self._robots_art_views[robot_name].num_dof) + "\n" + \
                "dof names: " + str(self._robots_art_views[robot_name].dof_names) + "\n" + \
                "solver_position_iteration_counts: " + str(self._solver_position_iteration_counts[robot_name]) + "\n" + \
                "solver_velocity_iteration_counts: " + str(self._solver_velocity_iteration_counts[robot_name]) + "\n" + \
                "stabiliz. thresholds: " + str(self._solver_stabilization_threshs[robot_name])
            # print("dof limits: " + str(self._robots_art_views[robot_name].get_dof_limits()))
            # print("effort modes: " + str(self._robots_art_views[robot_name].get_effort_modes()))
            # print("dof gains: " + str(self._robots_art_views[robot_name].get_gains()))
            # print("dof max efforts: " + str(self._robots_art_views[robot_name].get_max_efforts()))
            # print("dof gains: " + str(self._robots_art_views[robot_name].get_gains()))
            # print("physics handle valid: " + str(self._robots_art_views[robot_name].is_physics_handle_valid())
            Journal.log(self.__class__.__name__,
                "_print_envs_info",
                task_info,
                LogType.STAT,
                throw_when_excep = True)
    
    def _fill_robot_info_from_world(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._robot_bodynames[robot_name] = self._robots_art_views[robot_name].body_names
            self._robot_n_links[robot_name] = self._robots_art_views[robot_name].num_bodies
            self._robot_n_dofs[robot_name] = self._robots_art_views[robot_name].num_dof
            self._robot_dof_names[robot_name] = self._robots_art_views[robot_name].dof_names
    
    def _set_initial_camera_params(self, 
                                camera_position=[10, 10, 3], 
                                camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, 
                        target=camera_target, 
                        camera_prim_path="/OmniverseKit_Persp")
    
    def _init_contact_sensors(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            # creates base contact sensor (which is then cloned)
            if self.omni_contact_sensors[robot_name] is not None:
                self.omni_contact_sensors[robot_name].create_contact_sensors(
                                                        self._world)
    
    def _init_robots_state(self):

        self._calc_robot_distrib()

        for i in range(0, len(self._robot_names)):

            robot_name = self._robot_names[i]
            pose = self._robots_art_views[robot_name].get_world_poses( 
                clone = True) # tuple: (pos, quat)

            # root p (measured, previous, default)
            self._root_p[robot_name] = pose[0]  
            self._root_p_prev[robot_name] = torch.clone(pose[0])
            self._root_p_default[robot_name] = torch.clone(pose[0]) + self._distr_offset[robot_name]
            # root q (measured, previous, default)
            self._root_q[robot_name] = pose[1] # root orientation
            self._root_q_prev[robot_name] = torch.clone(pose[1])
            self._root_q_default[robot_name] = torch.clone(pose[1])
            # jnt q (measured, previous, default)
            self._jnts_q[robot_name] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True) # joint positions 
            self._jnts_q_prev[robot_name] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True) 
            self._jnts_q_default[robot_name] = torch.full((self._jnts_q[robot_name].shape[0], 
                                                           self._jnts_q[robot_name].shape[1]), 
                                                            0.0, 
                                                            dtype=self._dtype, 
                                                            device=self._device)
            
            self._homing
            # root v (measured, default)
            self._root_v[robot_name] = self._robots_art_views[robot_name].get_linear_velocities(
                                            clone = True) # root lin. velocity
            self._root_v_default[robot_name] = torch.full((self._root_v[robot_name].shape[0], self._root_v[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self._dtype, 
                                                        device=self._device)
            # root omega (measured, default)
            self._root_omega[robot_name] = self._robots_art_views[robot_name].get_angular_velocities(
                                            clone = True) # root ang. velocity
            self._root_omega_default[robot_name] = torch.full((self._root_omega[robot_name].shape[0], self._root_omega[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self._dtype, 
                                                        device=self._device)
            # joints v (measured, default)
            self._jnts_v[robot_name] = self._robots_art_views[robot_name].get_joint_velocities( 
                                            clone = True) # joint velocities
            self._jnts_v_default[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self._dtype, 
                                                        device=self._device)
            
            # joints efforts (measured, default)
            self._jnts_eff[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                0.0, 
                                                dtype=self._dtype, 
                                                device=self._device)
            self._jnts_eff_default[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                    0.0, 
                                                    dtype=self._dtype, 
                                                    device=self._device)
            self._root_pos_offsets[robot_name] = torch.zeros((self._num_envs, 3), 
                                device=self._device) # reference position offses
            
            self._root_q_offsets[robot_name] = torch.zeros((self._num_envs, 4), 
                                device=self._device)
            self._root_q_offsets[robot_name][:, 0] = 1.0 # init to valid identity quaternion

            self._update_root_offsets(robot_name)

    def current_tstep(self):
        self._world.current_time_step_index
    
    def current_time(self):
        return self._world.current_time
    
    def physics_dt(self):
        return self._world.get_physics_dt()
    
    def rendering_dt(self):
        return self._world.get_rendering_dt()
    
    def set_physics_dt(self, physics_dt:float):
        self._world.set_simulation_dt(physics_dt=physics_dt,rendering_dt=None)
    
    def set_rendering_dt(self, rendering_dt:float):
        self._world.set_simulation_dt(physics_dt=None,rendering_dt=rendering_dt)
    
    def _robot_jnt_names(self, robot_name: str):
        return self._robots_art_views[robot_name].dof_names
