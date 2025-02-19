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
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import numpy as np

from lrhcontrolenvs.utils.terrain_utils import *

from pxr import Usd

class RlTerrains():

    def __init__(self, 
                stage: Usd.Stage,
                prim_path: str = "/World/terrain"):
        
        self._stage = stage

        self._prim_path=prim_path

    def create_wave_terrain(self, 
            terrain_size = 40,
            num_waves = 10, 
            amplitude = 1, 
            position = np.array([0.0, 0.0, 0.0]), 
                dynamic_friction=0.5, 
                static_friction=0.5, 
                restitution=0.1):

        # creates a terrain
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.005  # [m]

        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)

        heightfield = np.zeros((num_terrains * num_rows, 
                                num_cols), dtype=np.int16)

        def new_sub_terrain(): 

            return SubTerrain(width=num_rows, 
                        length=num_cols,
                        vertical_scale=vertical_scale, 
                        horizontal_scale=horizontal_scale)

        heightfield[0:num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=num_waves, 
                                                    amplitude=amplitude).height_field_raw
        
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, 
                                    horizontal_scale=horizontal_scale,
                                    vertical_scale=vertical_scale, 
                                    slope_threshold=1.5)

        position = np.array([-terrain_width/2.0, terrain_length/2.0, 0]) + position

        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

        return add_terrain_to_stage(stage=self._stage, 
                    vertices=vertices, 
                    triangles=triangles,
                    position=position, 
                    orientation=orientation,
                    prim_path=self._prim_path,
                    static_friction=dynamic_friction,
                    dynamic_friction=static_friction,
                    restitution=restitution)
    
    def create_sloped_terrain(self, 
                    terrain_size = 40,
                    slope = -0.5, 
                    position = np.array([0.0, 0.0, 0.0]), 
                dynamic_friction=0.5, 
                static_friction=0.5, 
                restitution=0.1):

        # creates a terrain
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.005  # [m]

        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)

        heightfield = np.zeros((num_terrains * num_rows, 
                                num_cols), dtype=np.int16)

        def new_sub_terrain(): 

            return SubTerrain(width=num_rows, 
                        length=num_cols,
                        vertical_scale=vertical_scale, 
                        horizontal_scale=horizontal_scale)

        heightfield[0:num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(), 
                                        slope=slope).height_field_raw
    
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, 
                                    horizontal_scale=horizontal_scale,
                                    vertical_scale=vertical_scale, 
                                    slope_threshold=1.5)

        position = np.array([-terrain_width/2.0, terrain_length/2.0, 0]) + position

        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

        return add_terrain_to_stage(stage=self._stage, 
                    vertices=vertices, 
                    triangles=triangles,
                    position=position, 
                    orientation=orientation,
                    prim_path=self._prim_path,
                    static_friction=dynamic_friction,
                    dynamic_friction=static_friction,
                    restitution=restitution)
    
    def create_stairs_terrain(self, 
                terrain_size = 40,
                step_width = 0.75, 
                step_height = -0.5, 
                position = np.array([0.0, 0.0, 0.0]), 
                dynamic_friction=0.5, 
                static_friction=0.5, 
                restitution=0.1):

        # creates a terrain
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.005  # [m]

        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)

        heightfield = np.zeros((num_terrains * num_rows, 
                                num_cols), dtype=np.int16)

        def new_sub_terrain(): 

            return SubTerrain(width=num_rows, 
                        length=num_cols,
                        vertical_scale=vertical_scale, 
                        horizontal_scale=horizontal_scale)

        heightfield[0:num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=step_width, 
                                                    step_height=step_height).height_field_raw
        
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, 
                                    horizontal_scale=horizontal_scale,
                                    vertical_scale=vertical_scale, 
                                    slope_threshold=1.5)

        position = np.array([-terrain_width/2.0, terrain_length/2.0, 0]) + position

        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

        return add_terrain_to_stage(stage=self._stage, 
                    vertices=vertices, 
                    triangles=triangles,
                    position=position, 
                    orientation=orientation,
                    prim_path=self._prim_path,
                    static_friction=dynamic_friction,
                    dynamic_friction=static_friction,
                    restitution=restitution)

    def create_random_patched_terrain(self, 
                    terrain_size=40, 
                    min_height=-0.2, 
                    max_height=0.2, 
                    step=0.2, 
                    downsampled_scale=0.5, 
                    position=np.array([0.0, 0.0, 0.0]), 
                    dynamic_friction=0.5, 
                    static_friction=0.5, 
                    restitution=0.1,
                    patch_ratio=0.3,
                    patch_size=10,
                    with_walls: bool =True,
                    wall_height: float = 2.0):

        # Terrain dimensions
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.005  # [m]

        num_rows = int(terrain_width / horizontal_scale)
        num_cols = int(terrain_length / horizontal_scale)

        heightfield = np.zeros((num_terrains * num_rows, num_cols), dtype=np.int16)

        def new_sub_terrain(): 
            return SubTerrain(width=num_rows, 
                            length=num_cols,
                            vertical_scale=vertical_scale, 
                            horizontal_scale=horizontal_scale)

        # Generate base terrain
        terrain = random_uniform_terrain(new_sub_terrain(), 
                                        min_height=min_height, max_height=max_height, 
                                        step=step, 
                                        downsampled_scale=downsampled_scale)
        heightfield[0:num_rows, :] = terrain.height_field_raw

        # Apply flat patches
        num_patches_x = terrain_width // patch_size
        num_patches_y = terrain_length // patch_size

        for i in range(int(num_patches_x * num_patches_y * patch_ratio)):
            patch_x = np.random.randint(0, num_patches_x) * (patch_size / horizontal_scale)
            patch_y = np.random.randint(0, num_patches_y) * (patch_size / horizontal_scale)
            patch_x = int(patch_x)
            patch_y = int(patch_y)

            patch_height = np.random.uniform(min_height, max_height)
            heightfield[patch_x:patch_x + int(patch_size / horizontal_scale),
                        patch_y:patch_y + int(patch_size / horizontal_scale)] = patch_height / vertical_scale

        if with_walls:
            # Add vertical walls at the borders (2 meters height)
            wall_height = wall_height / vertical_scale  # Convert meters to heightmap scale
            heightfield[0, :] = wall_height  # Top border
            heightfield[-1, :] = wall_height  # Bottom border
            heightfield[:, 0] = wall_height  # Left border
            heightfield[:, -1] = wall_height  # Right border

        # Convert to mesh
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, 
                                                            horizontal_scale=horizontal_scale,
                                                            vertical_scale=vertical_scale, 
                                                            slope_threshold=1.5)

        position = np.array([-terrain_width / 2.0, terrain_length / 2.0, 0]) + position
        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

        return add_terrain_to_stage(stage=self._stage, 
                                    vertices=vertices, 
                                    triangles=triangles,
                                    position=position, 
                                    orientation=orientation,
                                    prim_path=self._prim_path,
                                    static_friction=dynamic_friction,
                                    dynamic_friction=static_friction,
                                    restitution=restitution)

    def create_random_uniform_terrain(self, 
                    terrain_size=40, 
                    min_height=-0.2, 
                    max_height=0.2, 
                    step=0.2, 
                    downsampled_scale=0.5, 
                    position=np.array([0.0, 0.0, 0.0]), 
                    dynamic_friction=0.5, 
                    static_friction=0.5, 
                    restitution=0.1,
                    with_walls: bool =True,
                    wall_height: float = 2.0):

        # Terrain dimensions
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.2  # [m]
        vertical_scale = 0.005  # [m]

        num_rows = int(terrain_width / horizontal_scale)
        num_cols = int(terrain_length / horizontal_scale)

        heightfield = np.zeros((num_terrains * num_rows, num_cols), dtype=np.int16)

        def new_sub_terrain(): 
            return SubTerrain(width=num_rows, 
                            length=num_cols,
                            vertical_scale=vertical_scale, 
                            horizontal_scale=horizontal_scale)

        # Generate base terrain
        terrain = random_uniform_terrain(new_sub_terrain(), 
                                        min_height=min_height, max_height=max_height, 
                                        step=step, 
                                        downsampled_scale=downsampled_scale)
        heightfield[0:num_rows, :] = terrain.height_field_raw

        if with_walls:
            # Add vertical walls at the borders (2 meters height)
            wall_height = wall_height / vertical_scale  # Convert meters to heightmap scale
            heightfield[0, :] = wall_height  # Top border
            heightfield[-1, :] = wall_height  # Bottom border
            heightfield[:, 0] = wall_height  # Left border
            heightfield[:, -1] = wall_height  # Right border

        # Convert to mesh
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, 
                                                            horizontal_scale=horizontal_scale,
                                                            vertical_scale=vertical_scale, 
                                                            slope_threshold=1.5)

        position = np.array([-terrain_width / 2.0, terrain_length / 2.0, 0]) + position
        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

        return add_terrain_to_stage(stage=self._stage, 
                                    vertices=vertices, 
                                    triangles=triangles,
                                    position=position, 
                                    orientation=orientation,
                                    prim_path=self._prim_path,
                                    static_friction=dynamic_friction,
                                    dynamic_friction=static_friction,
                                    restitution=restitution)
    
    def get_obstacles_terrain(self, 
                    terrain_size = 40.0, 
                    num_obs = 50, 
                    max_height = 0.5,
                    min_size = 0.5,
                    max_size = 5.0,
                    position = np.array([0.0, 0.0, 0.0]), 
                dynamic_friction=0.5, 
                static_friction=0.5, 
                restitution=0.1):

        # create all available terrain types
        num_terains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.005  # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16)

        def new_sub_terrain(): 
            return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

        heightfield[0:num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(), 
                                            max_height=max_height, 
                                            min_size=min_size, 
                                            max_size=max_size,
                                            num_rects=num_obs).height_field_raw

        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)

        position = np.array([-terrain_width/2.0, terrain_length/2.0, 0]) + position

        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])
        return add_terrain_to_stage(stage=self._stage, 
                    vertices=vertices, 
                    triangles=triangles, 
                    position=position, 
                    orientation=orientation,
                    prim_path=self._prim_path,
                    static_friction=dynamic_friction,
                    dynamic_friction=static_friction,
                    restitution=restitution)

    def post_reset(self):

        a = 1
        
    def get_observations(self):

        pass

    def calculate_metrics(self) -> None:

        pass

    def is_done(self) -> None:

        pass
