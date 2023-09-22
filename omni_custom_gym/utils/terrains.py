import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import numpy as np

from omni_custom_gym.utils.terrain_utils import *

from pxr import Usd

class RlTerrains():

    def __init__(self, 
                stage: Usd.Stage):
        
        self._stage = stage

    def get_wave_terrain(self, 
            terrain_size = 40,
            num_waves = 10, 
            amplitude = 1, 
            position = np.array([0.0, 0.0, 0.0])):

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

        add_terrain_to_stage(stage=self._stage, 
                    vertices=vertices, 
                    triangles=triangles,
                    position=position, 
                    orientation=orientation)
    
    def get_sloped_terrain(self, 
                    terrain_size = 40,
                    slope = -0.5, 
                    position = np.array([0.0, 0.0, 0.0])):

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

        add_terrain_to_stage(stage=self._stage, 
                    vertices=vertices, 
                    triangles=triangles,
                    position=position, 
                    orientation=orientation)
    
    def get_stairs_terrain(self, 
                terrain_size = 40,
                step_width = 0.75, 
                step_height = -0.5, 
                position = np.array([0.0, 0.0, 0.0])):

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

        add_terrain_to_stage(stage=self._stage, 
                    vertices=vertices, 
                    triangles=triangles,
                    position=position, 
                    orientation=orientation)

    def get_random_terrain(self, 
                terrain_size = 40, 
                min_height = -0.2, 
                max_height = 0.2, 
                step = 0.2, 
                downsampled_scale=0.5, 
                position = np.array([0.0, 0.0, 0.0])):

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

        heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), 
                                            min_height=min_height, max_height=max_height, 
                                            step=step, 
                                            downsampled_scale=downsampled_scale).height_field_raw
        
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, 
                                    horizontal_scale=horizontal_scale,
                                    vertical_scale=vertical_scale, 
                                    slope_threshold=1.5)

        position = np.array([-terrain_width/2.0, terrain_length/2.0, 0]) + position

        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

        add_terrain_to_stage(stage=self._stage, 
                    vertices=vertices, 
                    triangles=triangles,
                    position=position, 
                    orientation=orientation)
    
    def get_obstacles_terrain(self, 
                    terrain_size = 40.0, 
                    num_obs = 50, 
                    max_height = 0.5,
                    min_size = 0.5,
                    max_size = 5.0,
                    position = np.array([0.0, 0.0, 0.0])):

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
        add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position, orientation=orientation)

    def post_reset(self):

        a = 1
        
    def get_observations(self):

        pass

    def calculate_metrics(self) -> None:

        pass

    def is_done(self) -> None:

        pass
