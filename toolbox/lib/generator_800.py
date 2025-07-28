# Wind Turbine Synthetic Vision
# Copyright (C) 2025 Arash Shahirpour, Jakob Gebler, Manuel Sanders
# Institute of Automatic Control - RWTH Aachen University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import random
import math
import os
import mathutils
from typing import List

import numpy as np
import blenderproc as bproc
from enum import Enum
import bpy
import bpy_extras
import cv2
from PIL import Image

import helper as helper


class Parameter(Enum):
    MIN_DIST = 0
    MAX_DIST = 800
    MIN_HEIGHT = 10
    MAX_HEIGHT = 260
    ANGLE = 0
    X_RES = 640
    Y_RES = 480
    ROTATION_ANGLE_HOUSING_MIN = 0
    ROTATION_ANGLE_HOUSING_MAX = 0
    ROTATION_ANGLE_ROTOR_MIN = 0
    ROTATION_ANGLE_ROTOR_MAX = 360
    RANDOM_CAMERA_SETTINGS = False
    FIXED_CAMERA_LENS = False
    LENS_MM = 3.4
    SENSOR_HEIGHT = 5.3
    CAMERA_DISTANCE_MIN = 80
    CAMERA_DISTANCE_MAX = 800

class DatasetGenerator:
    def __init__(
        self,
        parameter: Parameter,
        number_images: int,
        yolo_scene_file_path_abs: str,
        output_paths: helper.OutputPaths,
        background_images_path_abs: str,
        output_path_abs: str,
    ):
        self.MIN_DIST = parameter.MIN_DIST.value
        self.MAX_DIST = parameter.MAX_DIST.value
        self.MIN_HEIGHT = parameter.MIN_HEIGHT.value
        self.MAX_HEIGHT = parameter.MAX_HEIGHT.value
        self.ANGLE = parameter.ANGLE.value
        self.X_RES = parameter.X_RES.value
        self.Y_RES = parameter.Y_RES.value
        self.ROTATION_ANGLE_HOUSING_LOW = parameter.ROTATION_ANGLE_HOUSING_MIN.value
        self.ROTATION_ANGLE_HOUSING_MAX = parameter.ROTATION_ANGLE_HOUSING_MAX.value
        self.ROTATION_ANGLE_ROTOR_MIN = parameter.ROTATION_ANGLE_ROTOR_MIN.value
        self.ROTATION_ANGLE_ROTOR_MAX = parameter.ROTATION_ANGLE_ROTOR_MAX.value

        self.CAMERA_DISTANCE_MIN = parameter.CAMERA_DISTANCE_MIN.value
        self.CAMERA_DISTANCE_MAX = parameter.CAMERA_DISTANCE_MAX.value
        self.RANDOM_CAMERA_SETTINGS = parameter.RANDOM_CAMERA_SETTINGS.value
        self.FIXED_CAMERA_LENS = parameter.FIXED_CAMERA_LENS.value
        self.LENS_MM = parameter.LENS_MM.value
        self.SENSOR_HEIGHT = parameter.SENSOR_HEIGHT.value

        self.path_scene = yolo_scene_file_path_abs
        self.number_images = number_images
        self.output_paths = output_paths
        self.background_images_path_abs = background_images_path_abs
        self.output_path_abs = output_path_abs
        bproc.renderer.set_render_devices("OPTIX")

        bproc.init()

        self.objs = bproc.loader.load_blend(self.path_scene)

        helper.create_world_nodes()

        bpy.context.scene.cycles.feature_set = "EXPERIMENTAL"
        bpy.context.scene.render.engine = "CYCLES"

        self.camera = bpy.data.objects["Camera"].data

        # Setze die Sensor-Breite und -Höhe
            #Halbe Sensorhöhe: 1536 pixel * 3.45 um/pixel * 0.5
            #Rotor Radius: 60 m
            #Abstand: 120 m
        #Maximale Brennweite = Abstand / Radius * Halbe Sensorhöhe = 5.3 mm
        self.camera.sensor_fit = "VERTICAL"
        self.camera.sensor_height = self.SENSOR_HEIGHT

        bproc.camera.set_resolution(self.X_RES, self.Y_RES)
        bproc.camera.set_intrinsics_from_blender_params(
            lens=None,
            image_width=None,
            image_height=None,
            clip_start=None,
            clip_end=100000,
            pixel_aspect_x=None,
            pixel_aspect_y=None,
            shift_x=None,
            shift_y=None,
            lens_unit=None,
        )

    def generate(self, outputpaths: helper.OutputPaths = None):
        output_paths = outputpaths or self.output_paths

        distance_list = []
        lens_list = []

        for i in range(self.number_images):
            print(f"Generating image {i+1} of {self.number_images}")
            bproc.utility.reset_keyframes()

            # helper.randomization_sun()
            helper.randomization_material()
            wea_selection = helper.generate_wea_set(self.MIN_DIST, self.MAX_DIST)

            helper.rotate_housing(
                wea_selection,
                mean_angle=random.uniform(0, 360),
                std_deviation=5.0,
                normal_distributed=True,
            )
            helper.rotate_rotor(
                wea_selection,
                self.ROTATION_ANGLE_ROTOR_MIN,
                self.ROTATION_ANGLE_ROTOR_MAX,
            )
            scaling_factors = helper.scale_shaft(wea_selection, scale_random=True)


            # sample camera distance and sensor lens with rejection sampling
            def rejection_sample(d_min, d_max, l_min, l_max):
                d = np.random.uniform(d_min, d_max)
                l = np.random.uniform(l_min, l_max)

                turbine_radius = 60
                sensor_height = 5.3

                l_max_deala = (abs(d) / turbine_radius) * (sensor_height / 2.0)

                if l > l_max_deala:
                    return rejection_sample(d_min, d_max, l_min, l_max)

                return d, l

            if not self.FIXED_CAMERA_LENS:
                distance, lens = rejection_sample(self.CAMERA_DISTANCE_MIN, self.CAMERA_DISTANCE_MAX, 3.0, 55.0)
                self.camera.lens = lens
            else:
                distance = random.uniform(self.CAMERA_DISTANCE_MIN, self.CAMERA_DISTANCE_MAX)
                self.camera.lens = self.LENS_MM
           
            distance_list.append(distance)
            lens_list.append(self.camera.lens)

            position, euler_rotation = self.set_camera(
                distance=distance,
                height_min=self.MIN_HEIGHT,
                height_max=self.MAX_HEIGHT,
                pitch_centered=True,
                housing_height=89,
                angle_bound=0,
            )

            scene = bpy.context.scene
            camera = bpy.context.scene.camera

            # implement 80/20 training/test split and set output path for the image and keypoints
            random_number = random.random()
            if random_number < 0.8:
                path_image = os.path.join(
                    output_paths["training"]["path_images"], f"{i:05d}.png"
                )
                path_keypoints = os.path.join(
                    output_paths["training"]["path_keypoints"], f"{i:05d}.txt"
                )
                path_image_keypoints = os.path.join(
                    output_paths["training"]["path_images_keypoints"], f"{i:05d}.png"
                )
            else:
                path_image = os.path.join(
                    output_paths["validation"]["path_images"], f"{i:05d}.png"
                )
                path_keypoints = os.path.join(
                    output_paths["validation"]["path_keypoints"], f"{i:05d}.txt"
                )
                path_image_keypoints = os.path.join(
                    output_paths["validation"]["path_images_keypoints"], f"{i:05d}.png"
                )

            keypoints = self.create_yolo_text_file(
                path_keypoints,
                wea_selection,
                camera,
                scene,
                scaling_factors,
            )

            # only the components of the front wind turbine are made visible.
            for obj in bpy.context.scene.objects:
                obj.hide_render = True
            for wea in wea_selection:
                wea.obj_all.hide_render = False
                wea.obj_housing.hide_render = False
                wea.obj_rotor_wrapper.hide_render = False
                wea.obj_rotor_all.hide_render = False
                wea.obj_rotor_middle.hide_render = False
                wea.obj_rotor_tip_1.hide_render = False
                wea.obj_rotor_tip_2.hide_render = False
                wea.obj_rotor_tip_3.hide_render = False
                wea.obj_housing_back.hide_render = False

            # randomly select a background image from the background folder
            background_img = os.path.join(
                self.background_images_path_abs,
                random.choice(os.listdir(self.background_images_path_abs)),
            )
            background_img = Image.open(background_img).convert("RGBA")
            background_img = background_img.resize((self.X_RES, self.Y_RES))

            if random.random() < 0.1:
                background_img = np.random.uniform(
                    0, 256, (self.X_RES, self.Y_RES, 3)
                ).astype(np.uint8)
                background_img = Image.fromarray(background_img).convert("RGBA")
                background_img = background_img.resize((self.X_RES, self.Y_RES))

            helper.custom_properties_eins()
            bproc.renderer.set_output_format(enable_transparency=True)
            bproc.renderer.set_noise_threshold(0.01)  

            # Render RGB images
            data = bproc.renderer.render()

            # foreground to pil image
            foreground_img = Image.fromarray(data["colors"][0]).convert("RGBA")
            foreground_img = helper.shift_hue_random(foreground_img)
            background_img = helper.shift_hue_random(background_img)

            # paste foreground on background
            background_img.paste(foreground_img, mask=foreground_img)
            # background_img = foreground_img

            # add random noise add some random level
            if random.random() < 0.4:
                sigma = random.randint(1, 8)
                background_img = helper.add_gaussian_noise(
                    np.array(background_img), sigma=sigma
                )
            else:
                background_img = np.array(background_img)

            # Save image with low JPEG quality
            # random compression quality between 1 and 100
            # 50 % of time do jpg compression
            if random.random() < 0.4:
                compression_quality = random.randint(45, 100)
                _, encoded_img = cv2.imencode(
                    ".jpg",
                    background_img,
                    [cv2.IMWRITE_JPEG_QUALITY, compression_quality],
                )
                compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            else:
                compressed_img = background_img

            cv2.imwrite(path_image, cv2.cvtColor(compressed_img, cv2.COLOR_RGBA2BGRA))

            self.draw_keypoints(keypoints, path_image, path_image_keypoints)

        import matplotlib.pyplot as plt
        import datetime

        # plot ds and ls in a headmap plot
        plt.scatter(distance_list, lens_list, s=2, alpha=0.2)
        plt.xlabel("Camera Distance in m")
        plt.ylabel("Camera Lens in mm") 
        plt.title("Camera Distance and Lens Scatter Plot")
        plt.savefig(f"{self.output_path_abs}/distance_lens_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    def set_camera(
        self,
        distance: float,
        height_min: int = 55,
        height_max: int = 90,
        pitch_centered: bool = False,
        angle_bound: int = 5,
        housing_height: int = 50,
    ):
        """The distance, height and viewing angle of the camera can be set. The wind turbine is always positioned at coordinate 0.0.

        Args:
            distance_min (int, optional): minimal distance of the camera to 0,0. Defaults to 120.
            distance_max (int, optional): maximal distance of the camera to 0,0. Defaults to 220.
            height_min (int, optional): minimal height of the camera in perspective to the wind turbine. Defaults to 55.
            height_max (int, optional): maximal height of the camera in perspective to the wind turbine. Defaults to 90.
            pitch_centered (bool, optional): whether or not the camera should be aimed at the wind turbine housing. The height of the housing must also be provided. Defaults to False.
            angle_bound (int, optional): if pitch_centered is false, this angle is used as the minimum and maximum limit for all yaw, pitch and roll angles in degree. Defaults to 5.
            housing_height (int, optional): if pitch_centered is true, is used to adjust the camera pitch angle to look at the housing. Defaults to 50.
        """
        height = random.uniform(height_min, height_max)

        if pitch_centered:
            delta = height - housing_height
            angle_x = math.pi / 2 - math.atan(
                delta / distance
            )  # pitch angle of the camera
            angle_y = math.radians(np.random.normal(loc=0.0, scale=3)) + math.radians(
                random.uniform(-angle_bound, angle_bound)
            )  # roll angle of the camera
            angle_z = math.radians(
                random.uniform(-angle_bound, angle_bound)
            )  # yaw angle of the camera rotation
        else:
            angle_x = math.pi / 2 + math.radians(
                random.uniform(-angle_bound, angle_bound)
            )
            angle_y = math.radians(random.uniform(-angle_bound, angle_bound))
            angle_z = math.radians(random.uniform(-angle_bound, angle_bound))

        position = [0, -distance, height]
        euler_rotation = [angle_x, angle_y, angle_z]

        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(matrix_world)

        return position, euler_rotation

    def draw_keypoints(
        self,
        keypoints: List[List[float]],
        input_image_path: str,
        output_image_path: str,
    ):
        image_new = cv2.imread(input_image_path)

        # Define colors for each keypoint
        colors = [
            (255, 0, 0),    # Red - shaft bottom
            (0, 255, 0),    # Green - shaft top  
            (0, 0, 255),    # Blue - housing back
            (255, 255, 0),  # Yellow - rotor middle
            (255, 0, 255),  # Magenta - rotor tip 1 (top)
            (0, 255, 255),  # Cyan - rotor tip 2 (right)
            (255, 128, 0)   # Orange - rotor tip 3 (left)
        ]

        for keypoint in keypoints:
            pos_start = (
                int(self.X_RES * (keypoint[1] - keypoint[3] / 2)),
                int(self.Y_RES * (keypoint[2] - keypoint[4] / 2)),
            )
            pos_end = (
                int(pos_start[0] + keypoint[3] * self.X_RES),
                int(pos_start[1] + keypoint[4] * self.Y_RES),
            )

            image_new = cv2.rectangle(image_new, pos_start, pos_end, (255, 0, 0), 1)

            for i in range(int(len(keypoint[5:]) / 2)):
                image_new = cv2.circle(
                    image_new,
                    (
                        int(keypoint[5 + 2 * i] * self.X_RES),
                        int(keypoint[6 + 2 * i] * self.Y_RES),
                    ),
                    2,
                    colors[i],
                    -1,
                )

        cv2.imwrite(output_image_path, image_new)

    def create_yolo_text_file(
        self,
        output_path,
        wea_selection,
        camera,
        scene,
        rotor_scaling_factors: List[float] = None,
    ):
        return_keypoints = []

        if rotor_scaling_factors is None:
            rotor_scaling_factors = [1.0] * len(wea_selection)

        if len(rotor_scaling_factors) != len(wea_selection):
            raise ValueError(f"The number of rotor scaling factors must be equal to the number of WEAs. scaling factors: {len(rotor_scaling_factors)}, WEAs: {len(wea_selection)}")

        with open(output_path, "a") as file:
            for wea, rotor_scaling_factor in zip(wea_selection, rotor_scaling_factors):
                key_points_all = []

                key_points_all.append(0)
                Werte = []
                x_Werte = []
                y_Werte = []

                P2 = np.array(wea.obj_rotor_wrapper.location)
                RZ = [
                    [
                        math.cos(wea.obj_housing.rotation_euler[2]),
                        -math.sin(wea.obj_housing.rotation_euler[2]),
                        0,
                    ],
                    [
                        math.sin(wea.obj_housing.rotation_euler[2]),
                        math.cos(wea.obj_housing.rotation_euler[2]),
                        0,
                    ],
                    [0, 0, 1],
                ]
                RY = [
                    [
                        math.cos(wea.obj_rotor_wrapper.rotation_euler[1]),
                        0,
                        math.sin(wea.obj_rotor_wrapper.rotation_euler[1]),
                    ],
                    [0, 1, 0],
                    [
                        -math.sin(wea.obj_rotor_wrapper.rotation_euler[1]),
                        0,
                        math.cos(wea.obj_rotor_wrapper.rotation_euler[1]),
                    ],
                ]

                RY_neg = [
                    [
                        math.cos(-wea.obj_rotor_wrapper.rotation_euler[1]),
                        0,
                        math.sin(-wea.obj_rotor_wrapper.rotation_euler[1]),
                    ],
                    [0, 1, 0],
                    [
                        -math.sin(-wea.obj_rotor_wrapper.rotation_euler[1]),
                        0,
                        math.cos(-wea.obj_rotor_wrapper.rotation_euler[1]),
                    ],
                ]

                RX = [
                    [1, 0, 0],
                    [
                        0,
                        math.cos(wea.obj_rotor_wrapper.rotation_euler[0]),
                        -math.sin(wea.obj_rotor_wrapper.rotation_euler[0]),
                    ],
                    [
                        0,
                        math.sin(wea.obj_rotor_wrapper.rotation_euler[0]),
                        math.cos(wea.obj_rotor_wrapper.rotation_euler[0]),
                    ],
                ]

                RZ = np.array(RZ)
                RY = np.array(RY)
                RX = np.array(RX)

                def rotate_tip(objekt):
                    Relativposition = (
                        np.array(objekt.location) - P2
                    ) * rotor_scaling_factor
                    X_Achse = RY_neg @ Relativposition
                    Drehung = RX @ X_Achse
                    Gedreht = RY @ Drehung
                    Kugel_neu = Gedreht + P2
                    co = RZ @ Kugel_neu
                    co = mathutils.Vector(co)
                    return co

                co_housing_back = (
                    mathutils.Vector(RZ @ np.array(wea.obj_housing_back.location))
                    + wea.obj_all.location
                )
                co_tower_top = (
                    mathutils.Vector(RZ @ np.array(wea.obj_tower_top.location))
                    + wea.obj_all.location
                )
                co_tower_bottom = (
                    mathutils.Vector(RZ @ np.array(wea.obj_tower_bottom.location))
                    + wea.obj_all.location
                )
                co_rotor_middle = (
                    mathutils.Vector(RZ @ np.array(wea.obj_rotor_middle.location))
                    + wea.obj_all.location
                )

                co_rotor_tip_1 = (
                    mathutils.Vector(rotate_tip(wea.obj_rotor_tip_1))
                    + wea.obj_all.location
                )
                co_rotor_tip_2 = (
                    mathutils.Vector(rotate_tip(wea.obj_rotor_tip_2))
                    + wea.obj_all.location
                )
                co_rotor_tip_3 = (
                    mathutils.Vector(rotate_tip(wea.obj_rotor_tip_3))
                    + wea.obj_all.location
                )

                co_2d_housing_back = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, co_housing_back
                )
                co_2d_tower_top = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, co_tower_top
                )
                co_2d_tower_bottom = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, co_tower_bottom
                )
                co_2d_rotor_middle = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, co_rotor_middle
                )
                co_2d_rotor_tip_1 = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, co_rotor_tip_1
                )
                co_2d_rotor_tip_2 = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, co_rotor_tip_2
                )
                co_2d_rotor_tip_3 = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, co_rotor_tip_3
                )

                if np.sign(co_2d_rotor_tip_1[2]) == -1:
                    co_2d_rotor_tip_1[0] = co_2d_rotor_middle[0] + (co_2d_rotor_middle[0] - co_2d_rotor_tip_1[0])
                    co_2d_rotor_tip_1[1] = co_2d_rotor_middle[1] + (co_2d_rotor_middle[1] - co_2d_rotor_tip_1[1])

                if np.sign(co_2d_rotor_tip_2[2]) == -1:
                    co_2d_rotor_tip_2[0] = co_2d_rotor_middle[0] + (co_2d_rotor_middle[0] - co_2d_rotor_tip_2[0])
                    co_2d_rotor_tip_2[1] = co_2d_rotor_middle[1] + (co_2d_rotor_middle[1] - co_2d_rotor_tip_2[1])

                if np.sign(co_2d_rotor_tip_3[2]) == -1:
                    co_2d_rotor_tip_3[0] = co_2d_rotor_middle[0] + (co_2d_rotor_middle[0] - co_2d_rotor_tip_3[0])
                    co_2d_rotor_tip_3[1] = co_2d_rotor_middle[1] + (co_2d_rotor_middle[1] - co_2d_rotor_tip_3[1])

                render_size = (
                    bpy.context.scene.render.resolution_x,
                    bpy.context.scene.render.resolution_y,
                )

                def f_0_0(m_x, m_y, x):
                    return x * m_y / m_x

                def f_0_1(m_x, m_y, x):
                    return (-x * (1 - m_y) / m_x) + 1

                def f_1_0(m_x, m_y, x):
                    return x * m_y / (m_x - 1) + m_y / (1 - m_x)

                def f_1_1(m_x, m_y, x):
                    return (x * (1 - m_y) / (1 - m_x)) + (m_y - m_x) / (1 - m_x)

                def compare_tip(rotor_middle, rotor_tip):
                    c_0_0 = f_0_0(
                        rotor_middle[0],
                        rotor_middle[1],
                        rotor_tip[0],
                    )
                    c_0_1 = f_0_1(
                        rotor_middle[0],
                        rotor_middle[1],
                        rotor_tip[0],
                    )
                    c_1_0 = f_1_0(
                        rotor_middle[0],
                        rotor_middle[1],
                        rotor_tip[0],
                    )
                    c_1_1 = f_1_1(
                        rotor_middle[0],
                        rotor_middle[1],
                        rotor_tip[0],
                    )
                    x_res = 0
                    y_res = 0

                    if c_0_0 > rotor_tip[1] and c_1_0 > rotor_tip[1] and rotor_tip[1] < 0:
                        y_res = 0
                        x_res = rotor_middle[0] - rotor_middle[1]*(rotor_middle[0]-rotor_tip[0])/(rotor_middle[1]-rotor_tip[1])
                    elif c_0_1 < rotor_tip[1] and c_1_1 < rotor_tip[1]:
                        y_res = 1
                        x_res = rotor_middle[0] + (1-rotor_middle[1])*(rotor_tip[0]-rotor_middle[0])/(rotor_tip[1]-rotor_middle[1])
                    elif c_0_0 < rotor_tip[1] and c_0_1 > rotor_tip[1]:
                        y_res = rotor_middle[1] - rotor_middle[0]*(rotor_tip[1]-rotor_middle[1])/(rotor_tip[0]-rotor_middle[0])
                        x_res = 0
                    elif c_1_0 < rotor_tip[1] and c_1_1 > rotor_tip[1]:
                        y_res = rotor_middle[1] + (1-rotor_middle[0])*(rotor_tip[1]-rotor_middle[1])/(rotor_tip[0]-rotor_middle[0])
                        x_res = 1

                    return x_res, y_res

                if co_2d_tower_bottom[1] < 0:
                    scaler = co_2d_tower_bottom[1] / (
                        co_2d_tower_top[1] - co_2d_tower_bottom[1]
                    )
                    co_2d_tower_bottom[1] = 0
                    co_2d_tower_bottom[0] = co_2d_tower_bottom[0] - scaler * (
                        co_2d_tower_top[0] - co_2d_tower_bottom[0]
                    )

                if co_2d_rotor_tip_1[0] < 0 or co_2d_rotor_tip_1[0] > 1 or co_2d_rotor_tip_1[1] < 0 or co_2d_rotor_tip_1[1] > 1:
                    co_2d_rotor_tip_1[0], co_2d_rotor_tip_1[1] = compare_tip(
                        co_2d_rotor_middle, co_2d_rotor_tip_1
                    )
                
                if co_2d_rotor_tip_2[0] < 0 or co_2d_rotor_tip_2[0] > 1 or co_2d_rotor_tip_2[1] < 0 or co_2d_rotor_tip_2[1] > 1:
                    co_2d_rotor_tip_2[0], co_2d_rotor_tip_2[1] = compare_tip(
                        co_2d_rotor_middle, co_2d_rotor_tip_2
                    )
                
                if co_2d_rotor_tip_3[0] < 0 or co_2d_rotor_tip_3[0] > 1 or co_2d_rotor_tip_3[1] < 0 or co_2d_rotor_tip_3[1] > 1:
                    co_2d_rotor_tip_3[0], co_2d_rotor_tip_3[1] = compare_tip(
                        co_2d_rotor_middle, co_2d_rotor_tip_3
                    )

                # Create a list of rotor tips and shuffle them
                rotor_tips = [
                    co_2d_rotor_tip_1,
                    co_2d_rotor_tip_2,
                    co_2d_rotor_tip_3,
                ]
                random.shuffle(rotor_tips)

                co_2d_all = [
                    co_2d_housing_back,
                    co_2d_tower_top,
                    co_2d_tower_bottom,
                    co_2d_rotor_middle,
                    *rotor_tips,  # Unpack the shuffled rotor tips
                ]
                for co_2d in co_2d_all:
                    pixel_x = round(co_2d[0] * (render_size[0] - 1))
                    pixel_y = round((1 - co_2d[1]) * (render_size[1] - 1))

                    if (
                        pixel_x >= self.X_RES
                        or pixel_x < 0
                        or pixel_y >= self.Y_RES
                        or pixel_y < 0
                    ):
                        Werte.append(0)
                        Werte.append(0)
                    else:
                        Werte.append(pixel_x / self.X_RES)
                        Werte.append(pixel_y / self.Y_RES)

                    if pixel_x < 0 and pixel_y < 0:
                        x_Werte.append(0)
                        y_Werte.append(0)

                    elif pixel_x < 0 and 0 <= pixel_y <= self.Y_RES:
                        x_Werte.append(0)
                        y_Werte.append(pixel_y / self.Y_RES)

                    elif pixel_x < 0 and pixel_y > self.Y_RES:
                        x_Werte.append(0)
                        y_Werte.append(1)

                    elif 0 <= pixel_x <= self.X_RES and pixel_y < 0:
                        x_Werte.append(pixel_x / self.X_RES)
                        y_Werte.append(0)

                    elif (
                        0 <= pixel_x <= self.X_RES and pixel_y > self.Y_RES
                    ):
                        x_Werte.append(pixel_x / self.X_RES)
                        y_Werte.append(1)

                    elif (
                        0 <= pixel_x <= self.X_RES
                        and 0 <= pixel_y <= self.Y_RES
                    ):
                        x_Werte.append(pixel_x / self.X_RES)
                        y_Werte.append(pixel_y / self.Y_RES)

                    elif (
                        pixel_x > self.X_RES and 0 <= pixel_y <= self.Y_RES
                    ):
                        x_Werte.append(1)
                        y_Werte.append(pixel_y / self.Y_RES)

                    elif pixel_x > self.X_RES and pixel_y > self.Y_RES:
                        x_Werte.append(1)
                        y_Werte.append(1)

                    elif pixel_x > self.X_RES and pixel_y < 0:
                        x_Werte.append(1)
                        y_Werte.append(0)

                max_x = max(x_Werte)
                min_x = min(x_Werte)
                max_y = max(y_Werte)
                min_y = min(y_Werte)

                mid_x = (max_x + min_x) / 2
                mid_y = (max_y + min_y) / 2
                height = max_y - min_y
                width = max_x - min_x

                # only write the bounding box is within the image
                if mid_x > 0 and mid_x < 1 and mid_y > 0 and mid_y < 1 and width > 0 and width < 1 and height > 0 and height < 1:
                    file.write(f"0 {mid_x} {mid_y} {width} {height} ")
                    key_points_all.append(mid_x)
                    key_points_all.append(mid_y)
                    key_points_all.append(width)
                    key_points_all.append(height)

                    for Wert in Werte:
                        file.write(f"{Wert} ")
                        key_points_all.append(Wert)

                    file.write("\n")
                    return_keypoints.append(key_points_all)

            return return_keypoints
