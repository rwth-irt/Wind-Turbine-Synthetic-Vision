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

import bpy
import random
import math
import numpy as np
from PIL import Image
import colorsys
import os
import datetime
from typing import TypedDict, NamedTuple, List, Callable
import mathutils
import bpy_types


class OutputPathsDict(TypedDict):
    path_images: str
    path_images_keypoints: str
    path_keypoints: str


class OutputPaths(TypedDict):
    training: OutputPathsDict
    validation: OutputPathsDict


class WeaPoints(NamedTuple):
    obj_all: bpy_types.Object
    obj_housing: bpy_types.Object
    obj_rotor_wrapper: bpy_types.Object
    obj_rotor_all: bpy_types.Object
    obj_rotor_middle: bpy_types.Object
    obj_rotor_tip_1: bpy_types.Object
    obj_rotor_tip_2: bpy_types.Object
    obj_rotor_tip_3: bpy_types.Object
    obj_housing_back: bpy_types.Object
    obj_tower_top: bpy_types.Object
    obj_tower_bottom: bpy_types.Object


# In Blender können Eigenschaften von Objekten und der Welt durch Nodes zusammengesetzt und eingestellt werden. Die Nodes werden aber beim Aufruf der Szene in
# BlenderProc fehlerhaft übernommen. Daher werden diese mithilfe der Funktionen 'create_world_nodes' und 'set_ocean_texture' neu erstellt. 'create_world_nodes'
# benutzt dabei die Funktion 'create_node', welche Nodes in Blender erzeugt.
def create_node(node_type, node_name, location):
    node = bpy.context.scene.world.node_tree.nodes.new(type=node_type)
    node.location = location
    node.name = node_name
    return node


# rekursive Funkiton, um alle Kindobjekte eines Objekts mit einer bestimmten Eigenschaft zu versehen (wird bei der nächsten Funktion benötigt)
def set_property(obj, property_name, property_value):
    obj[property_name] = property_value
    for child in obj.children:
        set_property(child, property_name, property_value)


# Die Funktion 'custom_properties_eins' setzt die Eigenschaft 'category_id' auf 0 für alle Objekte, die nicht 'shaft' im Namen haben. Für alle Objekte, die 'shaft' im Namen
# haben, wird die Eigenschaft auf 1 gesetzt. Für die Welt wird die Eigenschaft ebenfalls auf 0 gesetzt.
def custom_properties_eins():
    prop_name = "category_id"

    for obj in bpy.data.objects:
        obj[prop_name] = 0

    for obj in bpy.context.scene.objects:
        if "shaft_" in obj.name.lower():
            prop_value = 1
            if obj:
                set_property(obj, prop_name, prop_value)

    bpy.context.scene.world[prop_name] = 0


def create_world_nodes():
    # Überprüfe, ob eine Welt-Node-Gruppe vorhanden ist, erstelle eine falls nicht
    if bpy.context.scene.world.use_nodes and bpy.context.scene.world.node_tree is None:
        bpy.context.scene.world.node_tree = bpy.data.node_groups.new(
            type="ShaderNodeTree", name="MyWorldNodes"
        )

    # Da bei der Ausführung von BlenderProc eine zusätzliche Background und Output Node erstellt wird, werden diese zunächst gelöscht.
    world_tree = bpy.data.worlds["World"].node_tree
    for node in world_tree.nodes:
        if node.type in {"BACKGROUND", "OUTPUT_WORLD"}:
            world_tree.nodes.remove(node)

    # Hier beginnt die Erstellung und die Verknüpfung der eigenen Nodes:
    # Erstellen der Skytexture Node
    sky_texture = create_node("ShaderNodeTexSky", "Sky Texture", (0, 0))
    # Einstellungen für die Skytexture Node
    sky_texture.sky_type = "NISHITA"
    sky_texture.sun_disc = True
    sky_texture.sun_intensity = 10
    sky_texture.sun_size = 0.523599
    sky_texture.sun_elevation = 1.173
    sky_texture.sun_rotation = 3.22699
    sky_texture.air_density = 0.2
    sky_texture.dust_density = 0
    sky_texture.ozone_density = 5.07
    sky_texture.altitude = 0

    # Erstellen der AddShader Nodes
    add_shader = create_node("ShaderNodeAddShader", "Add Shader", (400, 0))

    # Verbinden des AddShader Nodes mit der Background Node
    bpy.context.scene.world.node_tree.links.new(
        add_shader.inputs[0], sky_texture.outputs["Color"]
    )

    # Erstellen und Verbinden der Output Node
    output_node = create_node("ShaderNodeOutputWorld", "Output", (600, 0))
    bpy.context.scene.world.node_tree.links.new(
        output_node.inputs["Surface"], add_shader.outputs["Shader"]
    )

    # Erstellen einer weiteren Background Node
    background2 = create_node("ShaderNodeBackground", "Background2", (200, -600))

    # Erstellen und Verbinden der ColorRamp Node
    color_ramp = create_node("ShaderNodeValToRGB", "Color Ramp", (-100, -600))
    color_ramp.color_ramp.interpolation = "LINEAR"
    color_ramp.color_ramp.elements[1].position = 1
    color_ramp.color_ramp.elements[0].position = 0
    bpy.context.scene.world.node_tree.links.new(
        background2.inputs["Color"], color_ramp.outputs["Color"]
    )

    # Erstellen und Verbinden der MusgraveTexture Node
    musgrave_texture = create_node(
        "ShaderNodeTexMusgrave", "Musgrave Texture", (-400, -600)
    )
    musgrave_texture.inputs[2].default_value = 1
    musgrave_texture.inputs[3].default_value = 10
    musgrave_texture.inputs[4].default_value = 0.3
    musgrave_texture.inputs[5].default_value = 2
    bpy.context.scene.world.node_tree.links.new(
        color_ramp.inputs["Fac"], musgrave_texture.outputs["Fac"]
    )

    # Erstellen und Verbinden der Gradient Texture Node
    gradient_texture = create_node(
        "ShaderNodeTexGradient", "Gradient Texture", (-200, -1200)
    )
    bpy.context.scene.world.node_tree.links.new(
        background2.inputs["Strength"], gradient_texture.outputs["Fac"]
    )

    # Erstellen und Verbinden der Mapping Node
    mapping_node = create_node("ShaderNodeMapping", "Mapping", (-400, -1200))
    mapping_node.inputs[2].default_value[1] = 1.5708
    bpy.context.scene.world.node_tree.links.new(
        gradient_texture.inputs["Vector"], mapping_node.outputs["Vector"]
    )

    # Erstellen der Texture Coordinate Node
    tex_coord = create_node("ShaderNodeTexCoord", "Texture Coordinate", (-600, -1200))
    bpy.context.scene.world.node_tree.links.new(
        mapping_node.inputs["Vector"], tex_coord.outputs["Generated"]
    )

    bpy.context.scene.world.node_tree.links.new(
        add_shader.inputs[1], background2.outputs["Background"]
    )

def brightness_distribution_sample_normal():
    return 1.0 + random.normalvariate(mu=0, sigma=0.3)


def shift_hue_random(image, brightness_distribution_sample: Callable[[], float] = brightness_distribution_sample_normal):
    """Shift the hue of a PIL RGBA image.
    Args:
        image (PIL.Image.Image): The input image.
    Returns:
        PIL.Image.Image: The shifted image.
    """
    hue_shift = random.normalvariate(mu=0, sigma=10)
    sat_shift = random.normalvariate(mu=0, sigma=10)
    # val_shift = 1 + abs(random.normalvariate(mu=0, sigma=0.1))
    val_factor = brightness_distribution_sample()

    # Convert hue shift to PIL's 0-255 scale
    h_shift = int((hue_shift / 360) * 256)
    s_shift = int((sat_shift / 100) * 255)

    # Split into RGB and Alpha channels
    r, g, b, a = image.split()
    rgb_img = Image.merge("RGB", (r, g, b))

    # Convert to HSV
    hsv_img = rgb_img.convert("HSV")
    h, s, v = hsv_img.split()

    # Apply adjustments with numpy
    h_np = (np.array(h) + h_shift) % 256  # Hue wrapping
    s_np = np.clip(np.array(s) + s_shift, 0, 255)  # Saturation clamping
    v_np = np.clip(np.array(v) * val_factor, 0, 255)  # Value clamping

    # Convert back to PIL Images
    h_new = Image.fromarray(h_np.astype(np.uint8), "L")
    s_new = Image.fromarray(s_np.astype(np.uint8), "L")
    v_new = Image.fromarray(v_np.astype(np.uint8), "L")

    # Rebuild and convert back to RGB
    hsv_adj = Image.merge("HSV", (h_new, s_new, v_new))
    rgb_adj = hsv_adj.convert("RGB")

    # Split new RGB and merge with original Alpha
    r_new, g_new, b_new = rgb_adj.split()
    return Image.merge("RGBA", (r_new, g_new, b_new, a))


def rotate_rotor(
    wea_selection: List[WeaPoints],
    angle_rotor_low: int = 0,
    angle_rotor_high: int = 119,
):
    """The rotate_rotor function adjusts the rotor blades of the wind turbine (WEA). Since each blade is assigned a fixed positional identifier (1 = top, 2 = right, 3 = left), the rotor rotation is restricted to angles <120° to ensure the blades maintain their positional assignments. The angle of rotation is sampled from a uniform distribution.
    The rotors are installed in relation to another coordinate system. The reason for this is that the rotors are slightly inclined and should also be rotated around their axis. This cannot be implemented in Blender without an additional coordinate system. The coordinate systems each have the name 'empty.XXX'. Now all objects containing the name 'empty' are called up. This means the coordinate systems on the rotors.

    Args:
        angle_rotor_low (int, optional): smallest angle that can be sampled from the distribution in degrees. Defaults to 0.
        angle_rotor_high (int, optional): largest angle that can be sampled from the distribution in degrees. Defaults to 119.
    """
    for wea in wea_selection:
        angle = random.uniform(angle_rotor_low, angle_rotor_high)
        wea.obj_rotor_wrapper.rotation_euler[0] = math.radians(angle)


def scale_shaft(
    wea_selection: List[WeaPoints], scale_random: bool = True, factor: float = 0.0
):
    """Scales the shaft of the wind turbine (WEA).

    Args:
        wea_selection (List[WeaPoints]): List of WeaPoints objects.
        scale_random (bool, optional): If True, the shaft is scaled randomly. Defaults to True.
        factor (float, optional): The factor by which the shaft is scaled when scale_random is False. Defaults to 0.0.
    """
    scaling_factors = []
    for wea in wea_selection:
        if scale_random:
            factor = random.uniform(0.5, 1.2)
            wea.obj_rotor_all.delta_scale = mathutils.Vector((1.0 * factor, 1.0 * factor, 1.0))
        else:
            factor = 1.0
            wea.obj_rotor_all.delta_scale = mathutils.Vector((1.0 * factor, 1.0 * factor, 1.0))
        scaling_factors.append(factor)

    return scaling_factors

def rotate_housing(
    wea_selection: List[WeaPoints],
    mean_angle=0,
    std_deviation=45 / 3,
    normal_distributed=True,
):
    """Rotarotates objects in a blender scene that start with 'housing_' by a random angle.
    The normal distribution for the rotation can be useful if there are several wind turbines on the scene and they should all point in the wind direction,
    with slight deviations.


    Args:
        mean_angle (int, optional): When normal_distributed this is the mean of the distribution. Defaults to 0.
        std_deviation (_type_, optional): When normal_distributed this is the standard deviation of the distribution. Defaults to 45/3.
        normal_distributed (bool, optional): when false, the function rotate each object uniform randomly. Defaults to True.
    """

    for wea in wea_selection:
        if normal_distributed:
            angle = random.gauss(mean_angle, std_deviation)
        else:
            angle = random.uniform(0, 360)
        wea.obj_housing.rotation_euler[2] = math.radians(angle)


# Die Funktion 'randomization_Wolken' verändert die Fülle der Wolken.
def randomization_Wolken():
    # Mit einer Wahrscheinlichkeit von 20% wird die Dichte der Wolken auf 0 gesetzt. Ansonsten wird die Dichte der Wolken zufällig zwischen 0.1 und 10 gewählt.
    if random.random() < 0.2:
        Dichte_Wolken = 0
    else:
        Dichte_Wolken = random.uniform(0.1, 10)

    # Die Dichte der Wolken wird in der Musgrave Texture Node eingestellt.
    musgrave_texture = bpy.data.worlds["World"].node_tree.nodes["Musgrave Texture"]
    musgrave_texture.inputs[2].default_value = math.radians(Dichte_Wolken)


def randomization_sky_texture(
    fog_intensity_min: float = 0.1,
    fog_intensity_max: float = 3.0,
    sun_elevation_min: int = 0,
    sun_elevation_max: int = 90,
    sun_rotation_min: int = 0,
    sun_rotation_max: int = 360,
):
    """changes the intensity of the fog and the position of the sun of a blender scene

    Args:
        fog_intensity_min (float, optional): minimum value for the fog intensity. Defaults to 0.1.
        fog_intensity_max (float, optional): maximum value for the fog intensity. Defaults to 3.0.
        sun_elevation_min (int, optional): solar height angle minimum. Defaults to 0.
        sun_elevation_max (int, optional): solar height angle maximum. Defaults to 90.
        sun_rotation_min (int, optional): azimuth angle minimum. Defaults to 0.
        sun_rotation_max (int, optional): azimuth angle maximum. Defaults to 360.
    """

    # The Sky_Texture Node is selected in order to manipulate it
    sky_texture = bpy.data.worlds["World"].node_tree.nodes["Sky Texture"]

    sky_texture.dust_density = random.uniform(fog_intensity_min, fog_intensity_max)
    sky_texture.sun_elevation = math.radians(
        random.uniform(sun_elevation_min, sun_elevation_max)
    )
    sky_texture.sun_rotation = math.radians(
        random.uniform(sun_rotation_min, sun_rotation_max)
    )
    sky_texture.sun_disc = False


def generate_wea_set(distance_close: int = 800, distance_far: int = 800):
    """Positions the wind turbines randomly within a specific area. The wind turbines are positioned in the x and y directions.

    Args:
        distance_close (int, optional): Minimum distance with random selection of distance. Defaults to 0.
        distance_far (int, optional): Maximum distance with random selection of distance. Defaults to 1000.
    """

    # Change the wind turbine in the foreground (pay attention to the naming of the wind turbine in Blender)
    return_objects = []

    # get all shaft objects
    shafts = [
        obj
        for obj in bpy.context.scene.objects
        if "shaft" in obj.name.lower()
        and "shaft_oben" not in obj.name.lower()
        and "shaft_unten" not in obj.name.lower()
    ]
    shafts_selected = random.sample(
        shafts, random.choice([1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
    )

    for obj in shafts_selected:
        y_koordinate = random.uniform(distance_close, distance_far)

        x_koordinate = random.uniform(20, y_koordinate)
        if random.random() < 0.5:
            x_koordinate = -x_koordinate

        obj.location[0] = x_koordinate
        obj.location[1] = y_koordinate

        if obj.children[0].rotation_euler[2] > math.pi:
            tip_2 = obj.children[0].children[0].children[0].children[2]
            tip_3 = obj.children[0].children[0].children[0].children[3]
        else:
            tip_2 = obj.children[0].children[0].children[0].children[3]
            tip_3 = obj.children[0].children[0].children[0].children[2]

        wea_point = WeaPoints(
            obj_all=obj,
            obj_housing=obj.children[0],
            obj_rotor_wrapper=obj.children[0].children[0],
            obj_rotor_all=obj.children[0].children[0].children[0],
            obj_rotor_middle=obj.children[0].children[0].children[0].children[0],
            obj_rotor_tip_1=obj.children[0].children[0].children[0].children[1],
            obj_rotor_tip_2=tip_2,
            obj_rotor_tip_3=tip_3,
            obj_housing_back=obj.children[0].children[1],
            obj_tower_top=obj.children[1],
            obj_tower_bottom=obj.children[2],
        )
        return_objects.append(wea_point)

    foreground_shaft = random.choice(return_objects)
    foreground_shaft.obj_all.location[0] = 0
    foreground_shaft.obj_all.location[1] = 0

    return return_objects


def randomization_material():
    """Blender materials are identified by numerical codes. This function selects materials using
    a shared prefix (consistent across all WEAs) and applies randomization to their visual
    properties (colors/textures) while preserving structural identifiers.
    """
    # Generiere neue Werte für den Base Color
    value_1 = random.uniform(0.7, 0.8)  # für Helligkeit
    saturation_1 = random.uniform(0, 0.1)  # für Sättigung
    hue_1 = random.random()  # für Hue zwischen 0 und 1
    metallic_1 = random.uniform(0, 0.35)  # für Metallic zwischen 0 und 0,5
    roughness_1 = random.uniform(0, 0.5)  # für Roughness zwischen 0 und 1
    for material in bpy.data.materials:
        # Überprüfe, ob das Material das gewünschte Muster im Namen hat
        if "191" in material.name:
            # Auswählen der Base Color Node
            if material.node_tree:
                for node in material.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        rgb_color = colorsys.hsv_to_rgb(hue_1, saturation_1, value_1)
                        # Ändere den Wert des Base Color-Eingangs
                        node.inputs["Base Color"].default_value = (
                            rgb_color[0],
                            rgb_color[1],
                            rgb_color[2],
                            1.0,
                        )
                        # Passe den Metallic-Wert an
                        node.inputs["Metallic"].default_value = metallic_1
                        # Passe den Roughness-Wert an
                        node.inputs["Roughness"].default_value = roughness_1
                        node.inputs["Emission"].default_value = (0, 0, 0, 1)

        if "204" in material.name:
            # Auswählen der Base Color Node
            if material.node_tree:
                for node in material.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        # Generiere neue Werte für den Base Color
                        value = random.uniform(0.3, 0.7)  # für Helligkeit
                        saturation = random.uniform(0.8, 1)  # für Sättigung
                        hue = 1
                        metallic = random.uniform(
                            0, 0.5
                        )  # für Metallic zwischen 0 und 0,5
                        roughness = random.random()  # für Roughness zwischen 0 und 1
                        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
                        # Ändere den Wert des Base Color-Eingangs
                        node.inputs["Base Color"].default_value = (
                            rgb_color[0],
                            rgb_color[1],
                            rgb_color[2],
                            1.0,
                        )
                        # Passe den Metallic-Wert an
                        node.inputs["Metallic"].default_value = metallic
                        # Passe den Roughness-Wert an
                        node.inputs["Roughness"].default_value = roughness

        if "225" in material.name:
            # Auswählen der Base Color Node
            if material.node_tree:
                for node in material.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        # Generiere neue Werte für den Base Color
                        value = random.uniform(0.2, 0.7)  # für Helligkeit
                        saturation = random.uniform(0, 1)  # für Sättigung
                        hue = 0.167
                        metallic = random.uniform(
                            0, 0.5
                        )  # für Metallic zwischen 0 und 0,5
                        roughness = random.random()  # für Roughness zwischen 0 und 1
                        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
                        # Ändere den Wert des Base Color-Eingangs
                        node.inputs["Base Color"].default_value = (
                            rgb_color[0],
                            rgb_color[1],
                            rgb_color[2],
                            1.0,
                        )
                        # Passe den Metallic-Wert an
                        node.inputs["Metallic"].default_value = metallic
                        # Passe den Roughness-Wert an
                        node.inputs["Roughness"].default_value = roughness

    # nun wird die Dicke aller Bauteiler aller WEA angepasst.
    shaft_objects = [
        obj
        for obj in bpy.data.objects
        if "shaft" in obj.name.lower()
        and "shaft_oben" not in obj.name.lower()
        and "shaft_unten" not in obj.name.lower()
    ]
    for obj in shaft_objects:
        # Überprüfe und entferne vorhandene Solidify-Modifier vom Hauptobjekt
        for modifier in obj.modifiers:
            if modifier.type == "SOLIDIFY":
                obj.modifiers.remove(modifier)

        # Überprüfe und entferne vorhandene Solidify-Modifier vom ersten Kind
        if obj.children:
            for modifier in obj.children[0].modifiers:
                if modifier.type == "SOLIDIFY":
                    obj.children[0].modifiers.remove(modifier)

            # Überprüfe und entferne vorhandene Solidify-Modifier vom ersten Enkel des ersten Kindes
            if obj.children[0].children:
                for modifier in obj.children[0].children[0].modifiers:
                    if modifier.type == "SOLIDIFY":
                        obj.children[0].children[0].modifiers.remove(modifier)

                # Überprüfe und entferne vorhandene Solidify-Modifier vom ersten Urenkel des ersten Enkels
                if obj.children[0].children[0].children:
                    for modifier in obj.children[0].children[0].children[0].modifiers:
                        if modifier.type == "SOLIDIFY":
                            obj.children[0].children[0].children[0].modifiers.remove(
                                modifier
                            )

    # Füge den Solidify-Modifier zum Hauptobjekt hinzu
    solidify_modifier = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    solidify_modifier.offset = 0
    solidify_modifier.thickness = random.uniform(0, 1500)

    solidify_modifier = obj.children[0].modifiers.new(name="Solidify", type="SOLIDIFY")
    solidify_modifier.offset = 0
    solidify_modifier.thickness = random.uniform(0, 1500)

    solidify_modifier = (
        obj.children[0]
        .children[0]
        .children[0]
        .modifiers.new(name="Solidify", type="SOLIDIFY")
    )
    solidify_modifier.offset = 0
    solidify_modifier.thickness = random.uniform(0, 1500)


def add_gaussian_noise(image: np.array, mean=0, sigma=25) -> np.array:
    """Add Gaussian noise to an image.

    Args:
        image: Input image (numpy array)
        mean: Mean of Gaussian distribution
        sigma: Standard deviation of Gaussian distribution

    Returns:
        Noisy image (numpy array)
    """

    image = image.astype(np.float32) / 255.0

    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma / 255.0, image.shape).astype(np.float32)

    # Add noise to image
    noisy_image = image + noise

    # Clip values to [0, 1] range and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)

    return noisy_image


def create_noise_image(image: np.array, mean=0, sigma=25) -> np.array:
    """Add Gaussian noise to an image.

    Args:
        image: Input image (numpy array)
        mean: Mean of Gaussian distribution
        sigma: Standard deviation of Gaussian distribution

    Returns:
        Noisy image (numpy array)
    """

    # Generate Gaussian noise
    noise = np.random.uniform(0, 255, image.shape).astype(np.float32)

    return noise


def get_output_paths(pre_path_str: str, base_path: str) -> OutputPaths:
    # Function to get and create ourput dir structure for the image generation.
    # The folder structure is defined by OutputPaths.
    # The path structure get created inside a 'data' dir from base_path.

    new_folder_name_uuid = (
        pre_path_str + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    )
    new_folder_path_abs = os.path.join(base_path, "data/" + new_folder_name_uuid)

    path_images = os.path.join(new_folder_path_abs, "images")
    path_images_keypoints = os.path.join(new_folder_path_abs, "label_vis")
    path_keypoints = os.path.join(new_folder_path_abs, "labels")

    path_images_train = os.path.join(path_images, "train")
    path_images_keypoints_train = os.path.join(path_images_keypoints, "train")
    path_keypoints_train = os.path.join(path_keypoints, "train")

    path_images_test = os.path.join(path_images, "val")
    path_images_keypoints_test = os.path.join(path_images_keypoints, "val")
    path_keypoints_test = os.path.join(path_keypoints, "val")

    if not os.path.exists(new_folder_path_abs):
        os.makedirs(new_folder_path_abs)
        os.makedirs(path_images)
        os.makedirs(path_images_keypoints)
        os.makedirs(path_keypoints)
        os.makedirs(path_images_train)
        os.makedirs(path_images_keypoints_train)
        os.makedirs(path_keypoints_train)
        os.makedirs(path_images_test)
        os.makedirs(path_images_keypoints_test)
        os.makedirs(path_keypoints_test)
    else:
        raise ValueError("Pathname already exists.")

    return new_folder_path_abs,OutputPaths(
        training=OutputPathsDict(
            path_images=path_images_train,
            path_images_keypoints=path_images_keypoints_train,
            path_keypoints=path_keypoints_train,
        ),
        validation=OutputPathsDict(
            path_images=path_images_test,
            path_images_keypoints=path_images_keypoints_test,
            path_keypoints=path_keypoints_test,
        ),
    )
