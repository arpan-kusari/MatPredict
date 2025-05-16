import bpy
import os
from math import radians  # Ensure radians is imported for angle conversion
import math
import mathutils
# import sys
# sys.path.append("/home/yuzhen/Desktop/CVPR/blender_script")
# from change_cube_texture import change_target_object_material


def setup_camera_pos():
    # Clear existing cameras
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()

    # Define camera positions and orientations
    camera_positions = [(0, 0, 10), (5, 5, 5), (10, 0, 5)]
    camera_rotations = [(0, 0, 0), (55, 0, 135), (60, 0, 90)]

    # Create a camera at each position
    for position, rotation in zip(camera_positions, camera_rotations):
        bpy.ops.object.camera_add(location=position)
        camera = bpy.context.object
        camera.rotation_euler = [radians(r) for r in rotation]
        camera.name = f"Camera_{position[0]}_{position[1]}_{position[2]}"


def render_camera_view(camera):
    # Set render resolution
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    # Set the storage path
    output_path = '/home/yuzhen/Desktop/CVPR/blender_script/screenshoot'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Render and save images
    bpy.context.scene.camera = camera
    camera_position = camera.location
    camera_rotation = camera.rotation_euler
    file_name = f"room0_render_{camera_position[0]:.2f}_{camera_position[1]:.2f}.png"
    bpy.context.scene.render.filepath = os.path.join(output_path, file_name)
    bpy.ops.render.render(write_still=True)

    print(f"rendered and save the image to {output_path}")


def detect_camera_pos():
    # Get all cameras in the scene
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']

    if cameras:
        for camera in cameras:
            # Get the position and rotation
            position = camera.location
            rotation = camera.rotation_euler

            # Print the position and rotation
            print("Camera Position:")
            print(f"X: {position.x}, Y: {position.y}, Z: {position.z}")

            print("Camera Rotation (Euler):")
            print(f"X: {rotation.x}, Y: {rotation.y}, Z: {rotation.z}")
    else:
        print("No active camera in the scene.")


def find_target_object_position(target_object_name):
    object_name = target_object_name

    # Get the object
    obj = bpy.data.objects.get(object_name)
    if obj:
        # Get the object's center coordinates
        center_x = obj.location.x
        center_y = obj.location.y
        center_z = obj.location.z

        print(f"The center coordinates of the object '{object_name}' are: X={center_x}, Y={center_y}, Z={center_z}")
    else:
        print(f"Object '{object_name}' not found")


def change_target_object_material(target_object_name, material_type, material_index):
    object_name = target_object_name
    obj = bpy.data.objects.get(object_name)

    texture_name = f"{material_type}_{material_index}"
    base_color_path = f"/home/yuzhen/Desktop/CVPR/blender_script/texture/{texture_name}/basecolor.png"
    metallic_path = f"/home/yuzhen/Desktop/CVPR/blender_script/texture/{texture_name}/metallic.png"
    normal_path = f"/home/yuzhen/Desktop/CVPR/blender_script/texture/{texture_name}/normal.png"
    roughness_path = f"/home/yuzhen/Desktop/CVPR/blender_script/texture/{texture_name}/roughness.png"
    opacity_path = f"/home/yuzhen/Desktop/CVPR/blender_script/texture/{texture_name}/opacity.png"
    specular_path = f"/home/yuzhen/Desktop/CVPR/blender_script/texture/{texture_name}/specular.png"

    if obj:
        # Create a new material
        new_mat = bpy.data.materials.new(name=f"Material_{material_type}_{material_index}")
        new_mat.use_nodes = True
        nodes = new_mat.node_tree.nodes
        links = new_mat.node_tree.links

        # Clear the default nodes
        for node in nodes:
            nodes.remove(node)

        # Add a Principled BSDF shader node
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = 400, 0
        x_position = -350

        # ---- step1: load basecolor ----
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.location = x_position, 0
        if os.path.exists(base_color_path):
            texture_node.image = bpy.data.images.load(base_color_path)
        links.new(texture_node.outputs['Color'], shader.inputs['Base Color'])

        # ---- step2: load metallic ----
        texture_node_metallic = nodes.new('ShaderNodeTexImage')
        texture_node_metallic.location = x_position, -300
        if os.path.exists(metallic_path):
            texture_node_metallic.image = bpy.data.images.load(metallic_path)
        links.new(texture_node_metallic.outputs['Color'], shader.inputs['Metallic'])

        # ---- step3: load normal map ----
        texture_node_normal = nodes.new('ShaderNodeTexImage')
        texture_node_normal.location = x_position, -600
        if os.path.exists(normal_path):
            texture_node_normal.image = bpy.data.images.load(normal_path)
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.location = 300, -900
        links.new(texture_node_normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], shader.inputs['Normal'])

        # ---- step4: load roughness ----
        texture_node_roughness = nodes.new('ShaderNodeTexImage')
        texture_node_roughness.location = x_position, -900
        if os.path.exists(roughness_path):
            texture_node_roughness.image = bpy.data.images.load(roughness_path)
        links.new(texture_node_roughness.outputs['Color'], shader.inputs['Roughness'])

        # ---- step5: load alpha ----
        texture_node_opacity = nodes.new('ShaderNodeTexImage')
        texture_node_opacity.location = x_position, -1200
        if os.path.exists(opacity_path):
            texture_node_opacity.image = bpy.data.images.load(opacity_path)
        links.new(texture_node_opacity.outputs['Color'], shader.inputs['Alpha'])

        # ---- step6: load specular ----
        texture_node_specular = nodes.new('ShaderNodeTexImage')
        texture_node_specular.location = x_position, -1500
        if os.path.exists(specular_path):
            texture_node_specular.image = bpy.data.images.load(specular_path)
        links.new(texture_node_specular.outputs['Color'], shader.inputs['Specular Tint'])

        # --- Insert Texture Coordinate and Mapping ---
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (-800, 0)
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-600, 0)
        mapping.inputs['Scale'].default_value = (1, 1, 1.0)

        # Link UV → Mapping
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])

        # Link Mapping → Vector input of every texture node
        for tex_node in (
            texture_node, texture_node_metallic,
            texture_node_normal, texture_node_roughness,
            texture_node_opacity, texture_node_specular
        ):
            if 'Vector' in tex_node.inputs:
                links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])

        # Replace the object's material with the new material
        if obj.data.materials:
            links.new(shader.outputs['BSDF'], output_node.inputs['Surface'])
            obj.data.materials[0] = new_mat  # Replace the first material slot
        else:
            obj.data.materials.append(new_mat)  # Add the new material

        print("Connecting BSDF to Material Output")
        print(f"Material of '{object_name}' replaced with 'NewMaterial'.")
    else:
        print(f"Object '{object_name}' not found.")

    return texture_name


def camera_pipeline_sphere(
    target_object_name: str,
    camera_name: str,
    radius: float,
    output_dir: str,
    num_lat: int = 4,
    num_lon: int = 8,
    material_type: str = "default",
    material_index: int = 0
):
    """
    Place a camera on the surface of a sphere centered at the target object
    with a given radius. Cameras are distributed on a (num_lat × num_lon)
    latitude/longitude grid. Each camera points toward the target center and
    renders an image.

    Args:
        target_object_name: Name of the target object.
        camera_name: Name of an existing camera in the scene.
        radius: Distance from the camera to the target center.
        output_dir: Directory to save rendered images.
        num_lat: Number of latitude slices in [0, π].
        num_lon: Number of longitude slices in [0, 2π].
        material_type, material_index: Used only for file naming.
    """
    # Get the target object and camera
    target_object = bpy.data.objects.get(target_object_name)
    if not target_object:
        raise ValueError(f"Cannot find target object: {target_object_name}")

    camera = bpy.data.objects.get(camera_name)
    if not camera:
        raise ValueError(f"Cannot find camera: {camera_name}")

    # Clear camera animation (if not needed)
    camera.animation_data_clear()

    # Center of target object
    center = target_object.location

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    import math
    index = 0

    # Restrict phi and theta to the front hemisphere if desired
    phi_min, phi_max = 0, math.pi / 2
    theta_min, theta_max = -math.pi / 2, math.pi / 2  # Front half-ring only

    for i in range(num_lat):
        # Sample phi within (phi_min, phi_max)
        phi = phi_min + (i + 0.5) * (phi_max - phi_min) / num_lat

        for j in range(num_lon):
            # Sample theta within (theta_min, theta_max)
            theta = theta_min + (j + 0.5) * (theta_max - theta_min) / num_lon

            # Spherical → Cartesian
            x = center.x + radius * math.sin(phi) * math.cos(theta)
            y = center.y + radius * math.sin(phi) * math.sin(theta)
            z = center.z + radius * math.cos(phi)

            # Set camera position
            camera.location = mathutils.Vector((x, y, z))

            # Make the camera look at the target
            direction = center - camera.location
            camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

            # Build output file path
            file_name = f"{index}.png"
            index += 1
            output_path = os.path.join(output_dir, file_name)

            # Render
            bpy.context.scene.camera = camera
            bpy.context.scene.render.filepath = output_path
            bpy.ops.render.render(write_still=True)

            print(f"Rendered: lat={i}, lon={j}, saved to {output_path}")


def create_area_light(
    name="MyAreaLight",
    location=(0, 0, 0),
    rotation=(0, 0, 0),
    energy=1000.0,
    color=(1.0, 1.0, 1.0),
    size=2.0,
    shape='SQUARE'
):
    """
    Create an Area Light in Blender with the specified properties.

    :param name: Name of the new light object
    :param location: (x, y, z) coordinate for the light
    :param rotation: (rotX, rotY, rotZ) in degrees
    :param energy: Light intensity (Power)
    :param color: (R, G, B) light color, each in [0..1]
    :param size: Dimension of the area light
    :param shape: 'SQUARE', 'DISK', 'RECTANGLE', or 'ELLIPSE'
    :return: The newly created light object
    """

    # 1) Create a new area light data block
    light_data = bpy.data.lights.new(name=name + "_Data", type='AREA')
    light_data.energy = energy
    light_data.color = color
    light_data.shape = shape
    light_data.size = size

    # 2) Create a new object using that light data
    light_obj = bpy.data.objects.new(name, light_data)

    # 3) Set location and rotation
    light_obj.location = location
    light_obj.rotation_mode = 'XYZ'
    light_obj.rotation_euler = (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2])
    )

    # 4) Link the object to the scene
    bpy.context.collection.objects.link(light_obj)
    bpy.context.view_layer.objects.active = light_obj
    light_obj.select_set(True)

    return light_obj


def set_separated_background(
    camera_color=(0.2, 0.3, 0.8, 1.0),
    env_color=(0.8, 0.8, 0.8, 1.0)
):
    """
    In the World node tree, create a Mix Shader that combines:
    - env_color for environment lighting (non-camera rays)
    - camera_color for rays seen by the camera
    The mix factor is driven by Light-Path → Is Camera Ray.
    """
    # Ensure the scene has a World and nodes are enabled
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    world.use_nodes = True

    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    # Clear existing nodes
    for n in list(nodes):
        nodes.remove(n)

    # 1) Light Path node
    lp = nodes.new(type='ShaderNodeLightPath')
    lp.location = (-400, 0)

    # 2) Mix Shader node
    mix = nodes.new(type='ShaderNodeMixShader')
    mix.location = (-100, 0)

    # 3) Environment Background
    env_bg = nodes.new(type='ShaderNodeBackground')
    env_bg.location = (-300, 200)
    env_bg.inputs['Color'].default_value = env_color
    env_bg.inputs['Strength'].default_value = 0.1

    # 4) Camera-visible Background
    cam_bg = nodes.new(type='ShaderNodeBackground')
    cam_bg.location = (-300, -200)
    cam_bg.inputs['Color'].default_value = camera_color
    cam_bg.inputs['Strength'].default_value = 1.0

    # 5) World Output
    out = nodes.new(type='ShaderNodeOutputWorld')
    out.location = (100, 0)

    # 6) Link nodes
    links.new(lp.outputs['Is Camera Ray'], mix.inputs['Fac'])
    links.new(env_bg.outputs['Background'], mix.inputs[1])
    links.new(cam_bg.outputs['Background'], mix.inputs[2])
    links.new(mix.outputs['Shader'], out.inputs['Surface'])


def main():
    # Example background setup
    set_separated_background(
        camera_color=(0, 1.0, 0.0, 1.0),
        env_color=(0.8, 0.8, 0.8, 1.0)
    )

    # Example material list
    mat_name = ["leather"]
    mat_index = ["1"]

    target_object_name = "swivel_chair"

    for index in range(len(mat_name)):
        material_type = mat_name[index]
        material_index = int(mat_index[index])

        print(material_type, material_index)
        texture_name = change_target_object_material(target_object_name, material_type, material_index)

        camera_pipeline_sphere(
            target_object_name=target_object_name,
            camera_name="Camera",
            radius=2.0,
            output_dir=f"/home/yuzhen/Desktop/CVPR/blender_script/screenshoot/Feb_2/{texture_name}",
            num_lat=4,
            num_lon=4,
            material_type=material_type,
            material_index=material_index
        )


if __name__ == "__main__":
    main()

