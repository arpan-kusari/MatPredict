import blenderproc as bproc
import os
import json

# Initialize BlenderProc
bproc.init()

# ----------------------------- Path settings -----------------------------
room_0_path = "/home/yuzhen/Desktop/CVPR/Replica-Dataset/mini_test_dataset/room_0"       # Change to your actual room_0 path
ply_file = os.path.join(room_0_path, "/home/yuzhen/Desktop/CVPR/Replica-Dataset/mini_test_dataset/room_0/mesh.ply")            # PLY file
texture_folder = os.path.join(room_0_path, "/home/yuzhen/Desktop/CVPR/Replica-Dataset/mini_test_dataset/room_0/textures")      # Folder that contains HDR textures
parameters_file = os.path.join(texture_folder, "/home/yuzhen/Desktop/CVPR/Replica-Dataset/mini_test_dataset/room_0/textures/parameters.json")  # parameters.json
output_dir = "/home/yuzhen/Desktop/CVPR/blender_script/result_test"                      # Directory for rendered outputs

# ----------------------------- Load scene -----------------------------
obj = bproc.loader.load_obj(ply_file)          # Load the PLY mesh

# Load parameters.json
with open(parameters_file, "r") as f:
    parameters = json.load(f)

# ----------------------------- Assign materials & textures -----------------------------
hdr_files = [
    os.path.join(texture_folder, f"{i}-color-ptex.hdr") for i in range(parameters["splitSize"])
]

for i, obj in enumerate(bproc.object.get_objects()):
    # Make sure the number of objects matches the number of HDR files
    if i < len(hdr_files):
        material = obj.new_material(f"Material_{i}")
        material.use_nodes = True
        node_tree = material.get_node_tree()

        # Create an image-texture node and load the HDR file
        texture_node = node_tree.new_node("ShaderNodeTexImage")
        texture_node.image = bproc.loader.load_image(hdr_files[i])

        # Link the texture node to the BSDF node’s Base Color
        bsdf_node = node_tree.get_principled_bsdf()
        node_tree.link(texture_node.outputs["Color"], bsdf_node.inputs["Base Color"])

# ----------------------------- World background -----------------------------
# Use one HDR as the environment background (optional)
bproc.world.set_world_background_hdr_img(hdr_files[0])

# ----------------------------- Camera poses -----------------------------
positions = [
    {"location": [-5, 5, 2],  "rotation": bproc.math.build_transformation_matrix([0, 0, 0], [1, 0, 0])},
    {"location": [5, -5, 2],  "rotation": bproc.math.build_transformation_matrix([0, 0, 0], [0, 1, 0])},
]
for position in positions:
    bproc.camera.add_camera_pose(
        bproc.math.build_transformation_matrix(position["location"], position["rotation"])
    )

# Camera intrinsics
bproc.camera.set_intrinsics_from_blender_params(lens_mm=35, sensor_width_mm=32)

# ----------------------------- Renderer configuration -----------------------------
bproc.renderer.set_output_dir(output_dir)
bproc.renderer.enable_depth_output(True)
bproc.renderer.set_cycles_samples(128)      # Sampling rate
bproc.renderer.enable_normals_output()      # Output normal maps
bproc.renderer.set_denoiser("OPTIX")        # Enable denoising

# ----------------------------- Render -----------------------------
data = bproc.renderer.render()

# ----------------------------- Save results -----------------------------
bproc.writer.write_hdf5(output_dir, data)
print(f"Rendered results saved to: {output_dir}")

