import bpy
import os
import json
import math
 

# Specify the path to the PLY file
# ply_file_path = "/home/yuzhen/Desktop/CVPR/Replica-Dataset/mini_test_dataset/room_0/mesh_folder/mesh_semantic.ply_2.ply"

# # Specify the path to the PNG file
# base_color_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/basecolor.png"  # Modify this path to the location of your PNG file
# metallic_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/metallic.png" # Modify this path to the location of your metallic PNG file
# normal_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/normal.png"  # Modify this path to the location of your normal map PNG file
# roughness_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/roughness.png"  # Modify this path to the location of your roughness PNG file
# opacity_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/opacity.png"  # Path to the opacity PNG file
# specular_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/specular.png"

 

 




def get_full_texture_path(base_path, relative_path):
    print("-----------------------------------------------------------------")
    print (base_path)
    print(relative_path)
    print("-----------------------------------------------------------------")
    return os.path.join(base_path, relative_path.strip("/")) if relative_path else ''


def import_one_ply_and_change_material(curr_object_info, base_path, mat_index):
    print("enter the helper function")
    ply_file_num = len(curr_object_info["ply_file_path"])
    print(f"for this material, there are totally {ply_file_num} num of ply file")

    for ply_index in range (ply_file_num):
        ply_file_path =  curr_object_info["ply_file_path"][ply_index - 1]
        print(f"current ply file input is: {ply_file_path}")
        # Import the PLY object
        bpy.ops.wm.ply_import(filepath=ply_file_path)
        imported_object = bpy.context.selected_objects[0]
        imported_object.name = f"Object_{mat_index}_{ply_index}"

        """add the smart UV mapping"""
        # Set the imported object as the active object
        bpy.context.view_layer.objects.active = imported_object

        # Switch to Edit Mode
        bpy.ops.object.mode_set(mode='EDIT')

        # Select all faces
        bpy.ops.mesh.select_all(action='SELECT')

        # Perform Smart UV Project
        bpy.ops.uv.smart_project()

        # Switch back to Object Mode
        bpy.ops.object.mode_set(mode='OBJECT')


    
        base_color_path = get_full_texture_path (base_path,curr_object_info["textures"]["base_color"])
        metallic_path = get_full_texture_path(base_path, curr_object_info["textures"]["metallic"])
        normal_path = get_full_texture_path(base_path,  curr_object_info["textures"]["normal"])
        roughness_path = get_full_texture_path(base_path,  curr_object_info["textures"]["roughness"])
        opacity_path = get_full_texture_path(base_path,  curr_object_info["textures"]["opacity"])
        specular_path = get_full_texture_path(base_path,  curr_object_info["textures"]["specular"])


        # Create a new material
        mat = bpy.data.materials.new(name=f"Material_{mat_index}_{ply_index}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear the default nodes
        for node in nodes:
            nodes.remove(node)

        # Add a Principled BSDF shader node
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = 400, 0


        x_position = -350 



    
        #--------------step1: load the basecolor 
        # Create an image texture node
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.location = x_position, 0

        # Load the image file
        if os.path.exists(base_color_path):
            texture_node.image = bpy.data.images.load(base_color_path)

        # Connect the texture node to the BSDF shader
        links.new(texture_node.outputs['Color'], shader.inputs['Base Color'])



        #--------------step2: load the metallic  
        # Create an image texture node for metallic ------> metallic png 
        texture_node_metallic = nodes.new('ShaderNodeTexImage')
        texture_node_metallic.location = x_position, -300

        # Load the metallic image file
        if os.path.exists(metallic_path):
            texture_node_metallic.image = bpy.data.images.load(metallic_path)

        # Connect the metallic texture node to the BSDF shader
        links.new(texture_node_metallic.outputs['Color'], shader.inputs['Metallic'])



        #--------------step3: load the normal  
        # Create an image texture node for normal map
        texture_node_normal = nodes.new('ShaderNodeTexImage')
        texture_node_normal.location = x_position, -600

        # Load the normal map image file
        if os.path.exists(normal_path):
            texture_node_normal.image = bpy.data.images.load(normal_path)

        # Add a normal map node
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.location = 300, -900

        # Connect the normal map image to the normal map node
        links.new(texture_node_normal.outputs['Color'], normal_map.inputs['Color'])

        # Connect the normal map node to the BSDF shader
        links.new(normal_map.outputs['Normal'], shader.inputs['Normal'])





        #--------------step4: load the normal  
        # Create an image texture node for roughness
        texture_node_roughness = nodes.new('ShaderNodeTexImage')
        texture_node_roughness.location = x_position, -900

        # Load the roughness image file
        if os.path.exists(roughness_path):
            texture_node_roughness.image = bpy.data.images.load(roughness_path)

        # Connect the roughness texture node to the BSDF shader
        links.new(texture_node_roughness.outputs['Color'], shader.inputs['Roughness'])



        #--------------step5: load the alpha 
        # Create an image texture node for opacity
        texture_node_opacity = nodes.new('ShaderNodeTexImage')
        texture_node_opacity.location = x_position, -1200  # Adjust position as needed

        # Load the opacity image file
        if os.path.exists(opacity_path):
            texture_node_opacity.image = bpy.data.images.load(opacity_path)

        # Connect the opacity texture node to the BSDF shader
        links.new(texture_node_opacity.outputs['Color'], shader.inputs['Alpha'])




        #--------------step6: load the specular
        # Create an image texture node for specular
        texture_node_specular = nodes.new('ShaderNodeTexImage')
        texture_node_specular.location = x_position, -1500  # Adjust position as needed

        # Load the specular image file
        if os.path.exists(specular_path):
            texture_node_specular.image = bpy.data.images.load(specular_path)

        # Connect the specular texture node to the BSDF shader
        links.new(texture_node_specular.outputs['Color'], shader.inputs['Specular Tint'])



        

        # Connect the BSDF shader to the material output node
        links.new(shader.outputs['BSDF'], output_node.inputs['Surface'])

        # Assign the material to the cube
        imported_object.data.materials.append(mat)



def change_target_object_material(target_object_name,material_type, material_index):
    object_name = target_object_name
    obj = bpy.data.objects.get(object_name)

    base_color_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/basecolor.png"  # Modify this path to the location of your PNG file
    metallic_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/metallic.png" # Modify this path to the location of your metallic PNG file
    normal_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/normal.png"  # Modify this path to the location of your normal map PNG file
    roughness_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/roughness.png"  # Modify this path to the location of your roughness PNG file
    opacity_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/opacity.png"  # Path to the opacity PNG file
    specular_path = "/home/yuzhen/Desktop/CVPR/blender_script/texture/specular.png"

 

    if obj:
        
        # Create a new material
        new_mat = bpy.data.materials.new(name=f"Material_{material_type}_{material_index}")
        new_mat.use_nodes = True
        nodes = new_mat.node_tree.nodes
        links = new_mat.node_tree.links

        # Clear the default
        for node in nodes:
            nodes.remove(node)

        # Add a Principled BSDF shader node
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = 400, 0
        x_position = -350 


        #--------------step1: load the basecolor 
        # Create an image texture node
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.location = x_position, 0

        # Load the image file
        if os.path.exists(base_color_path):
            texture_node.image = bpy.data.images.load(base_color_path)

        # Connect the texture node to the BSDF shader
        links.new(texture_node.outputs['Color'], shader.inputs['Base Color'])

         #--------------step2: load the metallic  
        # Create an image texture node for metallic ------> metallic png 
        texture_node_metallic = nodes.new('ShaderNodeTexImage')
        texture_node_metallic.location = x_position, -300

        # Load the metallic image file
        if os.path.exists(metallic_path):
            texture_node_metallic.image = bpy.data.images.load(metallic_path)

        # Connect the metallic texture node to the BSDF shader
        links.new(texture_node_metallic.outputs['Color'], shader.inputs['Metallic'])



        #--------------step3: load the normal  
        # Create an image texture node for normal map
        texture_node_normal = nodes.new('ShaderNodeTexImage')
        texture_node_normal.location = x_position, -600

        # Load the normal map image file
        if os.path.exists(normal_path):
            texture_node_normal.image = bpy.data.images.load(normal_path)

        # Add a normal map node
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.location = 300, -900

        # Connect the normal map image to the normal map node
        links.new(texture_node_normal.outputs['Color'], normal_map.inputs['Color'])

        # Connect the normal map node to the BSDF shader
        links.new(normal_map.outputs['Normal'], shader.inputs['Normal'])





        #--------------step4: load the normal  
        # Create an image texture node for roughness
        texture_node_roughness = nodes.new('ShaderNodeTexImage')
        texture_node_roughness.location = x_position, -900

        # Load the roughness image file
        if os.path.exists(roughness_path):
            texture_node_roughness.image = bpy.data.images.load(roughness_path)

        # Connect the roughness texture node to the BSDF shader
        links.new(texture_node_roughness.outputs['Color'], shader.inputs['Roughness'])



        #--------------step5: load the alpha 
        # Create an image texture node for opacity
        texture_node_opacity = nodes.new('ShaderNodeTexImage')
        texture_node_opacity.location = x_position, -1200  # Adjust position as needed

        # Load the opacity image file
        if os.path.exists(opacity_path):
            texture_node_opacity.image = bpy.data.images.load(opacity_path)

        # Connect the opacity texture node to the BSDF shader
        links.new(texture_node_opacity.outputs['Color'], shader.inputs['Alpha'])




        #--------------step6: load the specular
        # Create an image texture node for specular
        texture_node_specular = nodes.new('ShaderNodeTexImage')
        texture_node_specular.location = x_position, -1500  # Adjust position as needed

        # Load the specular image file
        if os.path.exists(specular_path):
            texture_node_specular.image = bpy.data.images.load(specular_path)

        # Connect the specular texture node to the BSDF shader
        links.new(texture_node_specular.outputs['Color'], shader.inputs['Specular Tint'])




        # Replace the object's material with the new material
        if obj.data.materials:
            # Connect the BSDF shader to the material output node
            print("Connecting BSDF to Material Output")
            links.new(shader.outputs['BSDF'], output_node.inputs['Surface'])
            obj.data.materials[0] = new_mat  # Replace the first material slot
        else:
            print("Connecting BSDF to Material Output")
            obj.data.materials.append(new_mat)  # Add the new material

         
        print("Connecting BSDF to Material Output")
        print(f"Material of '{object_name}' replaced with 'NewMaterial'.")
    else:
        print(f"Object '{object_name}' not found.")


def show_only_object(target_name):
 
    for obj in bpy.data.objects:
        if obj.name == target_name:
            # show on the window 
            obj.hide_set(False)
            # show on the render
            obj.hide_render = False
        else:
            # hide on the window
            obj.hide_set(True)
            # hide on the render
            obj.hide_render = True


 

def main():
     
    # Create a new cube
    bpy.ops.mesh.primitive_cube_add(size=2)
    cube = bpy.context.object

    """json file path"""
    json_file_path = "/home/yuzhen/Desktop/CVPR/blender_script/ply_to_material.json"

    # Read the JSON file containing object info
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    base_path = data.get('base_path', '')
    print(base_path)
    objects_info = data.get('object', [])
    print(objects_info)

    for mat_index, curr_object_info in enumerate(objects_info):
        import_one_ply_and_change_material(curr_object_info, base_path, mat_index)


    show_only_object("Object_1_0")

if __name__ == "__main__":
    main()
