import bpy

# Create a new sun light datablock
sun_light_data = bpy.data.lights.new(name="NewSunLight", type='SUN')
sun_light_data.energy = 5  # Sun light usually requires less energy value as it's quite strong

# Create a new object with the sun light datablock
sun_light_object = bpy.data.objects.new(name="NewSunLight", object_data=sun_light_data)

# Link light object to the collection in the current context scene
bpy.context.collection.objects.link(sun_light_object)

# Set light location and rotation
sun_light_object.location = (10, 10, 10)
sun_light_object.rotation_euler = (0.785, 0.785, 0)

