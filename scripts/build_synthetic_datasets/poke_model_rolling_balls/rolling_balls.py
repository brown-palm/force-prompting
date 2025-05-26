from mathutils import Vector
import bpy_extras  
import numpy as np
import bpy
import math
import random
import os
import json

BOWLING_BALL_PROBABILITY = 0.333  # Change this to control how often the bowling ball is used
ClONE_BOWLING_BALL_PROBABILITY = 0.2  # Change this to control how often the bowling ball is used
BOWLING_BALL_MASS = 4.0 # Mass of the bowling ball in kg

global RANDOMIZATION_PARAMS
RANDOMIZATION_PARAMS = {
    "INITIAL_FORCE_ANGLE" : None,
    "PIXEL_ANGLE" : None,
    "GENERATED_FORCE_STRENGTH" : None,
    "OUTPUT_PATH" : None,
    "BALL_GROUND_COORDS" : None,
    "BALL_PIXEL_COORDS" : None,
    "NUM_DISTRACTOR_BALLS": None,
    "DISTRACTOR_BALL_POSITIONS": [],
    "CAMERA_ZOOM_FACTOR": None,
    "GROUND_TEXTURE": None,
    "SUN_ENERGY": None
}

# Hard-coded path to texture files
TEXTURE_PATH = "scripts/build_synthetic_datasets/poke_model_rolling_balls/football_textures"  # Replace with your actual texture path
FOOTBALL_DIFFUSE = os.path.join(TEXTURE_PATH, "dirty_football_diff_4k.jpg")
FOOTBALL_NORMAL = os.path.join(TEXTURE_PATH, "dirty_football_nor_gl_4k.exr")
FOOTBALL_ROUGHNESS = os.path.join(TEXTURE_PATH, "dirty_football_rough_4k.exr")

# Base path for ground textures
GROUND_TEXTURE_BASE_PATH = "scripts/build_synthetic_datasets/poke_model_rolling_balls/ground_textures"

import glob


def compute_pixel_angle_from_video(scene):
    """Compute the 2D pixel movement angle between frame 1 and frame 100, based on screen-space pixel coordinates."""
    
    ball = bpy.data.objects.get(RANDOMIZATION_PARAMS["MOVING_BALL_NAME"])
    if not ball:
        print("Ball object not found for pixel angle computation")
        return

    # Find the camera
    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
    if not camera:
        print("Camera not found")
        return

    # Save current frame to restore later
    current_frame = scene.frame_current

    # Get pixel position at frame 1
    scene.frame_set(1)
    start_world_pos = ball.matrix_world.translation
    pixel_start = world_to_camera_view(scene, camera, start_world_pos)

    # Get pixel position at frame 100
    scene.frame_set(100)
    end_world_pos = ball.matrix_world.translation
    pixel_end = world_to_camera_view(scene, camera, end_world_pos)

    # Restore original frame
    scene.frame_set(current_frame)

    # Compute displacement in pixel space
    start_x, start_y = pixel_start
    end_x, end_y = pixel_end

    dx = -(start_x - end_x) # Flip X axis for screen space
    dy = start_y - end_y

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360

    # Save into RANDOMIZATION_PARAMS
    RANDOMIZATION_PARAMS["PIXEL_ANGLE"] = angle_deg

    print(f"Computed pixel movement angle: {angle_deg:.2f} degrees (pixels)")


def apply_ball_textures(obj, texture_folder=None):
    """Apply textures to the given ball object using files from a chosen texture folder."""
    if not obj:
        print("Warning: Target object not found")
        return False

    if "bowling_ball" in obj.name:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

        subsurf = obj.modifiers.new(name="Subsurf", type='SUBSURF')
        subsurf.levels = 2
        subsurf.render_levels = 2

        # Apply bowling ball (solid color and shiny)
        mat = bpy.data.materials.new(name="Bowling_Ball_Material")
        obj.data.materials.clear()
        obj.data.materials.append(mat)

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])

        # Choose a random solid color
        solid_colors = [
            (0.0, 0.0, 0.0, 1.0),      # Jet Black
            (0.1, 0.1, 0.5, 1.0),      # Deep Blue
            (0.6, 0.0, 0.0, 1.0),      # Dark Red
            (0.1, 0.4, 0.1, 1.0),      # Forest Green
            (0.4, 0.0, 0.4, 1.0),      # Purple
            (0.3, 0.3, 0.3, 1.0),      # Graphite Gray
            (0.8, 0.5, 0.0, 1.0),      # Bronze
            (1.0, 0.0, 0.0, 1.0),      # Bright Red
            (0.0, 0.0, 1.0, 1.0),      # Royal Blue
            (0.0, 1.0, 0.0, 1.0),      # Vivid Green
            (1.0, 1.0, 0.0, 1.0),      # Neon Yellow
            (1.0, 0.65, 0.0, 1.0),     # Vibrant Orange
            (0.6, 0.3, 0.8, 1.0),      # Soft Purple
            (0.8, 0.2, 0.6, 1.0),      # Hot Pink
            (0.5, 0.0, 1.0, 1.0),      # Electric Violet
            (0.0, 1.0, 1.0, 1.0),      # Aqua
            (1.0, 0.0, 1.0, 1.0),      # Magenta
            (0.2, 0.6, 0.9, 1.0),      # Sky Blue
            (0.7, 1.0, 0.1, 1.0),      # Lime Green
            (0.9, 0.9, 0.9, 1.0),      # Pearl White
            (0.5, 0.5, 0.5, 1.0),      # Metallic Gray
            (0.9, 0.75, 0.6, 1.0),     # Champagne
            (0.7, 0.1, 0.1, 1.0),      # Blood Red
            (0.2, 0.2, 0.2, 1.0),      # Charcoal
            (0.3, 0.6, 0.9, 1.0),      # Baby Blue
            (1.0, 0.4, 0.7, 1.0),      # Candy Pink
            (0.9, 0.4, 0.2, 1.0),      # Sunset Orange
            (0.7, 0.7, 0.2, 1.0),      # Mustard
            (0.4, 0.8, 0.2, 1.0),      # Avocado Green
            (0.6, 0.1, 0.8, 1.0),      # Royal Purple
            (0.1, 0.8, 0.5, 1.0),      # Mint Green
            (0.2, 0.2, 0.6, 1.0),      # Indigo
            (0.4, 0.0, 0.2, 1.0),      # Deep Maroon
            (0.8, 0.1, 0.4, 1.0),      # Raspberry
            (0.7, 0.3, 0.1, 1.0),      # Rust
            (0.5, 0.2, 0.0, 1.0),      # Mahogany
            (0.2, 0.5, 0.2, 1.0),      # Army Green
            (0.9, 0.8, 0.2, 1.0),      # Gold
            (0.6, 0.8, 1.0, 1.0),      # Ice Blue
            (1.0, 0.7, 0.7, 1.0),      # Rose Pink
            (0.5, 0.7, 0.1, 1.0),      # Olive
            (0.9, 0.6, 0.9, 1.0),      # Lavender
            (0.6, 0.6, 1.0, 1.0),      # Periwinkle
            (0.2, 0.4, 0.8, 1.0),      # Sapphire Blue
            (0.7, 0.0, 0.2, 1.0),      # Burgundy
            (0.0, 0.3, 0.6, 1.0),      # Navy Blue
            (0.4, 0.8, 0.7, 1.0),      # Turquoise
            (0.6, 1.0, 0.6, 1.0),      # Pale Green
            (0.8, 0.9, 1.0, 1.0),      # Light Sky Blue
        ]

        # we want half off the bowling balls to be jet black
        solid_colors = solid_colors + [(0.0, 0.0, 0.0, 1.0)] * 48

        base_color = random.choice(solid_colors)
        principled.inputs['Base Color'].default_value = base_color

        # Make it shiny
        principled.inputs['Roughness'].default_value = 0.05

        print(f"Applied bowling ball appearance with color {base_color}.")
        return True
    else:
        # Football texture (normal behavior)
        pass


    # Randomly pick a folder if not specified
    if texture_folder is None:
        subfolders = [f for f in os.listdir(TEXTURE_PATH) if os.path.isdir(os.path.join(TEXTURE_PATH, f))]
        texture_folder = random.choice(subfolders)

    pattern_path = os.path.join(TEXTURE_PATH, texture_folder)

    def find_texture(name_base):
        """Find the first matching file with the given base name regardless of extension."""
        matches = glob.glob(os.path.join(pattern_path, f"{name_base}.*"))
        return matches[0] if matches else None

    # Resolve texture file paths
    diffuse_path = find_texture("pattern")
    normal_path = find_texture("normal")
    roughness_path = find_texture("rough")

    # Create a unique material
    mat = bpy.data.materials.new(name=f"Football_Material_{texture_folder}")
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    if diffuse_path and os.path.exists(diffuse_path):
        tex_diffuse = nodes.new(type='ShaderNodeTexImage')
        tex_diffuse.location = (-300, 100)
        tex_diffuse.image = bpy.data.images.load(diffuse_path)
        # Ambient Occlusion node
        ao = nodes.new(type='ShaderNodeAmbientOcclusion')
        ao.location = (-500, 250)

        # Color ramp to remap AO for deeper shadows
        ao_ramp = nodes.new(type='ShaderNodeValToRGB')
        ao_ramp.location = (-300, 250)
        ao_ramp.color_ramp.interpolation = 'LINEAR'
        ao_ramp.color_ramp.elements[0].position = 0.3  # Pull blacks in
        ao_ramp.color_ramp.elements[1].position = 0.7  # Push whites out

        # Make black more black (optional): set element colors explicitly
        ao_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        ao_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        # Multiply AO with diffuse
        mix_ao = nodes.new(type='ShaderNodeMixRGB')
        mix_ao.blend_type = 'MULTIPLY'
        mix_ao.inputs['Fac'].default_value = 1.0
        mix_ao.location = (-100, 250)

        # Hook up
        links.new(ao.outputs['Color'], ao_ramp.inputs['Fac'])
        # Create a Hue/Saturation node to randomly shift color
        hue_sat = nodes.new(type='ShaderNodeHueSaturation')
        hue_sat.location = (-200, 100)

        # Random variation parameters
        hue_shift = random.uniform(0.85, 1.15)         # 1.0 = no shift
        sat_shift = random.uniform(0.9, 1.1)           # Saturation boost or drop
        val_shift = random.uniform(0.9, 1.1)           # Brightness boost or drop

        hue_sat.inputs['Hue'].default_value = hue_shift
        hue_sat.inputs['Saturation'].default_value = sat_shift
        hue_sat.inputs['Value'].default_value = val_shift

        # Link: Diffuse → Hue/Sat → (later into AO mix)
        links.new(tex_diffuse.outputs['Color'], hue_sat.inputs['Color'])
        links.new(hue_sat.outputs['Color'], mix_ao.inputs['Color1'])
        links.new(ao_ramp.outputs['Color'], mix_ao.inputs['Color2'])
        links.new(mix_ao.outputs['Color'], principled.inputs['Base Color'])
        
        print(f"Applied diffuse texture: {diffuse_path}")
    else:
        print(f"Warning: No diffuse texture found in {pattern_path}")

    if normal_path and os.path.exists(normal_path):
        tex_normal = nodes.new(type='ShaderNodeTexImage')
        tex_normal.location = (-300, -100)
        tex_normal.image = bpy.data.images.load(normal_path)
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (-100, -100)
        links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        print(f"Applied normal texture: {normal_path}")
    else:
        print(f"Warning: No normal texture found in {pattern_path}")

    if roughness_path and os.path.exists(roughness_path):
        tex_roughness = nodes.new(type='ShaderNodeTexImage')
        tex_roughness.location = (-300, -300)
        tex_roughness.image = bpy.data.images.load(roughness_path)
        links.new(tex_roughness.outputs['Color'], principled.inputs['Roughness'])
        print(f"Applied roughness texture: {roughness_path}")
    else:
        print(f"Warning: No roughness texture found in {pattern_path}")

    return True


def apply_ground_textures():
    """Apply randomly selected ground textures to the plane"""
    import os
    import random
    
    # Find the plane - typically named 'Plane' in Blender
    plane = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and (obj.name.startswith('Plane') or 'plane' in obj.name.lower()):
            plane = obj
            break
    
    if not plane:
        print("Warning: Plane object not found in scene")
        return False
    
    print(f"Found plane object: {plane.name}")
    
    # Get available ground texture directories
    ground_types = []
    try:
        ground_types = [d for d in os.listdir(GROUND_TEXTURE_BASE_PATH) 
                       if os.path.isdir(os.path.join(GROUND_TEXTURE_BASE_PATH, d))]
    except Exception as e:
        print(f"Error reading ground texture directories: {e}")
        return False
    
    if not ground_types:
        print(f"No ground texture directories found in {GROUND_TEXTURE_BASE_PATH}")
        return False
    
    # Randomly select a ground type
    selected_ground = random.choice(ground_types)
    texture_path = os.path.join(GROUND_TEXTURE_BASE_PATH, selected_ground, "textures")
    RANDOMIZATION_PARAMS["GROUND_TEXTURE"] = selected_ground.split(".blend")[0]
    
    print(f"Selected ground texture: {selected_ground}")
    
    # Improved texture file finding with different possible extensions
    diffuse_path = None
    normal_path = None
    roughness_path = None
    displacement_path = None
    
    try:
        # Get all files in the texture directory
        all_files = os.listdir(texture_path)
        
        # Find textures by identifying patterns in filenames
        for file in all_files:
            file_lower = file.lower()
            # Diffuse texture (always jpg)
            if "diff_4k" in file_lower and file_lower.endswith(".jpg"):
                diffuse_path = os.path.join(texture_path, file)
            
            # Normal texture (always exr)
            elif "nor_gl_4k" in file_lower and file_lower.endswith(".exr"):
                normal_path = os.path.join(texture_path, file)
            
            # Roughness texture (could be jpg or exr)
            elif "rough_4k" in file_lower:
                if file_lower.endswith(".jpg") or file_lower.endswith(".exr"):
                    roughness_path = os.path.join(texture_path, file)
            
            # Displacement texture (could be jpg or png)
            elif "disp_4k" in file_lower:
                if file_lower.endswith(".jpg") or file_lower.endswith(".png"):
                    displacement_path = os.path.join(texture_path, file)
    
    except Exception as e:
        print(f"Error finding texture files: {e}")
        return False
    
    # Print found textures for debugging
    print(f"Found diffuse: {os.path.basename(diffuse_path) if diffuse_path else 'None'}")
    print(f"Found normal: {os.path.basename(normal_path) if normal_path else 'None'}")
    print(f"Found roughness: {os.path.basename(roughness_path) if roughness_path else 'None'}")
    print(f"Found displacement: {os.path.basename(displacement_path) if displacement_path else 'None'}")
    
    # Get or create the material for the plane
    if len(plane.material_slots) == 0:
        # Create new material if none exists
        mat = bpy.data.materials.new(name="Ground_Material")
        plane.data.materials.append(mat)
    else:
        mat = plane.material_slots[0].material
        if not mat:
            mat = bpy.data.materials.new(name="Ground_Material")
            plane.material_slots[0].material = mat
    
    # Enable nodes for the material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create the principled BSDF shader
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    
    # Create output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    # Link principled to output
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Add texture coordinate and mapping nodes for better control
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)
    
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-600, 0)
    # Adjust scale to control texture tiling
    mapping.inputs['Scale'].default_value[0] = 5.0  # Scale X
    mapping.inputs['Scale'].default_value[1] = 5.0  # Scale Y
    
    # Link texture coordinates to mapping
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    
    # Create and link texture nodes for diffuse, normal, roughness, and displacement
    # Diffuse Texture
    if diffuse_path and os.path.exists(diffuse_path):
        tex_diffuse = nodes.new(type='ShaderNodeTexImage')
        tex_diffuse.location = (-400, 200)
        tex_diffuse.image = bpy.data.images.load(diffuse_path)
        links.new(mapping.outputs['Vector'], tex_diffuse.inputs['Vector'])
        links.new(tex_diffuse.outputs['Color'], principled.inputs['Base Color'])
        print(f"Applied ground diffuse texture: {diffuse_path}")
    else:
        # import pdb; pdb.set_trace()
        print(f"Warning: Ground diffuse texture not found")
    
    # Normal Texture
    if normal_path and os.path.exists(normal_path):
        tex_normal = nodes.new(type='ShaderNodeTexImage')
        tex_normal.location = (-400, 0)
        tex_normal.image = bpy.data.images.load(normal_path)
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (-200, 0)
        links.new(mapping.outputs['Vector'], tex_normal.inputs['Vector'])
        links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        print(f"Applied ground normal texture: {normal_path}")
    else:
        # import pdb; pdb.set_trace()
        print(f"Warning: Ground normal texture not found")
    
    # Roughness Texture
    if roughness_path and os.path.exists(roughness_path):
        tex_roughness = nodes.new(type='ShaderNodeTexImage')
        tex_roughness.location = (-400, -200)
        tex_roughness.image = bpy.data.images.load(roughness_path)
        links.new(mapping.outputs['Vector'], tex_roughness.inputs['Vector'])
        links.new(tex_roughness.outputs['Color'], principled.inputs['Roughness'])
        print(f"Applied ground roughness texture: {roughness_path}")
    else:
        # import pdb; pdb.set_trace()
        print(f"Warning: Ground roughness texture not found")
    
    # Displacement Texture
    if displacement_path and os.path.exists(displacement_path):
        tex_disp = nodes.new(type='ShaderNodeTexImage')
        tex_disp.location = (-400, -400)
        tex_disp.image = bpy.data.images.load(displacement_path)
        
        # Add a displacement node
        disp_node = nodes.new(type='ShaderNodeDisplacement')
        disp_node.location = (-200, -400)
        disp_node.inputs['Scale'].default_value = 0.05  # Adjust displacement strength
        
        links.new(mapping.outputs['Vector'], tex_disp.inputs['Vector'])
        links.new(tex_disp.outputs['Color'], disp_node.inputs['Height'])
        links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
        
        # Set up the plane for displacement
        # Subdivide the plane for better displacement detail
        if plane.modifiers.get("Subdivision") is None:
            subdiv = plane.modifiers.new(name="Subdivision", type='SUBSURF')
            subdiv.levels = 2
            subdiv.render_levels = 2
        
        # Enable displacement in material settings
        mat.cycles.displacement_method = 'BOTH'  # Using both displacement and bump
        
        print(f"Applied ground displacement texture: {displacement_path}")
    else:
        # import pdb; pdb.set_trace()
        print(f"Warning: Ground displacement texture not found")


    # After applying textures, ensure the plane has correct physics properties
    plane = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and (obj.name.startswith('Plane') or 'plane' in obj.name.lower()):
            plane = obj
            break
    
    if plane:
        # Make sure the plane is set as a rigid body collision object
        bpy.ops.object.select_all(action='DESELECT')
        plane.select_set(True)
        bpy.context.view_layer.objects.active = plane
        
        # Add rigid body if it doesn't exist
        if not plane.rigid_body:
            bpy.ops.rigidbody.object_add(type='PASSIVE')
        else:
            plane.rigid_body.type = 'PASSIVE'
        
        # Set collision shape to mesh for accurate surface
        plane.rigid_body.collision_shape = 'MESH'
        # Ensure sufficient friction to prevent ball from sliding too much
        plane.rigid_body.friction = 0.8
    
    return True


def randomize_camera_zoom():
    """Randomize the camera zoom by moving it along its view direction"""
    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
            
    if camera:
        # Get camera location and the point it's looking at
        cam_loc = camera.location.copy()
        # Get camera direction
        direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        direction.normalize()
        
        # Sample zoom factor (0 = no zoom, 1 = maximum zoom)
        zoom_factor = random.uniform(0.0, 0.6)
        RANDOMIZATION_PARAMS["CAMERA_ZOOM_FACTOR"] = zoom_factor
        
        # Calculate the new position
        # Find the distance to origin along the camera direction
        t = -cam_loc.dot(direction) / direction.dot(direction)
        max_travel = t * direction
        
        # Move camera toward the target based on zoom factor
        camera.location += max_travel * zoom_factor
        
        print(f"Camera zoom randomized with factor: {zoom_factor:.2f}")
    else:
        print("Warning: No camera found in scene")


def randomize_force_field_strength(scene):
    """Randomize the force field strength between 1 and 50"""

    current_frame = scene.frame_current

    force_field = bpy.data.objects.get("Force")

    APPLY_FORCE_AT_FRAME = 10 + 12
    STOP_APPLYING_FORCE_AFTER = 24

    if force_field and force_field.field:
        if current_frame < 3:
            # Disable force early on
            force_field.field.strength = 0

            ball = bpy.data.objects.get(RANDOMIZATION_PARAMS["MOVING_BALL_NAME"])
            if ball and ball.rigid_body:
                if current_frame == 1:
                    ball.rigid_body.kinematic = True

        elif current_frame == APPLY_FORCE_AT_FRAME:
            # Apply force
            force_field.field.strength = RANDOMIZATION_PARAMS['GENERATED_FORCE_STRENGTH']

            ball = bpy.data.objects.get(RANDOMIZATION_PARAMS["MOVING_BALL_NAME"])
            if ball and ball.rigid_body:
                ball.rigid_body.kinematic = False
                ball.rigid_body.type = 'ACTIVE'
                ball.rigid_body.collision_margin = 0.02

                # Set mass based on ball type
                if RANDOMIZATION_PARAMS["MOVING_BALL_NAME"] == "bowling_ball":
                    ball.rigid_body.mass = BOWLING_BALL_MASS
                else:
                    ball.rigid_body.mass = 1.0

            print(f"Force field strength applied: {RANDOMIZATION_PARAMS['GENERATED_FORCE_STRENGTH']}")

        elif current_frame >= APPLY_FORCE_AT_FRAME + STOP_APPLYING_FORCE_AFTER:
            # Disable force
            force_field.field.strength = 0


def update_output_path():
    """Update output path with randomized parameters"""
    # global OUTPUT_PATH
    if RANDOMIZATION_PARAMS["BALL_PIXEL_COORDS"] is None:
        RANDOMIZATION_PARAMS["BALL_PIXEL_COORDS"] = (0, 0)  # Default to (0, 0) if not set
    if RANDOMIZATION_PARAMS["PIXEL_ANGLE"] is None:
        RANDOMIZATION_PARAMS["PIXEL_ANGLE"] = 999.9
    
    if RANDOMIZATION_PARAMS["INITIAL_FORCE_ANGLE"] is not None and RANDOMIZATION_PARAMS["GENERATED_FORCE_STRENGTH"] is not None:
        # Generate the directory name based on parameters
        base_output_dir = bpy.context.scene.render.filepath
        base_output_dir = os.path.dirname(base_output_dir)
        if not base_output_dir.endswith('/'):
            base_output_dir += '/'
        
        # Create a unique folder name with parameters
        param_dir = f"angle_{RANDOMIZATION_PARAMS['INITIAL_FORCE_ANGLE']:.1f}_force_{RANDOMIZATION_PARAMS['GENERATED_FORCE_STRENGTH']:.2f}_coordx_{RANDOMIZATION_PARAMS['BALL_PIXEL_COORDS'][0]}_coordy_{RANDOMIZATION_PARAMS['BALL_PIXEL_COORDS'][1]}_pixangle_{RANDOMIZATION_PARAMS['PIXEL_ANGLE']:.1f}"
        RANDOMIZATION_PARAMS['OUTPUT_PATH'] = os.path.join(base_output_dir, param_dir)
        
        # Create the directory if it doesn't exist
        os.makedirs(RANDOMIZATION_PARAMS['OUTPUT_PATH'], exist_ok=True)
        
        # Set the new output path in Blender
        bpy.context.scene.render.filepath = os.path.join(RANDOMIZATION_PARAMS['OUTPUT_PATH'], "frame####")
        
        # Print output path for debugging
        print(f"Setting render output path to: {bpy.context.scene.render.filepath}")
        
        json_path = os.path.join(RANDOMIZATION_PARAMS['OUTPUT_PATH'], "params.json")
        with open(json_path, 'w') as f:
            json.dump(RANDOMIZATION_PARAMS, f, indent=4)
        print(f"Created params file at: {json_path}")
    else:
        print("Warning: Missing randomization parameters for output path")

def bake_physics():
    """Bake physics simulation"""
    # Clear any existing bake data
    bpy.ops.ptcache.free_bake_all()

    # Set the frame range for baking
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 130  # Match bash script end frame

    # Find the ball
    ball = bpy.data.objects.get(RANDOMIZATION_PARAMS["MOVING_BALL_NAME"])
    if ball and ball.rigid_body:
        # Make sure the ball's initial state is correct before baking
        # Start as kinematic to prevent it from moving
        ball.rigid_body.kinematic = True
        ball.rigid_body.keyframe_insert(data_path="kinematic", frame=1)
        
        # Then set it to switch to active at the force application frame
        force_frame = 10 + 12  # APPLY_FORCE_AT_FRAME
        ball.rigid_body.kinematic = False
        ball.rigid_body.keyframe_insert(data_path="kinematic", frame=force_frame)
        
        # Reset to ensure proper initial state
        ball.rigid_body.kinematic = True
    
    
    bpy.context.scene.frame_set(1)
   
    # Bake all dynamics in the scene
    bpy.ops.ptcache.bake_all(bake=True)

    print("Physics baking complete!")

def add_sun_light():
    # Create a new sun lamp
    sun = bpy.data.lights.new(name="Sun", type='SUN')
    sun.energy = 3.0  # Increase this value for a brighter light
    
    # Create a new object with the lamp
    sun_obj = bpy.data.objects.new(name="Sun", object_data=sun)
    
    # Link the object to the scene
    bpy.context.collection.objects.link(sun_obj)
    
    # Position the lamp (adjust as needed)
    sun_obj.rotation_euler = (math.radians(5), 0, math.radians(45))
    sun_obj.location = (5, 5, 10)
    
    return sun_obj

def enhance_existing_lights():
    """Increase the intensity of existing lights in the scene"""
    # Loop through all lights in the scene
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            print(f"Enhancing light: {obj.name}, Type: {obj.data.type}")
            
            # Increase the energy/strength of the light
            if obj.data.type == 'SUN':
                obj.data.energy = 4.0  # Increase from default (usually around 1.0)
            elif obj.data.type == 'POINT':
                obj.data.energy = 0.0  # Point lights typically need higher values
            elif obj.data.type == 'SPOT':
                obj.data.energy = 2000.0
                obj.data.spot_size = math.radians(60)  # Widen the spotlight angle
            elif obj.data.type == 'AREA':
                obj.data.energy = 500.0
                obj.data.size = 5.0  # Increase the size of area lights
            
            # For all light types, you can change the color to be brighter
            obj.data.color = (1.0, 1.0, 1.0)  # Pure white
            
            print(f"  - Increased energy to {obj.data.energy}")

def world_to_camera_view(scene, camera, world_point):
    """Convert world coordinates to camera view coordinates (2D) using bpy_extras utility function"""
    # Use the proper bpy_extras utility function
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, world_point)
    
    # Get render resolution with scale factor
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    
    # Convert normalized coordinates to pixel coordinates
    pixel_x = round(co_2d.x * render_size[0])
    #Change axis to match Blender's coordinate system
    pixel_y = round((1 - co_2d.y) * render_size[1])
    
    return (pixel_x, pixel_y)

def sample_from_quadrilateral(corners):
    """
    Sample a single point from the unit square [0,1]^2 and map it to a quadrilateral.
    
    Parameters:
    -----------
    corners : numpy.ndarray
        Four corner points of the quadrilateral as (x, y) coordinates in a (4,2) array.
        Order must be: [bottom-left, bottom-right, top-right, top-left]
    
    Returns:
    --------
    tuple : (x, y)
        Coordinates of the mapped point inside the quadrilateral
    """

    # Sample a point from the unit square
    u, v = np.random.random(2)
    
    # Extract corners for clarity
    p00 = corners[0]  # bottom-left
    p10 = corners[1]  # bottom-right
    p11 = corners[2]  # top-right
    p01 = corners[3]  # top-left
    
    # Bilinear interpolation
    point = (1-u)*(1-v)*p00 + u*(1-v)*p10 + u*v*p11 + (1-u)*v*p01
    
    # Return as a tuple (x, y)
    return (float(point[0]), float(point[1]))

def sample_distractor_positions(num_distractors, quad_corners, min_distance_main=2, min_distance_other=1, tube_radius=1.5):
    """Sample non-overlapping distractor ball positions avoiding a directional tube in front of the main ball."""
    positions = []
    max_tries = 50
    main_x, main_y = RANDOMIZATION_PARAMS["BALL_GROUND_COORDS"]
    force_angle_deg = RANDOMIZATION_PARAMS["INITIAL_FORCE_ANGLE"]

    # Compute force direction unit vector
    angle_rad = math.radians(force_angle_deg)
    force_dir = np.array([math.cos(angle_rad), math.sin(angle_rad)])

    for i in range(num_distractors):
        for _ in range(max_tries):
            x, y = sample_from_quadrilateral(quad_corners)

            dx = x - main_x
            dy = y - main_y
            if math.sqrt(dx * dx + dy * dy) < min_distance_main:
                continue

            vec_to_point = np.array([dx, dy])
            proj_length = np.dot(vec_to_point, force_dir)

            # Check direction
            if proj_length < 0:
                proj_vec = proj_length * force_dir
                perp_vec = vec_to_point - proj_vec
                perp_dist = np.linalg.norm(perp_vec)
                if perp_dist < tube_radius:
                    continue

            # Reject if too close to other distractors
            too_close = False
            for px, py, _ in positions:
                if math.sqrt((x - px)**2 + (y - py)**2) < min_distance_other:
                    too_close = True
                    break
            if too_close:
                continue

            positions.append((x, y, -0.282676))  # Fixed Z
            break
        else:
            print(f"Warning: Could not find valid position for distractor ball {i}")

    RANDOMIZATION_PARAMS["DISTRACTOR_BALL_POSITIONS"] = positions
    RANDOMIZATION_PARAMS["NUM_DISTRACTOR_BALLS"] = len(positions)

def randomly_place_ball():

    main_ball = bpy.data.objects.get(RANDOMIZATION_PARAMS["MOVING_BALL_NAME"])
    other_ball_name = "bowling_ball" if main_ball.name == "dirty_football_LOD0" else "dirty_football_LOD0"
    other_ball = bpy.data.objects.get(other_ball_name)
    if not main_ball or not other_ball:
        print("Warning: One of the ball objects not found in scene")
        return


    random_x, random_y = RANDOMIZATION_PARAMS["BALL_GROUND_COORDS"]

    # Use existing Z from main ball
    z_position = main_ball.location.z

    # Position the main ball and ensure it's visible
    main_ball.location = (random_x, random_y, z_position)
    main_ball.hide_viewport = False
    main_ball.hide_render = False
    print(f"Ball placed at original position: ({random_x:.2f}, {random_y:.2f}, {z_position:.2f})")

    for collection in bpy.data.collections:
        if main_ball.name in collection.objects:
            collection.hide_viewport = False
            collection.hide_render = False

    # Ensure correct rigid body settings for the main ball
    if hasattr(main_ball, 'rigid_body') and main_ball.rigid_body:
        main_ball.rigid_body.kinematic = True
        main_ball.rigid_body.type = 'ACTIVE'

        if hasattr(main_ball.rigid_body, 'linear_velocity'):
            main_ball.rigid_body.linear_velocity = (0, 0, 0)
        if hasattr(main_ball.rigid_body, 'angular_velocity'):
            main_ball.rigid_body.angular_velocity = (0, 0, 0)

        main_ball.keyframe_insert(data_path="location", frame=1)
        bpy.context.scene.frame_set(1)
        main_ball.rigid_body.keyframe_insert(data_path="kinematic", frame=1)

    # Move the other ball far away
    other_ball.location.x = -100  # Keep current y/z

def record_ball_pixel_coords_at_frame1(scene):
    """Record true ball pixel coordinates at frame 1 after all setup."""
    ball = bpy.data.objects.get(RANDOMIZATION_PARAMS["MOVING_BALL_NAME"])
    if not ball:
        print("Ball object not found")
        return

    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
    if not camera:
        print("Camera not found")
        return

    current_frame = scene.frame_current

    # Set frame to 1
    scene.frame_set(1)

    # Get ball location
    world_point = ball.matrix_world.translation
    pixel_coords = world_to_camera_view(scene, camera, world_point)

    # Update in RANDOMIZATION_PARAMS
    #Swap Y to be correct for Blender's coordinate system
    pixel_coords = (pixel_coords[0], scene.render.resolution_y - pixel_coords[1])
    RANDOMIZATION_PARAMS["BALL_PIXEL_COORDS"] = [pixel_coords[0], pixel_coords[1]]
    print(f"Corrected ball pixel coordinates recorded: {pixel_coords}")

    # Restore frame
    scene.frame_set(current_frame)




def clone_and_place_distractor_balls():
    dirty_football = bpy.data.objects.get("dirty_football_LOD0")
    bowling_ball = bpy.data.objects.get("bowling_ball")

    if not dirty_football or not bowling_ball:
        print("Error: Required base objects not found")
        return

    # Remove existing distractor balls
    for obj in list(bpy.data.objects):
        if obj.name.startswith("distractor_ball_"):
            bpy.data.objects.remove(obj, do_unlink=True)

    num_distractors = len(RANDOMIZATION_PARAMS["DISTRACTOR_BALL_POSITIONS"])

    # Decide whether to include a bowling ball distractor
    use_bowling_clone = random.random() < ClONE_BOWLING_BALL_PROBABILITY if num_distractors > 0 else False
    bowling_clone_index = random.randint(0, num_distractors - 1) if use_bowling_clone else -1

    for i, (x, y, z) in enumerate(RANDOMIZATION_PARAMS["DISTRACTOR_BALL_POSITIONS"]):
        base_ball = bowling_ball if i == bowling_clone_index else dirty_football

        new_ball = base_ball.copy()
        new_ball.data = base_ball.data.copy()
        new_ball.animation_data_clear()
        new_ball.name = f"distractor_{base_ball.name}_{i}"
        bpy.context.collection.objects.link(new_ball)

        apply_ball_textures(new_ball)

        new_ball.location = (x, y, z)
        print(f"Placed distractor ball {i} at ({x:.2f}, {y:.2f}, {z:.2f})")

        bpy.ops.object.select_all(action='DESELECT')
        new_ball.select_set(True)
        bpy.context.view_layer.objects.active = new_ball

        if not new_ball.rigid_body:
            bpy.ops.rigidbody.object_add()

        new_ball.rigid_body.type = 'ACTIVE'
        new_ball.rigid_body.kinematic = True
        new_ball.rigid_body.use_deactivation = False
        new_ball.rigid_body.use_start_deactivated = False
        new_ball.rigid_body.collision_shape = 'SPHERE'
        new_ball.rigid_body.use_margin = True
        new_ball.rigid_body.collision_margin = 0.02
        new_ball.rigid_body.mass = 1.0



def setup_random_scene():
    """Main function to set up all random parameters for the scene"""

    main_ball = bpy.data.objects.get(RANDOMIZATION_PARAMS["MOVING_BALL_NAME"])
    apply_ball_textures(main_ball)

    apply_ground_textures()

    add_sun_light()
    enhance_existing_lights()

    randomly_place_ball()

    clone_and_place_distractor_balls()

    randomize_force_field_strength(scene=bpy.context.scene)

    record_ball_pixel_coords_at_frame1(bpy.context.scene)

    update_output_path()

    print("Random scene setup complete with ball and distractors")


def render_pre_handler(scene):
    """Handler for command line rendering"""
    
    # Only set up the scene at the start frame
    if scene.frame_current == scene.frame_start:
        
        # Set up the scene
        #Why is there 2 scene setups?
        setup_random_scene()
        
        # Make sure we're on frame 1 to start the render sequence properly
        scene.frame_set(1)
        
        # Print confirmation of render path
        print(f"Render will output to: {bpy.context.scene.render.filepath}")

def render_complete_handler(scene):
    """Handler for when rendering is complete"""

    #Saves the correct angle
    compute_pixel_angle_from_video(scene)

    old_output_path = RANDOMIZATION_PARAMS["OUTPUT_PATH"]
    if not old_output_path or RANDOMIZATION_PARAMS["PIXEL_ANGLE"] is None:
        print("Missing output path or pixel angle")
        return

    new_dir_name = f"angle_{RANDOMIZATION_PARAMS['INITIAL_FORCE_ANGLE']:.1f}_force_{RANDOMIZATION_PARAMS['GENERATED_FORCE_STRENGTH']:.2f}_coordx_{RANDOMIZATION_PARAMS['BALL_PIXEL_COORDS'][0]}_coordy_{RANDOMIZATION_PARAMS['BALL_PIXEL_COORDS'][1]}_pixangle_{RANDOMIZATION_PARAMS['PIXEL_ANGLE']:.1f}"
    base_dir = os.path.dirname(old_output_path)
    new_output_path = os.path.join(base_dir, new_dir_name)

    try:
        os.rename(old_output_path, new_output_path)
        print(f"Renamed directory to: {new_output_path}")
        RANDOMIZATION_PARAMS["OUTPUT_PATH"] = new_output_path
        bpy.context.scene.render.filepath = os.path.join(new_output_path, "frame####")
    except Exception as e:
        print(f"Error renaming output directory: {e}")
        return

    # Update JSON with final path and pixel angle
    json_path = os.path.join(new_output_path, "params.json")
    with open(json_path, 'w') as f:
        json.dump(RANDOMIZATION_PARAMS, f, indent=4)

    
    if not RANDOMIZATION_PARAMS['OUTPUT_PATH'] or RANDOMIZATION_PARAMS['INITIAL_FORCE_ANGLE'] is None or RANDOMIZATION_PARAMS['GENERATED_FORCE_STRENGTH'] is None:
        print("Missing parameters for post-render processing")
        return
    
    print(f"Rendering complete! Files saved to: {RANDOMIZATION_PARAMS['OUTPUT_PATH']}")
    print(f"Parameters used: Force Angle = {RANDOMIZATION_PARAMS['INITIAL_FORCE_ANGLE']}, Force = {RANDOMIZATION_PARAMS['GENERATED_FORCE_STRENGTH']}")
    print(f"Ball ground coordinates: {RANDOMIZATION_PARAMS['BALL_GROUND_COORDS']}")
    print(f"Ball pixel coordinates: {RANDOMIZATION_PARAMS['BALL_PIXEL_COORDS']}")
    
    print(f"For MP4 creation, use: ffmpeg -framerate 24 -i {RANDOMIZATION_PARAMS['OUTPUT_PATH']}/frame%04d.png -c:v libx264 -pix_fmt yuv420p {RANDOMIZATION_PARAMS['OUTPUT_PATH']}/output.mp4")

def register():
    bpy.context.scene.frame_set(1)

    # 0. Choose which ball rolls
    use_bowling = random.random() < BOWLING_BALL_PROBABILITY
    main_ball = 'bowling_ball' if use_bowling else 'dirty_football_LOD0'
    other_ball = 'dirty_football_LOD0' if use_bowling else 'bowling_ball'
    RANDOMIZATION_PARAMS["MOVING_BALL_NAME"] = main_ball

    # 1. Randomize force angle and strength
    RANDOMIZATION_PARAMS["INITIAL_FORCE_ANGLE"] = random.uniform(0.0, 360.0)
    RANDOMIZATION_PARAMS["GENERATED_FORCE_STRENGTH"] = random.uniform(8.0, 64.0)

    randomize_camera_zoom()

    # 2. Randomize main ball position inside quad
    quad_corners = np.array([
        [-10.0, -10.0],
        [10.0, -10.0],
        [5.0, 4.3],
        [-5.0, 4.3]
    ]) * (1-RANDOMIZATION_PARAMS["CAMERA_ZOOM_FACTOR"]) * 0.75 # 0.9 to be tuned...
    RANDOMIZATION_PARAMS["BALL_GROUND_COORDS"] = sample_from_quadrilateral(quad_corners)

    ## 3. Sample distractor positions using same logic
    num_distractors = random.randint(1, 3)
    sample_distractor_positions(num_distractors, quad_corners)

    # 4. Compute force field position near ball
    angle_rad = math.radians(RANDOMIZATION_PARAMS["INITIAL_FORCE_ANGLE"])
    distance = 0.5
    x_pos = RANDOMIZATION_PARAMS["BALL_GROUND_COORDS"][0] + distance * math.cos(angle_rad)
    y_pos = RANDOMIZATION_PARAMS["BALL_GROUND_COORDS"][1] + distance * math.sin(angle_rad)

    force = bpy.data.objects.get('Force')
    if force:
        force.location.x = x_pos
        force.location.y = y_pos



    # 5. Remove existing handlers
    for handler in bpy.app.handlers.render_pre:
        if handler.__name__ == 'render_pre_handler':
            bpy.app.handlers.render_pre.remove(handler)
    for handler in bpy.app.handlers.render_complete:
        if handler.__name__ == 'render_complete_handler':
            bpy.app.handlers.render_complete.remove(handler)
    for handler in bpy.app.handlers.frame_change_pre:
        if handler.__name__ == 'randomize_force_field_strength':
            bpy.app.handlers.frame_change_pre.remove(handler)



    # 6. Register handlers
    bpy.app.handlers.render_pre.append(render_pre_handler)
    bpy.app.handlers.frame_change_pre.append(randomize_force_field_strength)
    bpy.app.handlers.render_complete.append(render_complete_handler)

    # 7. Prepare main ball physics state
    ball = bpy.data.objects.get(RANDOMIZATION_PARAMS["MOVING_BALL_NAME"])
    if ball:
        ball.hide_viewport = False
        ball.hide_render = False
        if hasattr(ball, 'rigid_body') and ball.rigid_body:
            ball.rigid_body.kinematic = True
            if hasattr(ball.rigid_body, 'linear_velocity'):
                ball.rigid_body.linear_velocity = (0, 0, 0)
            if hasattr(ball.rigid_body, 'angular_velocity'):
                ball.rigid_body.angular_velocity = (0, 0, 0)

    # 8. Set up the randomized scene and bake physics
    setup_random_scene()
    bake_physics()

    print("Random scene generation handlers registered successfully!")



def unregister():
    for handler in bpy.app.handlers.render_pre:
        if handler.__name__ == 'render_pre_handler':
            bpy.app.handlers.render_pre.remove(handler)

    for handler in bpy.app.handlers.render_complete:
        if handler.__name__ == 'render_complete_handler':
            bpy.app.handlers.render_complete.remove(handler)

    for handler in bpy.app.handlers.frame_change_pre:
        if handler.__name__ == 'randomize_force_field_strength':
            bpy.app.handlers.frame_change_pre.remove(handler)


if __name__ == "__main__":
    register()
    print("Random ball scene generator script loaded!")
    print("- Scene will be randomized automatically when rendering starts")