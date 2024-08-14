import bpy

names = [
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_roll_link",
    "logo_link",
    "left_palm_link",
    "left_zero_link",
    "left_one_link",
    "left_two_link",
    "left_three_link",
    "left_four_link",
    "left_five_link",
    "left_six_link",
    "right_palm_link",
    "right_zero_link",
    "right_one_link",
    "right_two_link",
    "right_three_link",
    "right_four_link",
    "right_five_link",
    "right_six_link",
]

# Filepath to your STL file
for name in names:
    # log name
    print(name)

    stl_file_path = "/home/daniel/Code/DeepMimic_mujoco/src/mujoco/humanoid_deepmimic/envs/asset/assets/"+name+".STL"

    # Import the STL file
    #bpy.ops.import_mesh.stl(filepath=stl_file_path)
    bpy.ops.wm.stl_import(filepath=stl_file_path)

    # Set the export path for the OBJ file
    obj_file_path =  "/home/daniel/Code/mujoco_wasm/examples/scenes/assets/"+name+".obj"

    # Select the imported object (ensure that it is the active object)
    imported_object = bpy.context.object
    bpy.ops.object.select_all(action='DESELECT')
    imported_object.select_set(True)
    bpy.context.view_layer.objects.active = imported_object

    # Export the selected object as an OBJ file
    bpy.ops.wm.obj_export(filepath=obj_file_path, up_axis="Z", forward_axis="Y")

    # delete the selected object
    bpy.ops.object.delete()