import bpy
import json
from mathutils import Quaternion, Euler, Vector

bl_info = {
    "name": "Import JSON Animation",
    "blender": (3, 5, 0),  # Adjust to your Blender version
    "category": "Animation",
}

class ImportJSONAnimationOperator(bpy.types.Operator):
    bl_idname = "import_animation.json"
    bl_label = "Import Animation from JSON"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        return self.import_animation(context, self.filepath)

    def import_animation(self, context, filepath):
        # Load the JSON file
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        frames = data.get("Frames", [])
        labels = data.get("Labels", [])
        
        obj = bpy.context.object
        if obj and obj.type == 'ARMATURE':
            # Prepare the armature for animation
            armature = obj
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            
            for frame_index, frame_data in enumerate(frames):
                bpy.context.scene.frame_set(frame_index + 1)  # Frames are 1-based in Blender
                
                dt = frame_data[0]
                frame_qpos = frame_data[1:]
                # Extract armature position and rotation (first 7 values)
                pos = Vector(frame_qpos[:3])
                quat = Quaternion(frame_qpos[3:7])
                
                armature.location = pos
                armature.rotation_mode = 'QUATERNION'
                armature.rotation_quaternion = quat
                
                armature.keyframe_insert(data_path="location", frame=frame_index + 1)
                armature.keyframe_insert(data_path="rotation_quaternion", frame=frame_index + 1)
                
                # Set each bone's Y rotation based on the labels
                for label_index, label in enumerate(labels):
                    if "joint" not in label:
                        continue
                    bone_name = label
                    if bone_name in armature.pose.bones:
                        pose_bone = armature.pose.bones[bone_name]
                        y_rotation = frame_data[label_index]
                        
                        pose_bone.rotation_mode = 'XYZ'
                        pose_bone.rotation_euler = Euler((0, y_rotation, 0), 'XYZ')
                        
                        pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame_index + 1)
                        
        else:
            self.report({'ERROR'}, "No armature selected!")
            return {'CANCELLED'}

        self.report({'INFO'}, "Animation imported successfully!")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class ImportJSONAnimationPanel(bpy.types.Panel):
    bl_label = "Import JSON Animation"
    bl_idname = "PANEL_PT_import_json_animation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        layout.operator("import_animation.json")

def register():
    bpy.utils.register_class(ImportJSONAnimationOperator)
    bpy.utils.register_class(ImportJSONAnimationPanel)

def unregister():
    bpy.utils.unregister_class(ImportJSONAnimationOperator)
    bpy.utils.unregister_class(ImportJSONAnimationPanel)

if __name__ == "__main__":
    register()