import bpy
import json
from mathutils import Quaternion, Euler, Vector

bl_info = {
    "name": "Export JSON Animation",
    "blender": (3, 5, 0),  # Adjust to your Blender version
    "category": "Animation",
}

class ExportJSONAnimationOperator(bpy.types.Operator):
    bl_idname = "export_animation.json"
    bl_label = "Export Animation to JSON"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        return self.export_animation(context, self.filepath)

    def export_animation(self, context, filepath):
        obj = bpy.context.object
        if obj and obj.type == 'ARMATURE':
            armature = obj
            anim_data = obj.animation_data
            if not anim_data or not anim_data.action:
                self.report({'WARNING'}, "No animation found!")
                return {'CANCELLED'}

            action = anim_data.action
            frame_start = int(action.frame_range[0])
            frame_end = int(action.frame_range[1])

            labels = ["armature_pos_x", "armature_pos_y", "armature_pos_z", "armature_quat_w", "armature_quat_x", "armature_quat_y", "armature_quat_z"]

            for bone_name in armature.pose.bones.keys():
                if "joint" in bone_name:
                    labels.append(bone_name)

            frames = []

            for frame in range(frame_start, frame_end + 1):
                bpy.context.scene.frame_set(frame)
                
                frame_data = [1.0 / bpy.context.scene.render.fps]  # dt = 1/fps
                
                # Armature position and rotation
                pos = armature.location
                quat = armature.rotation_quaternion

                frame_data.extend([pos.x, pos.y, pos.z, quat.w, quat.x, quat.y, quat.z])

                # Bone Y Euler rotation for each joint
                for bone_name in armature.pose.bones.keys():
                    if "joint" in bone_name:
                        pose_bone = armature.pose.bones[bone_name]
                        y_rotation = pose_bone.rotation_euler.y
                        frame_data.append(y_rotation)

                frames.append(frame_data)

            # Write the JSON data
            animation_data = {
                "Labels": labels,
                "Frames": frames
            }

            with open(filepath, 'w') as file:
                json.dump(animation_data, file, indent=4)

            self.report({'INFO'}, f"Animation exported to {filepath}")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No armature selected!")
            return {'CANCELLED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class ExportJSONAnimationPanel(bpy.types.Panel):
    bl_label = "Export JSON Animation"
    bl_idname = "PANEL_PT_export_json_animation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        layout.operator("export_animation.json")

def register():
    bpy.utils.register_class(ExportJSONAnimationOperator)
    bpy.utils.register_class(ExportJSONAnimationPanel)

def unregister():
    bpy.utils.unregister_class(ExportJSONAnimationOperator)
    bpy.utils.unregister_class(ExportJSONAnimationPanel)

if __name__ == "__main__":
    register()