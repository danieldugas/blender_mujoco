bl_info = {
    "name": "Bone Rotation Exporter",
    "blender": (3, 5, 0),  # Update to the version you are using
    "category": "Animation",
}

import bpy
import mathutils


class ExportBoneRotationsOperator(bpy.types.Operator):
    bl_idname = "export.bone_rotations"
    bl_label = "Export Bone Rotations"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return self.export_bone_rotations(context)

    def export_bone_rotations(self, context):
        obj = bpy.context.object
        if obj and obj.type == 'ARMATURE':
            armature = obj.data
            anim_data = obj.animation_data
            if not anim_data or not anim_data.action:
                self.report({'WARNING'}, "No animation found!")
                return {'CANCELLED'}

            action = anim_data.action
            frame_start = int(action.frame_range[0])
            frame_end = int(action.frame_range[1])
            
            # List to store all frames' bone rotations
            all_frame_data = []

            for frame in range(frame_start, frame_end + 1):
                bpy.context.scene.frame_set(frame)
                frame_data = {}
                for bone_name, pose_bone in obj.pose.bones.items():
                    # Directly access the Euler rotation from the pose bone
                    if pose_bone.rotation_mode == 'XYZ':
                        rotation_euler = pose_bone.rotation_euler
                        frame_data[bone_name] = rotation_euler.y
                        self.report({'INFO'}, f"Bone {bone_name} rotation: {rotation_euler.y}")

                all_frame_data.append((frame, frame_data))

            # Example of saving to a text file
            with open("/tmp/exported_data.txt", "w") as file:
                for frame, data in all_frame_data:
                    file.write(f"Frame {frame}:\n")
                    for bone, rotation in data.items():
                        file.write(f"  Bone {bone}: {rotation}\n")

            self.report({'INFO'}, "Bone rotations exported successfully!")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Selected object is not an armature!")
            return {'CANCELLED'}

class BoneRotationExporterPanel(bpy.types.Panel):
    bl_label = "Bone Rotation Exporter"
    bl_idname = "PANEL_PT_bone_rotation_exporter"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        layout.operator("export.bone_rotations")

def register():
    bpy.utils.register_class(ExportBoneRotationsOperator)
    bpy.utils.register_class(BoneRotationExporterPanel)

def unregister():
    bpy.utils.unregister_class(ExportBoneRotationsOperator)
    bpy.utils.unregister_class(BoneRotationExporterPanel)

if __name__ == "__main__":
    register()