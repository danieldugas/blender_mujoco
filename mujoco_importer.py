bl_info = {
    "name": "MuJoCo XML Importer",
    "blender": (3, 0, 0),
    "category": "Import-Export",
}

BLENDER = True

try:
    import bpy
    from bpy_extras.io_utils import ImportHelper
except:
    BLENDER = False

import xml.etree.ElementTree as ET
import numpy as np
import os

if True: # transforms.py
    def normalized(vec):
        norm = np.linalg.norm(vec)
        if norm == 0:
            print("Warning: attempting to normalize zero vector")
            return vec
        return vec / norm

    class Quaternion(object):
        def __init__(self, x, y, z, w):
            self._x = x
            self._y = y
            self._z = z
            self._w = w

        def __repr__(self):
            return "Quaternion(x={:.9f}, y={:.9f}, z={:.9f}, w={:.9f})".format(*self.xyzw())

        def as_quaternion(something):
            if isinstance(something, Quaternion):
                return something
            elif isinstance(something, tuple) and len(something) == 4:
                return Quaternion(*something)
            elif isinstance(something, list) and len(something) == 4:
                return Quaternion(*something)
            else:
                raise ValueError("Cannot convert {} to Quaternion".format(something))
        
        def xyzw(self):
            return [self._x, self._y, self._z, self._w]

        def from_transform_matrix(self, matrix4x4):
            x, y, z, w = quaternion_from_transform_matrix(matrix4x4)
            return Quaternion(x, y, z, w)

        def to_transform_matrix(self):
            return transform_matrix_from_quaternion(*self.xyzw())

        def to_rpy(self):
            """ returns roll, pitch, yaw in radians """
            import math
            q0 = self._w
            q1 = self._x
            q2 = self._y
            q3 = self._z
            roll = math.atan2(
                2 * ((q2 * q3) + (q0 * q1)),
                q0**2 - q1**2 - q2**2 + q3**2
            )  # radians
            pitch = math.asin(2 * ((q1 * q3) - (q0 * q2)))
            yaw = math.atan2(
                2 * ((q1 * q2) + (q0 * q3)),
                q0**2 + q1**2 - q2**2 - q3**2
            )
            return (roll, pitch, yaw)

    class Transform(object):
        def __init__(self, origin=None, x_axis=None, y_axis=None, quaternion=None):
            self._matrix = None
            self._quaternion = None
            if x_axis is not None or y_axis is not None:
                if x_axis is None or y_axis is None:
                    raise ValueError("Underspecified transform: Must specify neither or both axes.")
            if origin is not None or x_axis is not None or y_axis is not None:
                self._matrix = np.eye(4)
                if x_axis is not None and y_axis is not None:
                    if quaternion is not None:
                        raise ValueError("Overspecified transform: Cannot specify both quaternion and axes")
                    self._matrix[:3, 0] = x_axis
                    self._matrix[:3, 1] = y_axis
                    self._matrix[:3, 2] = np.cross(x_axis, y_axis)
                if origin is not None:
                    self._matrix[:3, 3] = origin
            if quaternion is not None:
                self._quaternion = Quaternion.as_quaternion(quaternion)

        def to_json_dict(self):
            json_dict = {
                "type": "Transform",
                "_matrix": self._matrix.tolist() if self._matrix is not None else None,
                "_quaternion": self._quaternion.xyzw() if self._quaternion is not None else None,
            }
            return json_dict
        
        def from_json_dict(json_dict):
            new = Transform()
            new._matrix = np.array(json_dict["_matrix"]) if json_dict["_matrix"] is not None else None
            new._quaternion = Quaternion(*json_dict["_quaternion"]) if json_dict["_quaternion"] is not None else None
            return new

        def __repr__(self):
            return "Transform(origin={}, x_axis={}, y_axis={})".format(self.origin(), self.x_axis(), self.y_axis())

        def __mul__(self, other):
            """ T_B_in_C * T_A_in_B = T_A_in_C """
            if isinstance(other, Transform):
                return Transform.from_matrix(np.dot(self.matrix(), other.matrix()))
            else:
                raise NotImplementedError
            
        def is_right_handed(self):
            return np.linalg.det(self.matrix()[:3, :3]) > 0
            
        def from_matrix(matrix4x4):
            new = Transform()
            new._matrix = matrix4x4
            return new

        def from_quaternion(quaternion, origin=None):
            return Transform(origin=origin, quaternion=quaternion)
        
        def from_axis_angle(axis, angle_rad, translation=None):
            """ translation is not the axis origin! To implement rotation around an origin, use from_rotation_around_point """
            new = Transform()
            new._matrix = transform_matrix_from_axis_angle(axis, angle_rad, translation)
            return new

        def from_rotation_around_point(axis, angle_rad, point):
            new = Transform()
            new._matrix = transform_matrix_from_axis_angle(axis, angle_rad)
            # to get translation, rotate point around axis at origin, compare to previous
            rotated_point = new.transform_points([point])[0]
            new._matrix[:3, 3] = point - rotated_point
            return new

        def inverse(self):
            return Transform.from_matrix(inverse(self.matrix()))
        
        def matrix(self):
            if self._quaternion is not None:
                rot_mat4x4 = self.quaternion().to_transform_matrix()
                if self._matrix is not None:
                    if not np.allclose(self._matrix[:3, :3], np.eye(3)):
                        raise ValueError("Overdefined transform: transform has a non-zero quaternion and non-zero rotation matrix.")
                    rot_mat4x4[:3, 3] = self._matrix[:3, 3]
                return rot_mat4x4
            elif self._matrix is not None:
                return self._matrix
            else:
                return np.eye(4)

        def quaternion(self):
            if self._quaternion is not None:
                return self._quaternion
            elif self._matrix is not None:
                return Quaternion(*quaternion_from_transform_matrix(self.matrix()))
            else:
                return Quaternion(0, 0, 0, 1)

        def origin(self):
            return self.matrix()[:3, 3]

        def translation(self):
            return self.origin()

        def x_axis(self):
            return self.matrix()[:3, 0]

        def y_axis(self):
            return self.matrix()[:3, 1]

        def z_axis(self):
            return self.matrix()[:3, 2]

        def to_axis_angle(self):
            return axis_angle_from_transform_matrix(self.matrix())

        def to_compas_frame(self):
            from compas.geometry import Frame
            return Frame(self.origin(), self.x_axis(), self.y_axis())

        def from_compas_frame(frame):
            return Transform([frame.point.x, frame.point.y, frame.point.z], frame.xaxis, frame.yaxis)

        def to_pose_msg(self):
            from geometry_msgs.msg import Pose
            pose = Pose()
            x, y, z = self.origin()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            qx, qy, qz, qw = self.quaternion().xyzw()
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            return pose
        
        def from_pose_msg(pose_msg):
            x = pose_msg.position.x
            y = pose_msg.position.y
            z = pose_msg.position.z
            qx = pose_msg.orientation.x
            qy = pose_msg.orientation.y
            qz = pose_msg.orientation.z
            qw = pose_msg.orientation.w
            return Transform.from_quaternion(Quaternion(qx, qy, qz, qw), [x, y, z])
        
        def transform_vector(self, vector_in_A_frame):
            """ If this transform is the transform of A in B, then this returns the vector in B frame """
            return transform_vector(vector_in_A_frame, self.matrix())
        
        def transform_point(self, point):
            return transform_point(point, self.matrix())

        def transform_points(self, points):
            return transform_points(points, self.matrix())
        
        def plot_polyscope(self, name="A_in_B", axis_length=0.1):
            show_frame_in_polyscope(self.matrix(), name=name, axis_length=axis_length)

        def print_matrix(self):
            # 1 decimal
            for row in self.matrix():
                for val in row:
                    print("{:.1f}".format(val), end=" ")
                print()


    def inverse(transform_matrix_A_in_B):
        transform_matrix_B_in_A = np.linalg.inv(transform_matrix_A_in_B)
        return transform_matrix_B_in_A


    def show_frame_in_polyscope(frame_in_world_matrix, name="frame", axis_length=1.0):
        import polyscope as ps
        origin = frame_in_world_matrix[:3, 3]
        x_axis = frame_in_world_matrix[:3, 0]
        y_axis = frame_in_world_matrix[:3, 1]
        z_axis = frame_in_world_matrix[:3, 2]
        ps.register_curve_network(
            "{}_x_axis".format(name),
            np.array([origin, origin + x_axis * axis_length]),
            np.array([[0, 1]]),
            color=(1.0, 0.0, 0.0),
        )
        ps.register_curve_network(
            "{}_y_axis".format(name),
            np.array([origin, origin + y_axis * axis_length]),
            np.array([[0, 1]]),
            color=(0.0, 1.0, 0.0),
        )
        ps.register_curve_network(
            "{}_z_axis".format(name),
            np.array([origin, origin + z_axis * axis_length]),
            np.array([[0, 1]]),
            color=(0.0, 0.0, 1.0),
        )


    def transform_matrix_from_origin_and_xy_axes(origin, x_axis, y_axis):
        """
        returns the matrix for the transform A in B, where origin is the origin of A in B, and x_axis and y_axis are the x and y axes of A in B
        """
        z_axis = np.cross(x_axis, y_axis)
        xx, xy, xz = x_axis
        yx, yy, yz = y_axis
        zx, zy, zz = z_axis
        ox, oy, oz = origin
        transform_matrix = np.array(
            [
                [xx, yx, zx, ox],
                [xy, yy, zy, oy],
                [xz, yz, zz, oz],
                [0, 0, 0, 1],
            ]
        )
        return transform_matrix

    def transform_matrix_from_translation(translation):
        transform_matrix = transform_matrix_from_origin_and_xy_axes(translation, [1, 0, 0], [0, 1, 0])
        return transform_matrix

    def transform_matrix_from_axis_angle(axis, angle, translation=None):
        """ angle in radians """
        ux, uy, uz = axis
        tx, ty, tz = translation if translation is not None else [0, 0, 0]
        transform_matrix = np.array(
            [
                [ux * ux * (1 - np.cos(angle)) + np.cos(angle), ux * uy * (1 - np.cos(angle)) - uz * np.sin(angle), ux * uz * (1 - np.cos(angle)) + uy * np.sin(angle), tx],
                [uy * ux * (1 - np.cos(angle)) + uz * np.sin(angle), uy * uy * (1 - np.cos(angle)) + np.cos(angle), uy * uz * (1 - np.cos(angle)) - ux * np.sin(angle), ty],
                [uz * ux * (1 - np.cos(angle)) - uy * np.sin(angle), uz * uy * (1 - np.cos(angle)) + ux * np.sin(angle), uz * uz * (1 - np.cos(angle)) + np.cos(angle), tz],
                [0, 0, 0, 1],
            ]
        )
        return transform_matrix

    def axis_angle_from_transform_matrix(transform_matrix):
        """ returns axis, angle """
        R = transform_matrix[:3, :3]
        angle = np.arccos((np.trace(R) - 1) / 2)
        if angle == 0:
            return None, 0
        if np.allclose(angle, np.pi):
            pass
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
        norm = np.linalg.norm(axis)
        if norm != 0:
            axis = axis / norm
        return axis, angle

    def axis_angle_from_transform_matrix(transform_matrix):
        """ returns axis, angle 
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/
        """
        R = transform_matrix[:3, :3]
        epsilon = 0.0001 # margin to allow for rounding errors
        epsilon2 = 0.1 # margin to distinguish between 0 and 180 degrees
        if ((np.abs(R[0, 1]-R[1, 0])< epsilon)      and (np.abs(R[0, 2]-R[2, 0])< epsilon)      and (np.abs(R[1, 2]-R[2, 1])< epsilon)) :
            # singularity found
            # first check for identity matrix which must have +1 for all terms
            #  in leading diagonaland zero in other terms
            if ((np.abs(R[0, 1]+R[1, 0]) < epsilon2) and (np.abs(R[0, 2]+R[2, 0]) < epsilon2) and (np.abs(R[1, 2]+R[2, 1]) < epsilon2) and (np.abs(R[0, 0]+R[1, 1]+R[2, 2]-3) < epsilon2)) :
                # this singularity is identity matrix so angle = 0
                return None, 0 # zero angle, arbitrary axis
            
            # otherwise this singularity is angle = 180
            angle = np.pi
            xx = (R[0, 0]+1)/2
            yy = (R[1, 1]+1)/2
            zz = (R[2, 2]+1)/2
            xy = (R[0, 1]+R[1, 0])/4
            xz = (R[0, 2]+R[2, 0])/4
            yz = (R[1, 2]+R[2, 1])/4
            if ((xx > yy) and (xx > zz)) : # R[0, 0] is the largest diagonal term
                if (xx< epsilon) :
                    x = 0
                    y = 0.7071
                    z = 0.7071
                else :
                    x = np.sqrt(xx)
                    y = xy/x
                    z = xz/x
                
            elif (yy > zz) : # R[1, 1] is the largest diagonal term
                if (yy< epsilon) :
                    x = 0.7071
                    y = 0
                    z = 0.7071
                else :
                    y = np.sqrt(yy)
                    x = xy/y
                    z = yz/y
                    
            else : # R[2, 2] is the largest diagonal term so base result on this
                if (zz< epsilon) :
                    x = 0.7071
                    y = 0.7071
                    z = 0
                else :
                    z = np.sqrt(zz)
                    x = xz/z
                    y = yz/z
                
            return np.array([x, y, z]), angle # return 180 deg rotation
        
        # as we have reached here there are no singularities so we can handle normally
        s = np.sqrt((R[2, 1] - R[1, 2])*(R[2, 1] - R[1, 2])
            +(R[0, 2] - R[2, 0])*(R[0, 2] - R[2, 0])
            +(R[1, 0] - R[0, 1])*(R[1, 0] - R[0, 1])) # used to normalise
        if (np.abs(s) < 0.001):
            s=1 
            # prevent divide by zero, should not happen if matrix is orthogonal and should be
            # caught by singularity test above, but I've left it in just in case
        angle = np.arccos(( R[0, 0] + R[1, 1] + R[2, 2] - 1)/2)
        x = (R[2, 1] - R[1, 2])/s
        y = (R[0, 2] - R[2, 0])/s
        z = (R[1, 0] - R[0, 1])/s
        return np.array([x, y, z]), angle

    def transform_matrix_from_quaternion(x, y, z, w):
        transform_matrix = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, 0],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w, 0],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y, 0],
            [0, 0, 0, 1],
        ])
        return transform_matrix

    def quaternion_from_transform_matrix(transform_matrix):
        """ Returns x, y, z, w components of the quaternion defined by the upper left part of the 4x4 transform_matrix """
        q = np.empty((4, ), dtype=np.float64)
        M = np.array(transform_matrix, dtype=np.float64, copy=False)[:4, :4]
        t = np.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / np.sqrt(t * M[3, 3])
        x, y, z, w = q
        return x, y, z, w

    def transform_points(points_in_A_frame, transform_matrix_A_in_B):
        points_in_A_frame = np.asanyarray(points_in_A_frame).reshape((-1, 3))
        transform_matrix_A_in_B = np.asanyarray(transform_matrix_A_in_B)
        _N, _3 = points_in_A_frame.shape
        _4, _4 = transform_matrix_A_in_B.shape
        # add a dimension to points_in_A_frame
        points_in_A_frame = np.hstack([points_in_A_frame, np.ones((_N, 1))])
        # transform_matrix_A_in_B
        points_in_B_frame = np.dot(points_in_A_frame, transform_matrix_A_in_B.T)
        # remove the dimension
        points_in_B_frame = points_in_B_frame[:, :-1]
        return points_in_B_frame

    def transform_point(point_in_A_frame, transform_matrix_A_in_B):
        return transform_points(np.asanyarray(point_in_A_frame).reshape((1, 3)), transform_matrix_A_in_B).reshape((3,))

    def transform_vectors(vectors_in_A_frame, transform_matrix_A_in_B):
        rotation_matrix_A_in_B = np.zeros_like(transform_matrix_A_in_B)
        rotation_matrix_A_in_B[:3, :3] = transform_matrix_A_in_B[:3, :3]
        return transform_points(vectors_in_A_frame, rotation_matrix_A_in_B)

    def transform_vector(vector_in_A_frame, transform_matrix_A_in_B):
        return transform_vectors(vector_in_A_frame.reshape((1, 3)), transform_matrix_A_in_B).reshape((3,))

    def transform_matrix_from_frame(frame):
        print("Warning: transform_matrix_from_frame is deprecated. Use transform_matrix_from_compas_frame instead.")
        return transform_matrix_from_compas_frame(frame)

    def transform_matrix_from_compas_frame(frame):
        from compas.geometry.transformations import Transformation
        return np.array(Transformation.from_frame(frame).matrix)

    def rotate_points_around_axis(points, axis_origin, axis, angle):
        translation_matrix = transform_matrix_from_translation(-np.array(axis_origin))
        rotation_matrix = transform_matrix_from_axis_angle(axis, angle)
        reverse_translation_matrix = transform_matrix_from_translation(np.array(axis_origin))
        return transform_points(points, reverse_translation_matrix @ rotation_matrix @ translation_matrix)

    def point_2d_to_3d(point_2d, z=0):
        return np.array([point_2d[0], point_2d[1], z])

    def points_2d_to_3d(points_2d, z=0):
        _N, _2 = points_2d.shape
        if _2 != 2:
            raise ValueError("points_2d must be Nx2")
        return np.hstack([np.asanyarray(points_2d), np.ones((len(points_2d), 1)) * z])

class BoneInfo:
    def __init__(self, name, parent_bone_name, head, tail, is_joint):
        self._name = name
        self._parent_bone_name = parent_bone_name
        self._head_xyz = head
        self._tail_xyz = tail
        if np.allclose(head, tail):
            raise ValueError("Bone {} has zero length. Blender will refuse to create it.".format(name))
        self._is_joint = is_joint

    def name(self):
        return self._name

    def parent_bone_name(self):
        return self._parent_bone_name

    def head(self):
        return self._head_xyz

    def tail(self):
        return self._tail_xyz

    def is_joint(self):
        return self._is_joint

class JointInfo:
    JBL = 0.05 # joint bone length
    JBP = "POST" # joint bone is either PRE (before the body pose) or POST (after the body pose)
    ADD_FREEJOINT_BONE = False
    def __init__(self, name, body, body_parent, axis, pos):
        self._name = name
        self._body = body
        self._body_parent = body_parent
        self._axis = axis
        self._pos = pos

    def __repr__(self):
        return "JointInfo(name={})".format(self._name)

    def get_name(self):
        return self._name
    
    def get_initial_pos(self, in_world=True):
        if in_world:
            return self._body.get_initial_tf_in_world().transform_point(self._pos)
        else:
            return self._body.get_initial_tf_in_armature().transform_point(self._pos)

    def get_initial_axis(self, in_world=True):
        if in_world:
            return self._body.get_initial_tf_in_world().transform_vector(self._axis)
        else:
            return self._body.get_initial_tf_in_armature().transform_vector(self._axis)

    def get_initial_bonehead(self):
        in_world = JointInfo.ADD_FREEJOINT_BONE
        o = self.get_initial_pos(in_world)
        v = self.get_initial_axis(in_world)
        if JointInfo.JBP == "PRE":
            return o - JointInfo.JBL * v
        else:
            return o

    def get_initial_bonetail(self):
        in_world = JointInfo.ADD_FREEJOINT_BONE
        o = self.get_initial_pos(in_world)
        v = self.get_initial_axis(in_world)
        if JointInfo.JBP == "PRE":
            return o
        else:
            return o + JointInfo.JBL * v

class MeshGeomInfo:
    def __init__(self, stl_path):
        self._stl_path = stl_path
    
    def get_path(self):
        return self._stl_path

class BodyInfo:
    def __init__(self, name, parent, body_in_parent_tf):
        self._name = name
        self._parent = parent
        self._body_in_parent_tf = body_in_parent_tf
        self._geoms = []
        # infer initial world pose
        tf_parent_in_world = Transform()
        if self._parent is not None:
            tf_parent_in_world = self._parent.get_initial_tf_in_world()
        body_in_world = tf_parent_in_world * self._body_in_parent_tf
        self._initial_tf_in_world = body_in_world
        # infer initial pose in armature (root body)
        tf_parent_in_armature = Transform()
        if self._parent is not None:
            tf_parent_in_armature = self._parent.get_initial_tf_in_armature()
        body_in_armature = tf_parent_in_armature * self._body_in_parent_tf
        if self._parent is not None and self._parent.get_parent() is None:
            body_in_armature = Transform() # root bodies at armature origin
        self._initial_tf_in_armature = body_in_armature
        # infer depth
        self._depth = 0
        if self._parent is not None:
            self._depth = self._parent.get_depth() + 1
        # joints
        self._joints = []
        # end bone
        self._end_bone_name = None # not initialized
        self._bones_generated = False

    def add_joint(self, joint_info):
        self._joints.append(joint_info)

    def add_mesh_geom(self, mesh_geom_info):
        self._geoms.append(mesh_geom_info)

    def __repr__(self):
        return "BodyInfo(name={}, parent_name={})".format(self._name, self._parent.get_name() if self._parent is not None else "None")

    def get_name(self):
        return self._name
    
    def get_parent(self):
        return self._parent

    def get_initial_tf_in_world(self):
        return self._initial_tf_in_world

    def get_initial_tf_in_armature(self):
        return self._initial_tf_in_armature

    def get_depth(self):
        return self._depth
    
    def get_mesh_geoms(self):
        return self._geoms

    def create_bones(self):
        """
        each body creates one or more bones:
        - a bone for each joint, at the joint pos, with bone y == joint axis
        - a bone near it's origin to represent the body

        no attempt is made to have the bones visually connect to one another
        """
        self._bones_generated = True

        own_origin = self.get_initial_tf_in_world().origin() if JointInfo.ADD_FREEJOINT_BONE else self.get_initial_tf_in_armature().origin()
        if self.get_parent() is None: # no bones for worldbody
            self._end_bone_name = None
            return []
        if not JointInfo.ADD_FREEJOINT_BONE: # parent is worldbody, create a small bone
            if self.get_parent().get_parent() is None: 
                self._end_bone_name = "to_" + self.get_name()
                if JointInfo.JBP == "PRE":
                    bone_tail = own_origin
                    bone_head = own_origin - np.array([JointInfo.JBL, 0, 0])
                else:
                    bone_tail = own_origin + np.array([JointInfo.JBL, 0, 0])
                    bone_head = own_origin
                parent_to_body_bone = BoneInfo(self._end_bone_name, None, bone_head, bone_tail, False)
                return [parent_to_body_bone]

        # check init order
        if not self.get_parent()._bones_generated:
            raise ValueError("Child body's bones should be created after parent body's bones")

        # Fetch parent bone name for use when creating Armature heiarchy later
        parent_bone_name = self.get_parent()._end_bone_name

        # Create a bone for each joint aligned with that joint's axis.
        joint_bones = []
        for joint_info in self._joints:
            bone1_name = "to_" + self.get_name()
            jbone_name = joint_info.get_name()
            jbone_head = joint_info.get_initial_bonehead()
            jbone_tail = joint_info.get_initial_bonetail()
            joint_bone = BoneInfo(jbone_name, parent_bone_name, jbone_head, jbone_tail, True)
            joint_bones.append(joint_bone)
            parent_bone_name = jbone_name

        # Create a bone dedicated to this body
        # Not strictly neccessary - but this way there is an entry in blender for every Body, which is intuitive
        self._end_bone_name = "to_" + self.get_name()
        if JointInfo.JBP == "PRE":
            bone_tail = own_origin
            bone_head = own_origin - np.array([JointInfo.JBL, 0, 0])
        else:
            bone_tail = own_origin + np.array([JointInfo.JBL, 0, 0])
            bone_head = own_origin
        body_bone = BoneInfo(self._end_bone_name, parent_bone_name, own_origin, bone_tail, False)

        return [*joint_bones, body_bone]


class Scene:
    def __init__(self):
        pass

    def set_bodies(self, bodies):
        self.bodies = bodies

    def get_bodies_in_world_tfs(self):
        return [(b.get_name(), b.get_initial_tf_in_world()) for b in self.bodies]

    def get_all_bodies(self):
        return self.bodies

    def set_joints(self, joints):
        self.joints = joints

    def get_joints_in_world(self):
        return [(j.get_name(), j.get_initial_pos(), j.get_initial_axis()) for j in self.joints]

    def create_all_bones(self):
        # each body has one or more bones:
        # - a bone for each joint, at the joint pos, with bone y == joint axis
        # - a bone near it's origin to represent the body
        all_bones = []
        for body in self.bodies:
            all_bones += body.create_bones()
        return all_bones

    def get_freejoint_pose(self):
        for body in self.bodies:
            if body.get_parent() is not None and body.get_parent().get_parent() is None:
                return body.get_initial_tf_in_world()

def parse_mujoco_xml(filepath, blenderclass):
    tree = ET.parse(filepath)
    root = tree.getroot()
    dirname = os.path.dirname(filepath)

    # process all includes
    while root.findall(".//include"):
        for parent in list(root.findall(".//include/..")):
            to_remove = []
            # Iterate again to get index
            for i, element in enumerate(list(parent)):
                if element.tag != "include":
                    continue
                if BLENDER:
                    # print in blender console
                    blenderclass.report({'INFO'}, "Including: " + element.get("file"))
                else:
                    print("Including: ", element.get("file"))
                subtree = ET.parse(os.path.join(dirname, element.get("file")))
                # Include children of the root element at the same place as the original include
                for c in reversed(subtree.getroot()):
                    parent.insert(i, c)
                to_remove.append(element)
            for e in to_remove:
                parent.remove(e)

    # find all mesh files
    stl_dir = ""
    for c in root.findall("compiler"):
        stl_dir = c.get("meshdir", "")
    stl_dir_full = os.path.join(dirname, stl_dir)
    mesh_files = {}
    for a in root.findall("asset"):
        for m in a.findall("mesh"):
            filename = m.get("file", None)
            if filename is not None:
                mesh_files[os.path.splitext(os.path.split(filename)[-1])[0]] = os.path.join(stl_dir_full, filename)

    joints = []
    bodies = []
    bones = []

    def process_body(body, parent_body):

        body_name = body.get("name", "unnamed_body")
        body_pos = np.array([float(x) for x in body.get("pos", "0 0 0").split()])
        body_wxyz = np.array([float(x) for x in body.get("quat", "1 0 0 0").split()])
        qw, qx, qy, qz = body_wxyz
        body_in_parent = Transform(origin=body_pos, quaternion=Quaternion(qx, qy, qz, qw))
        body_info = BodyInfo(body_name, parent_body, body_in_parent)
        bodies.append(body_info)
        depth = body_info.get_depth()
        if BLENDER:
            # print in blender console
            blenderclass.report({'INFO'}, "-" * depth + body_name)
        else:
            print("-" * depth + body_name)

        for joint in body.findall("joint"):
            joint_name = joint.get("name", "unnamed_joint")
            joint_axis = np.array([float(x) for x in joint.get("axis", "0 0 1").split()])
            joint_pos = np.array([float(x) for x in joint.get("pos", "0 0 0").split()])
            joint_info = JointInfo(joint_name, body_info, parent_body, joint_axis, joint_pos)
            body_info.add_joint(joint_info)
            joints.append(joint_info)

        # add geometries (meshes only for now)
        for geom in body.findall("geom"):
            geom_type = geom.get("type", "unknown")
            geom_class = geom.get("class", "unknown")
            if (geom_type == "mesh" or geom_type == "unknown") and (geom_class == "visual" or geom_class == "visual_light" or geom_class == "visual_dark"):
                geom_meshfile = geom.get("mesh", None)
                if geom_meshfile is not None:
                    body_info.add_mesh_geom(MeshGeomInfo(mesh_files[os.path.splitext(geom_meshfile)[0]]))


        for child_body in body.findall("body"):
            process_body(child_body, body_info)

    for worldbody in root.findall("worldbody"):
        print("Worldbody:", worldbody.get("name", "worldbody"))
        process_body(worldbody, None)

    scene = Scene()
    scene.set_bodies(bodies)
    scene.set_joints(joints)

    return scene


if BLENDER:
    class MUJOCO_OT_import(bpy.types.Operator, ImportHelper):
        """Import a MuJoCo XML File"""
        bl_idname = "import_scene.mujoco_xml"
        bl_label = "Import MuJoCo XML"
        filename_ext = ".xml"
        
        def execute(self, context):
            # Parse the XML file
            scene = parse_mujoco_xml(self.filepath, self)

            # Create armature, bone for each joint
            bpy.ops.object.armature_add(enter_editmode=True)
            armature = bpy.context.object
            armature.name = "MuJoCo_Armature"
            # move armature origin to the first freejoint position
            if not JointInfo.ADD_FREEJOINT_BONE:
                armature.location = scene.get_freejoint_pose().origin()

            bpy.ops.object.mode_set(mode='EDIT')
            edit_bones = armature.data.edit_bones
            default_bone = edit_bones[0]
            edit_bones.remove(default_bone)

            bone_dict = {}

            # Create a custom shape for joint bones (a simple disc)
            bpy.ops.mesh.primitive_circle_add(vertices=32, radius=2.0, fill_type='NOTHING', location=(0, 0, 0))
            custom_shape_obj = bpy.context.object
            custom_shape_obj.name = "Joint_Bone_Shape"

            # Return to the armature object
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='EDIT')

            all_bones = scene.create_all_bones()
            for bone_info in all_bones:
                bone = edit_bones.new(bone_info.name())
                if bone_info.parent_bone_name() is not None:
                    bone.parent = bone_dict[bone_info.parent_bone_name()]
                bone.head = bone_info.head()
                bone.tail = bone_info.tail()
                self.report({'INFO'}, "Created Bone: {} (p: {})".format(bone_info.name(), bone_info.parent_bone_name()))

                # Store bone in the dictionary
                bone_dict[bone_info.name()] = bone

            # Exit Edit Mode to set bone custom shapes
            bpy.ops.object.mode_set(mode='POSE')

            # list all pose bones
            # for b in armature.pose.bones.keys():
            #     self.report({'INFO'}, b)

            pose_bones = armature.pose.bones
            for bone_info in all_bones:
                pose_bone = pose_bones[bone_info.name()]
                if bone_info.is_joint():
                    pose_bone.custom_shape = custom_shape_obj
                    pose_bone.custom_shape_rotation_euler = (np.deg2rad(90), 0, 0)  # Align along the Y-axis if needed
                    pose_bone.rotation_mode = 'XYZ'

            # Attach STL meshes to corresponding bones
            bpy.ops.object.mode_set(mode='OBJECT')
            for body_info in scene.get_all_bodies():
                for geom_info in body_info.get_mesh_geoms():
                    bone_name = body_info._end_bone_name
                    if bone_name is None:
                        self.report({'WARNING'}, "No bone for body {}".format(body_info.get_name()))
                        continue
                    geom_full_path = geom_info.get_path()
                    if geom_full_path is None or not os.path.exists(geom_full_path):
                        continue

                    if geom_full_path.lower().endswith(".stl"):
                        # bpy.ops.import_mesh.stl(filepath=stl_full_path)
                        bpy.ops.wm.stl_import(filepath=geom_full_path)
                    elif geom_full_path.lower().endswith(".obj"):
                        # bpy.ops.import_mesh.stl(filepath=stl_full_path)
                        bpy.ops.wm.obj_import(filepath=geom_full_path)
                    else:
                        self.report({'WARNING'}, "Unkown mesh format {}".format(geom_full_path))
                        continue

                    imported_mesh = bpy.context.object
                    imported_mesh.name = body_info.get_name() + "_mesh"

                    # Parent the mesh to the corresponding bone
                    if bone_name in armature.data.bones:
                        imported_mesh.parent = armature
                        imported_mesh.parent_type = 'BONE'
                        imported_mesh.parent_bone = bone_name
                        imported_mesh.matrix_world = body_info.get_initial_tf_in_world().matrix().T

            return {'FINISHED'}

    def menu_func_import(self, context):
        self.layout.operator(MUJOCO_OT_import.bl_idname, text="MuJoCo XML (.xml)")

    def register():
        bpy.utils.register_class(MUJOCO_OT_import)
        bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

    def unregister():
        bpy.utils.unregister_class(MUJOCO_OT_import)
        bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    if BLENDER:
        register()
