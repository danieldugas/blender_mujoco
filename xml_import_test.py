import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt

from mujoco_importer import parse_mujoco_xml

def plot_joints(scene):
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    axdict = {"Side": axs[0][0], "Top": axs[1][0], "Front": axs[0][1]}
    axdims = {"Side": [0, 2], "Top": [0, 1], "Front": [1, 2]}
    for i, (view, ax) in enumerate(axdict.items()):
        ax.set_title(view + " View")
        ax.axis('equal')
    for body_name, body_tf in scene.get_bodies_in_world_tfs():
        # Plot the body
        body_pos = body_tf.origin()
        body_x_axis = body_tf.x_axis()
        body_y_axis = body_tf.y_axis()
        body_z_axis = body_tf.z_axis()
        for view, ax in axdict.items():
            S = 0.05
            ax.plot([body_pos[axdims[view][0]], body_pos[axdims[view][0]] + S * body_x_axis[axdims[view][0]]], [body_pos[axdims[view][1]], body_pos[axdims[view][1]] + S * body_x_axis[axdims[view][1]],], color='r')
            ax.plot([body_pos[axdims[view][0]], body_pos[axdims[view][0]] + S * body_y_axis[axdims[view][0]]], [body_pos[axdims[view][1]], body_pos[axdims[view][1]] + S * body_y_axis[axdims[view][1]],], color='g')
            ax.plot([body_pos[axdims[view][0]], body_pos[axdims[view][0]] + S * body_z_axis[axdims[view][0]]], [body_pos[axdims[view][1]], body_pos[axdims[view][1]] + S * body_z_axis[axdims[view][1]],], color='b')
            ax.text(body_pos[axdims[view][0]], body_pos[axdims[view][1]], body_name, fontsize=8, alpha=0.2)
    for joint_name, joint_pos, joint_axis in scene.get_joints_in_world():
        for view, ax in axdict.items():
            S = 0.1
            ax.plot([joint_pos[axdims[view][0]], joint_pos[axdims[view][0]] + S * joint_axis[axdims[view][0]]], [joint_pos[axdims[view][1]], joint_pos[axdims[view][1]] + S * joint_axis[axdims[view][1]],], color='y')
            ax.text(joint_pos[axdims[view][0]], joint_pos[axdims[view][1]], joint_name, fontsize=8, alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_bones(scene):
    bones = scene.create_all_bones()
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    axdict = {"Side": axs[0][0], "Top": axs[1][0], "Front": axs[0][1]}
    axdims = {"Side": [0, 2], "Top": [0, 1], "Front": [1, 2]}
    for i, (view, ax) in enumerate(axdict.items()):
        ax.set_title(view + " View")
        ax.axis('equal')
    for bone in bones:
        print("{} ({})".format(bone.name(), bone.parent_bone_name()))
        for view, ax in axdict.items():
            dim1, dim2 = axdims[view]
            head = bone.head()
            tail = bone.tail()
            delta = tail - head
            L = np.linalg.norm(delta)
            W = L * 0.2
            dd = np.array([delta[dim1], delta[dim2]])
            pp = np.array([delta[dim2], -delta[dim1]]) / np.linalg.norm(dd) * W
            hh = np.array([head[dim1], head[dim2]])
            tt = np.array([tail[dim1], tail[dim2]])
            triangle = np.array([
                hh + pp,
                tt,
                hh - pp,
                hh + pp
            ])
            ax.plot(triangle[:, 0], triangle[:, 1], color=('r' if bone.is_joint() else 'k'))
            ax.text(*((hh+tt)/2.0), bone.name(), fontsize=8, alpha=0.2)
    plt.tight_layout()
    plt.show()


            



if __name__ == "__main__":
    filepath = os.path.expanduser(
        "~/Code/DeepMimic_mujoco/src/mujoco/humanoid_deepmimic/envs/asset/deepmimic_unitree_g1.xml"
    )
    scene = parse_mujoco_xml(filepath, None)
    # plot_joints(scene)
    plot_bones(scene)