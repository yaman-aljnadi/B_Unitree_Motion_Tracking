"""Plot foot-ground contact and leg torque for a WHAM clip.

The kinematic pass matches teleop_wham.py: joint angles are written directly
and velocities stay zero, so the floor is not holding the robot up. The
physics pass sends those same joint targets to the position actuators and
steps the simulation. The free root can fall. That fall is the stability
result.

    python plot_foot_contact.py 5_full_Body_Updated
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import teleop_fullbody as body
import teleop_wham as wham

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIP = sys.argv[1] if len(sys.argv) > 1 else "5_full_Body_Updated"
LEG_TORQUE_JOINTS = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
)
# Pelvis height in the stand keyframe is 0.79 m. Below this, the physics run has fallen.
FALL_Z = 0.45


def load_clip(clip):
    joints_path = os.path.join(ROOT, f"{clip}_joints.npz")
    video_path = os.path.join(ROOT, f"{clip}.mp4")
    if not os.path.isfile(joints_path):
        raise SystemExit(
            f"Missing {joints_path}. Run WHAM, then export_joints.py, "
            "and copy the npz into Unitree_G1."
        )
    archive = np.load(joints_path)
    joints = archive["joints"]
    trans = archive["trans"]
    if "frame_ids" in archive.files:
        frame_ids = archive["frame_ids"].astype(int)
    else:
        frame_ids = np.arange(joints.shape[0])

    fps = 30.0
    timeline = int(frame_ids.max()) + 1
    if os.path.isfile(video_path):
        import cv2

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if video_fps > 1.0:
            fps = float(video_fps)
        if video_frames > 0:
            timeline = video_frames
    return joints, trans, frame_ids, fps, timeline


def retarget_targets(joints, trans, frame_ids, model, data, stand_qpos, joint_info):
    """Same smoothing as the viewer, stored once for both passes."""
    sample_for_frame = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    smoothed = {name: float(stand_qpos[spec["qadr"]]) for name, spec in joint_info.items()}
    root_yaw = 0.0
    held_index = 0
    timeline = max(sample_for_frame) + 1
    targets = []
    for video_index in range(timeline):
        if video_index in sample_for_frame:
            held_index = sample_for_frame[video_index]
        points = wham.frame_points(joints[held_index])
        for name, angle in body.retarget(points).items():
            spec = joint_info[name]
            clipped = body.clip_target(angle, spec)
            smoothed[name] = (1.0 - body.SMOOTH_ALPHA) * smoothed[name] + body.SMOOTH_ALPHA * clipped
        heading = body.pelvis_heading(points["left_hip"], points["right_hip"])
        if heading is not None:
            root_yaw = body.smooth_angle(root_yaw, heading, body.SMOOTH_ALPHA)
        targets.append((dict(smoothed), trans[held_index, :2].copy(), root_yaw))
    return targets


def foot_bodies(model):
    return {
        "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"),
        "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"),
    }


def sole_height(model, data, body_id):
    """Lowest foot-sphere bottom on this ankle, in meters above the floor."""
    lowest = np.inf
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_SPHERE:
            continue
        bottom = float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
        lowest = min(lowest, bottom)
    return lowest


def foot_site_xy(model, data, side):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
    return data.site_xpos[site_id, :2].copy()


def floor_contact(model, data, bodies, floor_id):
    """Sum normal and tangential floor force on each foot. Newtons."""
    normal = {side: 0.0 for side in bodies}
    tangent = {side: 0.0 for side in bodies}
    body_to_side = {body_id: side for side, body_id in bodies.items()}
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        geoms = (int(contact.geom1), int(contact.geom2))
        if floor_id not in geoms:
            continue
        other = geoms[1] if geoms[0] == floor_id else geoms[0]
        side = body_to_side.get(int(model.geom_bodyid[other]))
        if side is None:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        normal[side] += float(force[0])
        tangent[side] += float(np.hypot(force[1], force[2]))
    return normal, tangent


def kinematic_pass(model, data, stand_qpos, joint_info, ankle_ids, bodies, floor_id, targets, fps):
    frames = len(targets)
    height = {side: np.zeros(frames) for side in bodies}
    speed = {side: np.zeros(frames) for side in bodies}
    normal = {side: np.zeros(frames) for side in bodies}
    previous_xy = {side: None for side in bodies}
    for index, (smoothed, root_xy, root_yaw) in enumerate(targets):
        body.apply_pose(
            model, data, stand_qpos, smoothed, joint_info, root_xy, root_yaw, ankle_ids
        )
        contact_normal, _ = floor_contact(model, data, bodies, floor_id)
        for side, body_id in bodies.items():
            height[side][index] = sole_height(model, data, body_id)
            normal[side][index] = contact_normal[side]
            xy = foot_site_xy(model, data, side)
            if previous_xy[side] is not None:
                speed[side][index] = float(np.linalg.norm(xy - previous_xy[side]) * fps)
            previous_xy[side] = xy
    return height, speed, normal


def physics_pass(model, data, stand_qpos, stand_ctrl, joint_info, bodies, floor_id, targets, fps):
    frames = len(targets)
    normal = {side: np.zeros(frames) for side in bodies}
    tangent = {side: np.zeros(frames) for side in bodies}
    torque = {name: np.zeros(frames) for name in LEG_TORQUE_JOINTS}
    pelvis_z = np.zeros(frames)
    steps = max(1, int(round((1.0 / fps) / model.opt.timestep)))
    fall_time = None

    data.qpos[:] = stand_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = stand_ctrl
    mujoco.mj_forward(model, data)

    for index, (smoothed, _root_xy, _root_yaw) in enumerate(targets):
        for name, angle in smoothed.items():
            data.ctrl[joint_info[name]["actuator"]] = angle
        for _ in range(steps):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            normal = {side: np.full(frames, np.nan) for side in bodies}
            print("Physics pass produced non-finite state and was stopped.")
            return normal, tangent, torque, pelvis_z, fall_time
        contact_normal, contact_tangent = floor_contact(model, data, bodies, floor_id)
        for side in bodies:
            normal[side][index] = contact_normal[side]
            tangent[side][index] = contact_tangent[side]
        for name in LEG_TORQUE_JOINTS:
            torque[name][index] = float(data.actuator_force[joint_info[name]["actuator"]])
        pelvis_z[index] = float(data.qpos[2])
        if fall_time is None and pelvis_z[index] < FALL_Z:
            fall_time = index / fps
    return normal, tangent, torque, pelvis_z, fall_time


def save_figure(clip, time, kin_height, kin_speed, kin_normal, phy_normal, phy_tangent, torque):
    output_dir = os.path.join(ROOT, "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{clip}_foot_contact.png")

    figure, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for side in ("left", "right"):
        axes[0].plot(time, kin_height[side], label=side)
        axes[1].plot(time, kin_speed[side], label=side)
        axes[2].plot(time, kin_normal[side], label=side)
        axes[3].plot(time, phy_normal[side], label=f"{side} normal")
        axes[3].plot(time, phy_tangent[side], linestyle="--", label=f"{side} tangent")
    for name, values in torque.items():
        axes[4].plot(time, values, label=name.replace("_joint", ""))

    axes[0].set_ylabel("Sole height (m)")
    axes[0].set_title("Kinematic playback, the motion in the video")
    axes[1].set_ylabel("Foot speed (m/s)")
    axes[2].set_ylabel("Contact force (N)")
    axes[3].set_ylabel("Force (N)")
    axes[3].set_title("Physics playback, same joint targets")
    axes[4].set_ylabel("Torque (N·m)")
    axes[4].set_xlabel("Time (s)")
    for axis in axes:
        axis.legend(loc="upper right", fontsize=7, ncol=2)
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def summarize(clip, fps, kin_height, kin_speed, kin_normal, phy_normal, torque, pelvis_z, fall_time, joint_info, model):
    near_ground = 0.02
    print(f"clip {clip}, {len(pelvis_z)} frames at {fps:.2f} fps")
    for side in ("left", "right"):
        stance = kin_height[side] < near_ground
        slip = float(np.mean(kin_speed[side][stance])) if np.any(stance) else float("nan")
        print(
            f"kinematic {side}: sole {kin_height[side].min():.3f} to {kin_height[side].max():.3f} m, "
            f"mean speed while sole < {near_ground:.2f} m: {slip:.3f} m/s, "
            f"peak contact {kin_normal[side].max():.1f} N"
        )
        print(f"physics {side}: peak normal {np.nanmax(phy_normal[side]):.1f} N")
    print(
        f"physics pelvis min {np.nanmin(pelvis_z):.3f} m, final {pelvis_z[-1]:.3f} m"
    )
    if fall_time is None:
        print(f"physics pelvis stayed above {FALL_Z:.2f} m")
        before_fall = np.ones(len(pelvis_z), dtype=bool)
    else:
        print(f"physics pelvis fell below {FALL_Z:.2f} m at {fall_time:.2f} s")
        before_fall = np.arange(len(pelvis_z)) / fps < fall_time
    for name in LEG_TORQUE_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        limit = float(np.max(np.abs(model.jnt_actfrcrange[joint_id])))
        peak = float(np.nanmax(np.abs(torque[name])))
        early = torque[name][before_fall]
        early_peak = float(np.nanmax(np.abs(early))) if early.size else float("nan")
        print(
            f"torque {name}: peak before fall {early_peak:.1f} N·m, "
            f"peak overall {peak:.1f} N·m, joint limit {limit:.1f} N·m"
        )


def main():
    joints, trans, frame_ids, fps, timeline = load_clip(CLIP)
    model = mujoco.MjModel.from_xml_path(body.MODEL_PATH)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise SystemExit("G1 model has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    stand_qpos = data.qpos.copy()
    stand_ctrl = data.ctrl.copy()
    joint_info = body.joint_addresses(model)
    ankle_ids = list(foot_bodies(model).values())
    bodies = foot_bodies(model)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    targets = retarget_targets(joints, trans, frame_ids, model, data, stand_qpos, joint_info)
    if timeline < len(targets):
        targets = targets[:timeline]
    elif timeline > len(targets) and targets:
        targets = targets + [targets[-1]] * (timeline - len(targets))

    kin_height, kin_speed, kin_normal = kinematic_pass(
        model, data, stand_qpos, joint_info, ankle_ids, bodies, floor_id, targets, fps
    )
    phy_normal, phy_tangent, torque, pelvis_z, fall_time = physics_pass(
        model, data, stand_qpos, stand_ctrl, joint_info, bodies, floor_id, targets, fps
    )
    time = np.arange(len(targets)) / fps
    output_path = save_figure(
        CLIP, time, kin_height, kin_speed, kin_normal, phy_normal, phy_tangent, torque
    )
    summarize(
        CLIP, fps, kin_height, kin_speed, kin_normal, phy_normal, torque, pelvis_z, fall_time, joint_info, model
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
