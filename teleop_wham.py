"""Play a WHAM joint export on the Unitree G1.

The npz file is written by export_joints.py inside the WHAM environment.
This script stays in the Unitree environment and does not need PyTorch.
"""
import os
import sys

import cv2
import mujoco
import mujoco.viewer
import numpy as np

import teleop_fullbody as body

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIP = sys.argv[1] if len(sys.argv) > 1 else "4_legs"
JOINTS_PATH = os.path.join(ROOT, f"{CLIP}_joints.npz")
VIDEO_PATH = os.path.join(ROOT, f"{CLIP}.mp4")

# Names stored by export_joints.py, in SMPL order.
NAME_TO_INDEX = {
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
}


def frame_points(joints):
    return {name: joints[index] for name, index in NAME_TO_INDEX.items()}


def main():
    if not os.path.isfile(JOINTS_PATH):
        raise SystemExit(
            f"Missing {JOINTS_PATH}. Run WHAM, then export_joints.py, "
            "and copy the npz into Unitree_G1."
        )

    archive = np.load(JOINTS_PATH)
    joints = archive["joints"]
    trans = archive["trans"]
    if "frame_ids" in archive.files:
        frame_ids = archive["frame_ids"].astype(int)
    else:
        frame_ids = np.arange(joints.shape[0])
    sample_for_frame = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}

    model = mujoco.MjModel.from_xml_path(body.MODEL_PATH)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise SystemExit("G1 model has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    stand_qpos = data.qpos.copy()
    joint_info = body.joint_addresses(model)
    smoothed = {name: float(stand_qpos[spec["qadr"]]) for name, spec in joint_info.items()}
    ankle_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    root_yaw = 0.0

    cap = cv2.VideoCapture(VIDEO_PATH) if os.path.isfile(VIDEO_PATH) else None
    fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 0.0
    frame_dt = 1.0 / fps if fps and fps > 1.0 else 1.0 / 30.0

    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None else 0
    timeline = video_frames if video_frames > 0 else int(frame_ids.max()) + 1
    print(
        f"WHAM playback for {CLIP}: {len(frame_ids)} tracked poses, "
        f"{timeline} video frames. ESC in the video window quits."
    )
    video_index = 0
    held_index = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if video_index in sample_for_frame:
                held_index = sample_for_frame[video_index]
            sample = joints[held_index]
            points = frame_points(sample)
            for name, angle in body.retarget(points).items():
                spec = joint_info[name]
                held = smoothed[name]
                clipped = body.clip_target(angle, spec)
                smoothed[name] = (1.0 - body.SMOOTH_ALPHA) * held + body.SMOOTH_ALPHA * clipped
            heading = body.pelvis_heading(points["left_hip"], points["right_hip"])
            if heading is not None:
                root_yaw = body.smooth_angle(root_yaw, heading, body.SMOOTH_ALPHA)
            root_xy = trans[held_index, :2]
            body.apply_pose(
                model, data, stand_qpos, smoothed, joint_info, root_xy, root_yaw, ankle_ids
            )
            viewer.sync()

            image = None
            if cap is not None:
                ok, image = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, image = cap.read()
            if image is None:
                image = np.zeros((240, 640, 3), dtype=np.uint8)
            body.draw_angles(image, smoothed, root_yaw, root_xy)
            cv2.imshow("WHAM retarget", image)
            if cv2.waitKey(max(1, int(frame_dt * 1000))) & 0xFF == 27:
                break
            video_index = (video_index + 1) % timeline

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
