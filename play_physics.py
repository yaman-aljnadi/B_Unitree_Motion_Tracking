"""Play the WHAM joint targets with physics, so the feet can touch the floor.

This is the run behind the force plot. The root is not held up, so the robot
can fall. Close the MuJoCo window, or press ESC in the video window, to quit.

    python play_physics.py 5_full_Body_Updated
"""
import os
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

import plot_foot_contact as contact
import teleop_fullbody as body

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    clip = contact.CLIP
    joints, trans, frame_ids, fps, timeline = contact.load_clip(clip)
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
    targets = contact.retarget_targets(
        joints, trans, frame_ids, model, data, stand_qpos, joint_info
    )
    if timeline < len(targets):
        targets = targets[:timeline]
    elif timeline > len(targets) and targets:
        targets = targets + [targets[-1]] * (timeline - len(targets))

    frame_dt = 1.0 / fps
    steps = max(1, int(round(frame_dt / model.opt.timestep)))
    video_path = os.path.join(ROOT, f"{clip}.mp4")
    cap = cv2.VideoCapture(video_path) if os.path.isfile(video_path) else None

    print(
        f"Physics playback for {clip}. The robot is not held upright. "
        "On this clip it falls at about 0.8 s, then the clip restarts. "
        "Close the MuJoCo window, or press ESC in the video window, to quit."
    )

    def reset():
        data.qpos[:] = stand_qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = stand_ctrl
        mujoco.mj_forward(model, data)
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    reset()
    index = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            smoothed = targets[index][0]
            for name, angle in smoothed.items():
                data.ctrl[joint_info[name]["actuator"]] = angle
            for _ in range(steps):
                mujoco.mj_step(model, data)
            viewer.sync()

            image = None
            if cap is not None:
                ok, image = cap.read()
                if not ok:
                    image = None
            if image is None:
                image = np.zeros((240, 640, 3), dtype=np.uint8)
            cv2.imshow("Physics retarget", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            index += 1
            if index >= len(targets) or not np.isfinite(data.qpos).all():
                reset()
                index = 0
            time.sleep(frame_dt)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
