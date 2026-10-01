import os

import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np

# Video and model live next to this script. 3_full_body.mp4 is the full-body clip.
ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(ROOT, "4_legs.mp4")
MODEL_PATH = os.path.join(
    ROOT, "..", "mujoco_menagerie#getting-started", "unitree_g1", "scene.xml"
)

# Fraction of each new sample mixed into the held angle. Lower is smoother.
SMOOTH_ALPHA = 0.35
# Arms are reliable above this. Legs in a webcam clip often score much lower.
VISIBILITY_MIN = 0.5
LEG_VISIBILITY_MIN = 0.15
# Typical distance to the camera, so a change in shoulder width becomes a step.
CAMERA_DISTANCE_M = 2.5
# Largest pelvis slide accepted in one frame, so a bad detection cannot throw the robot.
MAX_ROOT_STEP = 0.05
# Foot spheres sit this far below the ankle body. Grounding to this height puts the sole on z=0.
SOLE_Z = 0.03

# G1 elbow zero is a right-angle bend in the mesh. A straight arm is +pi/2,
# and further flexion goes back toward zero and then negative.
ELBOW_STRAIGHT = np.pi / 2

# Wrists have no orientation in body pose. Ankle roll is too noisy from one camera.
# Those joints stay at the stand keyframe.
CONTROLLED_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# MediaPipe pose indices. Left/right are the person's, not the image's.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

ARM_LANDMARKS = {
    "left_shoulder": L_SHOULDER,
    "right_shoulder": R_SHOULDER,
    "left_elbow": L_ELBOW,
    "right_elbow": R_ELBOW,
    "left_wrist": L_WRIST,
    "right_wrist": R_WRIST,
}
LEG_LANDMARKS = {
    "left_hip": L_HIP,
    "right_hip": R_HIP,
    "left_knee": L_KNEE,
    "right_knee": R_KNEE,
    "left_ankle": L_ANKLE,
    "right_ankle": R_ANKLE,
    "left_foot": L_FOOT,
    "right_foot": R_FOOT,
}


def normalize(vec):
    """Return a unit vector, or None when the input is too short to have a direction."""
    n = np.linalg.norm(vec)
    if n < 1e-8:
        return None
    return vec / n


def interior_angle(a, b, c):
    """Angle at b between points a-b-c, in radians. None if a segment has no length."""
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-8 or nc < 1e-8:
        return None
    cosine = np.dot(ba, bc) / (na * nc)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def landmark_to_robot(lm):
    """Map one MediaPipe world landmark into the G1 frame.

    MediaPipe world landmarks are meters, hip origin, image-right +x, down +y,
    and depth that decreases toward the camera. The person is facing the camera.
    G1 is Z-up, +X forward, +Y to the robot's left.

    Image-right is the person's left, which is the robot's +Y, so x is not
    negated. The person's right landmarks still drive the robot's right joints.
    Reaching toward the camera becomes +X.
    """
    return np.array([-lm.z, lm.x, -lm.y], dtype=np.float64)


def pitch_roll(vec, forward, left, up):
    """Split a limb direction into pitch then roll, matching G1 joint order.

    Positive pitch swings the limb backward. Positive roll swings it toward +Y
    (out for a left limb, in for a right limb). A limb straight along -up is 0, 0.
    """
    unit = normalize(vec)
    if unit is None:
        return None, None
    forward_c = float(np.dot(unit, forward))
    left_c = float(np.dot(unit, left))
    up_c = float(np.dot(unit, up))
    sagittal = float(np.hypot(forward_c, up_c))
    roll = float(np.arctan2(left_c, sagittal))
    if sagittal < 1e-3:
        pitch = 0.0
    else:
        # Derived from R_y(pitch) * R_x(roll) * (0, 0, -1).
        # Forward is negative pitch on this model; hanging is 0.
        pitch = float(np.arctan2(-forward_c, -up_c))
    return pitch, roll


def swing_twist(origin, joint, tip, reference):
    """Twist of joint->tip around origin->joint.

    None when the distal segment is nearly parallel to the limb, because the
    twist is undefined with the arm or leg straight. Positive means the bend
    turns toward cross(reference, limb axis). With the arm hanging and
    reference = forward, a bend toward +Y is positive, which matches G1 yaw.
    """
    axis = normalize(joint - origin)
    if axis is None:
        return None
    distal = tip - joint
    distal_len = np.linalg.norm(distal)
    if distal_len < 1e-8:
        return None
    bend = distal - axis * np.dot(distal, axis)
    if np.linalg.norm(bend) / distal_len < 0.2:
        return None
    bend = normalize(bend)
    ref = reference - axis * np.dot(reference, axis)
    ref = normalize(ref)
    if bend is None or ref is None:
        return None
    side = normalize(np.cross(ref, axis))
    if side is None:
        return None
    return float(np.arctan2(np.dot(bend, side), np.dot(bend, ref)))


def ankle_pitch(knee, ankle, foot, left):
    """Signed ankle pitch. Toes-down is positive, a flat foot is 0."""
    shin = normalize(ankle - knee)
    foot_dir = normalize(foot - ankle)
    if shin is None or foot_dir is None:
        return None
    forward = normalize(np.cross(shin, left))
    if forward is None:
        return None
    return float(np.arctan2(np.dot(foot_dir, shin), np.dot(foot_dir, forward)))


def pelvis_heading(left_hip, right_hip):
    """Yaw of the whole pelvis. Facing the camera is 0; turning to the person's left is positive.

    The free-joint yaw uses the same sign: positive rotation about +Z.
    """
    left = np.array(left_hip - right_hip, dtype=np.float64)
    left[2] = 0.0
    if np.linalg.norm(left) < 1e-6:
        return None
    return float(np.arctan2(-left[0], left[1]))


def smooth_angle(held, sample, alpha):
    """Blend two angles without a jump when they wrap past ±pi."""
    delta = (sample - held + np.pi) % (2.0 * np.pi) - np.pi
    return float(held + alpha * delta)


def yaw_to_quat(yaw):
    """Free-joint quaternion for a yaw about +Z. MuJoCo order is w, x, y, z."""
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def walk_target(image_landmarks, world_points, reference):
    """Hip motion in the image, in meters, relative to the first good frame.

    Image-right is the robot's +Y. Wider shoulders mean the person stepped toward
    the camera, which is +X. Returns (xy, reference). xy is None when the frame
    cannot be used; the reference is cleared only by the caller.
    """
    checks = (
        (L_SHOULDER, VISIBILITY_MIN),
        (R_SHOULDER, VISIBILITY_MIN),
        (L_HIP, LEG_VISIBILITY_MIN),
        (R_HIP, LEG_VISIBILITY_MIN),
    )
    for index, limit in checks:
        if image_landmarks[index].visibility < limit:
            return None, reference
    if "left_shoulder" not in world_points or "right_shoulder" not in world_points:
        return None, reference

    left_shoulder = image_landmarks[L_SHOULDER]
    right_shoulder = image_landmarks[R_SHOULDER]
    left_hip = image_landmarks[L_HIP]
    right_hip = image_landmarks[R_HIP]
    shoulder_image = float(
        np.hypot(left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y)
    )
    shoulder_m = float(
        np.linalg.norm(world_points["left_shoulder"] - world_points["right_shoulder"])
    )
    if shoulder_image < 1e-4 or shoulder_m < 1e-4:
        return None, reference

    hip_x = 0.5 * (left_hip.x + right_hip.x)
    if reference is None:
        reference = {
            "hip_x": hip_x,
            "shoulder_image": shoulder_image,
            "shoulder_m": shoulder_m,
        }
        return np.zeros(2, dtype=np.float64), reference

    scale = reference["shoulder_m"] / reference["shoulder_image"]
    lateral = (hip_x - reference["hip_x"]) * scale
    width_ratio = float(np.clip(shoulder_image / reference["shoulder_image"], 0.5, 2.0))
    forward = CAMERA_DISTANCE_M * (1.0 - 1.0 / width_ratio)
    return np.array([forward, lateral], dtype=np.float64), reference


def limit_step(current, target, max_step):
    """Move current toward target by at most max_step meters."""
    delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length > max_step:
        delta = delta * (max_step / length)
    return np.asarray(current, dtype=np.float64) + delta


def pelvis_axes(left_hip, right_hip):
    """Horizontal pelvis frame. Legs use this so a waist lean does not fold the hips."""
    world_up = np.array([0.0, 0.0, 1.0])
    left = left_hip - right_hip
    left = left - world_up * np.dot(left, world_up)
    left = normalize(left)
    if left is None:
        left = np.array([0.0, 1.0, 0.0])
    forward = normalize(np.cross(left, world_up))
    if forward is None:
        forward = np.array([1.0, 0.0, 0.0])
    return forward, left, world_up


def torso_axes(left_shoulder, right_shoulder, left_hip, right_hip):
    """Torso frame from the spine and the shoulder line. Used for the arms."""
    up = normalize(0.5 * (left_shoulder + right_shoulder) - 0.5 * (left_hip + right_hip))
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    left = left_shoulder - right_shoulder
    left = left - up * np.dot(left, up)
    left = normalize(left)
    if left is None:
        left = np.array([0.0, 1.0, 0.0])
    forward = normalize(np.cross(left, up))
    if forward is None:
        forward = np.array([1.0, 0.0, 0.0])
    left = normalize(np.cross(up, forward))
    if left is None:
        left = np.array([0.0, 1.0, 0.0])
    return forward, left, up


def _has(points, *names):
    return all(name in points for name in names)


def retarget(points):
    """Map G1-frame body points to joint angles.

    points maps landmark names to (3,) arrays. Missing landmarks mean that
    limb is left out of the returned dict so the caller keeps the last angle.
    """
    targets = {}

    if _has(points, "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        l_sh = points["left_shoulder"]
        r_sh = points["right_shoulder"]
        l_hip = points["left_hip"]
        r_hip = points["right_hip"]
        forward, left, up = torso_axes(l_sh, r_sh, l_hip, r_hip)

        shoulder_line = l_sh - r_sh
        hip_line = l_hip - r_hip
        shoulder_h = shoulder_line.copy()
        hip_h = hip_line.copy()
        shoulder_h[2] = 0.0
        hip_h[2] = 0.0
        if np.linalg.norm(shoulder_h) > 1e-6 and np.linalg.norm(hip_h) > 1e-6:
            cross_z = hip_h[0] * shoulder_h[1] - hip_h[1] * shoulder_h[0]
            targets["waist_yaw_joint"] = float(
                np.arctan2(cross_z, np.dot(hip_h, shoulder_h))
            )

        width = float(np.hypot(shoulder_line[0], shoulder_line[1]))
        if width > 1e-6:
            targets["waist_roll_joint"] = float(
                np.arctan2(l_sh[2] - r_sh[2], width)
            )

        spine = 0.5 * (l_sh + r_sh) - 0.5 * (l_hip + r_hip)
        pelvis_forward, _, _ = pelvis_axes(l_hip, r_hip)
        spine_up = float(np.dot(spine, np.array([0.0, 0.0, 1.0])))
        spine_fwd = float(np.dot(spine, pelvis_forward))
        if abs(spine_up) + abs(spine_fwd) > 1e-6:
            targets["waist_pitch_joint"] = float(np.arctan2(spine_fwd, spine_up))

        for side, prefix in (("left", "left"), ("right", "right")):
            shoulder_name = f"{side}_shoulder"
            elbow_name = f"{side}_elbow"
            wrist_name = f"{side}_wrist"
            if not _has(points, shoulder_name, elbow_name):
                continue
            shoulder = points[shoulder_name]
            elbow = points[elbow_name]
            pitch, roll = pitch_roll(elbow - shoulder, forward, left, up)
            if pitch is not None:
                targets[f"{prefix}_shoulder_pitch_joint"] = pitch
                targets[f"{prefix}_shoulder_roll_joint"] = roll
            if _has(points, wrist_name):
                wrist = points[wrist_name]
                bend = interior_angle(shoulder, elbow, wrist)
                if bend is not None:
                    targets[f"{prefix}_elbow_joint"] = bend - ELBOW_STRAIGHT
                yaw = swing_twist(shoulder, elbow, wrist, forward)
                if yaw is not None:
                    targets[f"{prefix}_shoulder_yaw_joint"] = yaw

        pelvis_forward, pelvis_left, pelvis_up = pelvis_axes(l_hip, r_hip)
        for side, prefix in (("left", "left"), ("right", "right")):
            hip_name = f"{side}_hip"
            knee_name = f"{side}_knee"
            ankle_name = f"{side}_ankle"
            foot_name = f"{side}_foot"
            if not _has(points, hip_name, knee_name):
                continue
            hip = points[hip_name]
            knee = points[knee_name]
            pitch, roll = pitch_roll(knee - hip, pelvis_forward, pelvis_left, pelvis_up)
            if pitch is not None:
                targets[f"{prefix}_hip_pitch_joint"] = pitch
                targets[f"{prefix}_hip_roll_joint"] = roll
            if not _has(points, ankle_name):
                continue
            ankle = points[ankle_name]
            knee_bend = interior_angle(hip, knee, ankle)
            if knee_bend is not None:
                targets[f"{prefix}_knee_joint"] = np.pi - knee_bend
            # The knee bends backward, so zero yaw is -forward, not forward.
            yaw = swing_twist(hip, knee, ankle, -pelvis_forward)
            if yaw is not None:
                targets[f"{prefix}_hip_yaw_joint"] = yaw
            if _has(points, foot_name):
                pitch_ankle = ankle_pitch(knee, ankle, points[foot_name], pelvis_left)
                if pitch_ankle is not None:
                    targets[f"{prefix}_ankle_pitch_joint"] = pitch_ankle

    return targets


def visible_points(world_landmarks):
    """Keep landmarks above the visibility cutoff, already in the G1 frame.

    Arms stay strict. Legs use a lower cutoff so a step still bends the knee
    when the feet are only weakly detected.
    """
    points = {}
    for name, index in ARM_LANDMARKS.items():
        lm = world_landmarks[index]
        if lm.visibility < VISIBILITY_MIN:
            continue
        points[name] = landmark_to_robot(lm)
    for name, index in LEG_LANDMARKS.items():
        lm = world_landmarks[index]
        if lm.visibility < LEG_VISIBILITY_MIN:
            continue
        points[name] = landmark_to_robot(lm)
    return points


def joint_addresses(model):
    """qpos index and limits for every controlled joint, and the matching actuator."""
    info = {}
    for name in CONTROLLED_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"G1 model is missing joint or actuator {name}")
        info[name] = {
            "qadr": int(model.jnt_qposadr[joint_id]),
            "lo": float(model.jnt_range[joint_id, 0]),
            "hi": float(model.jnt_range[joint_id, 1]),
            "actuator": int(actuator_id),
        }
    return info


def clip_target(angle, info):
    return float(np.clip(angle, info["lo"], info["hi"]))


def ground_feet(model, data, ankle_ids):
    """Shift the pelvis height so the lower ankle sole rests on the floor."""
    lowest = min(float(data.xpos[body_id, 2]) for body_id in ankle_ids)
    data.qpos[2] += SOLE_Z - lowest
    mujoco.mj_forward(model, data)


def apply_pose(model, data, stand_qpos, smoothed, joints, root_xy, root_yaw, ankle_ids):
    """Write the smoothed pose and refresh kinematics without stepping physics."""
    data.qpos[:] = stand_qpos
    data.qpos[0] = stand_qpos[0] + float(root_xy[0])
    data.qpos[1] = stand_qpos[1] + float(root_xy[1])
    data.qpos[3:7] = yaw_to_quat(root_yaw)
    for name, angle in smoothed.items():
        spec = joints[name]
        data.qpos[spec["qadr"]] = clip_target(angle, spec)
        data.ctrl[spec["actuator"]] = data.qpos[spec["qadr"]]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    ground_feet(model, data, ankle_ids)


def draw_angles(image, smoothed, root_yaw, root_xy):
    """Overlay a short readout of the angles currently driving the robot."""
    lines = []
    pairs = [
        ("L arm", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint"),
        ("R arm", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint"),
        ("L leg", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint"),
        ("R leg", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"),
        ("Waist", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
    ]
    lines.append(
        f"Root: yaw {np.degrees(root_yaw):+.0f}  x {root_xy[0]:+.2f}  y {root_xy[1]:+.2f}"
    )
    for label, *names in pairs:
        bits = [f"{np.degrees(smoothed[name]):+.0f}" for name in names]
        lines.append(f"{label}: {'  '.join(bits)}")

    y = 24
    for line in lines:
        cv2.putText(
            image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 22


def main():
    if not os.path.isfile(VIDEO_PATH):
        raise SystemExit(f"Video not found: {VIDEO_PATH}")
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_dt = 1.0 / fps if fps and fps > 1.0 else 1.0 / 30.0

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise SystemExit("G1 model has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    stand_qpos = data.qpos.copy()
    joints = joint_addresses(model)
    smoothed = {name: float(stand_qpos[spec["qadr"]]) for name, spec in joints.items()}
    ankle_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    if any(body_id < 0 for body_id in ankle_ids):
        raise SystemExit("G1 model is missing an ankle body")
    root_xy = np.zeros(2, dtype=np.float64)
    root_yaw = 0.0
    walk_reference = None

    print("Kinematic full-body retarget. ESC in the video window quits.")
    print(f"Video: {VIDEO_PATH}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while cap.isOpened() and viewer.is_running():
            ok, image = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                walk_reference = None
                root_xy[:] = 0.0
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
            if results.pose_world_landmarks:
                points = visible_points(results.pose_world_landmarks.landmark)
                for name, angle in retarget(points).items():
                    spec = joints[name]
                    held = smoothed[name]
                    sample = clip_target(angle, spec)
                    smoothed[name] = (1.0 - SMOOTH_ALPHA) * held + SMOOTH_ALPHA * sample
                if "left_hip" in points and "right_hip" in points:
                    heading = pelvis_heading(points["left_hip"], points["right_hip"])
                    if heading is not None:
                        root_yaw = smooth_angle(root_yaw, heading, SMOOTH_ALPHA)
                if results.pose_landmarks:
                    target_xy, walk_reference = walk_target(
                        results.pose_landmarks.landmark, points, walk_reference
                    )
                    if target_xy is not None:
                        stepped = limit_step(root_xy, target_xy, MAX_ROOT_STEP)
                        root_xy = (1.0 - SMOOTH_ALPHA) * root_xy + SMOOTH_ALPHA * stepped

            apply_pose(model, data, stand_qpos, smoothed, joints, root_xy, root_yaw, ankle_ids)
            viewer.sync()
            draw_angles(image, smoothed, root_yaw, root_xy)
            cv2.imshow("Full-body retarget", image)
            if cv2.waitKey(max(1, int(frame_dt * 1000))) & 0xFF == 27:
                break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
