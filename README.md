# Unitree G1 full-body retarget

MediaPipe reads a video of a person and writes matching joint angles into a Unitree G1 in MuJoCo. The pose is kinematic: the viewer shows the body copying the video. The robot is not balance-controlled, so it does not walk under physics. The pelvis turns with the hips, slides with the person, and the lower foot stays on the floor.

## Folders Path

```
  Unitree_G1/                              scripts and videos
  mujoco_menagerie#getting-started/
    unitree_g1/scene.xml                   G1 model the scripts load
```

Leave the menagerie folder name as it is, including the `#`. The scripts load that path. Keep `4_legs.mp4` and `3_full_body.mp4` inside `Unitree_G1`.

## Setup

using Python 3.10 and a machine with a local display. The MuJoCo window and the OpenCV window both open on that display.

From `Unitree_G1`:

## Personally I'm using Conda as my python environment with Python 3.10
``` inside the console
python -m pip install -r requirements.txt
```


`mujoco` from that file includes the interactive viewer. MediaPipe is pinned to 0.10.9 because the scripts use `mp.solutions`. Later MediaPipe releases removed that API.

## Tests

Run each command from `Unitree_G1`, with the virtual environment active.

1. MuJoCo and the G1 model:

   ```inside the console
   python simulate_g1.py
   ```

   The G1 stands in the MuJoCo window. Close the window to exit.

2. MediaPipe and OpenCV:

   ```inside the console
   python track_pose.py
   ```

   A skeleton is drawn on `3_full_body.mp4`. Press ESC to quit.

3. Full-body retarget:

   ```inside the console
   python teleop_fullbody.py
   ```

   The MuJoCo window and a video window open together. The clip is whatever `VIDEO_PATH` is set to at the top of `teleop_fullbody.py` (currently `4_legs.mp4`). Press ESC in the video window to quit.

   The arms follow the person. The text line labeled `Root` shows hip yaw and the position on the floor. The lower foot stays on the ground.

`simulate_g1.py` and `track_pose.py` look up their files from the working directory, so start them from `Unitree_G1`. `teleop_fullbody.py` finds its video and the G1 model from the script's own location.

To try another clip, change `VIDEO_PATH` at the top of `teleop_fullbody.py` to a video inside `Unitree_G1`.


