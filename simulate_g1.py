import mujoco
import mujoco.viewer
import time

# Point this to the scene.xml file you just downloaded
xml_path = "../mujoco_menagerie#getting-started/unitree_g1/scene.xml"

# 1. Load the physics model and create a data state
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. Launch the graphical viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    print("Simulation started! Close the window to exit.")
    
    # 3. The main simulation loop
    while viewer.is_running():
        step_start = time.time()

        # Tell MuJoCo to calculate the next frame of physics
        mujoco.mj_step(model, data)

        # Update the visualizer
        viewer.sync()

        # Keep the simulation running at roughly real-time speed
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)