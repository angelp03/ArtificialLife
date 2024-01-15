import mujoco
import mujoco_viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('/Users/angelp/Desktop/ArtificialLife/HW1/legmovement.xml')
data = mujoco.MjData(model)


viewer = mujoco_viewer.MujocoViewer(model, data)
actuators = model.nu
velocities = np.array ([800, 300, 800, 300, 800, 300, 800, 300])

for i in range(10000):
    if viewer.is_alive:
        if i % 500 == 0: # invert velocities every 500 steps.
            velocities *= -1
        data.ctrl[:actuators] = velocities
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break


viewer.close()