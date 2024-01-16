import mujoco
import mujoco_viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('/Users/angelp/Desktop/ArtificialLife/HW1/simpleHand.xml')
data = mujoco.MjData(model)


viewer = mujoco_viewer.MujocoViewer(model, data)
actuators = model.nu
#array of velocities
velocities = np.array ([800, 300, 800, 300, 800, 300, 800, 300])

for i in range(10000):
    if viewer.is_alive:
        # when all joints have reached terminal velocity, invert
        if np.all(np.abs(data.qvel) < 0.01):
            data.ctrl[:actuators] = velocities
            velocities *= -1
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break


viewer.close()