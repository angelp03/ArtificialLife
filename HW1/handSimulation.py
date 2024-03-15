import mujoco
import mujoco_viewer
import numpy as np

xml = """<mujoco>
    <option gravity = '0 0 -9.81'/>
    <worldbody>
        <light diffuse = '.5 .5 .5' pos = '0 0 6' dir = '0 0 -1'/>
        <geom type = 'plane' size = '10 10 0.1' rgba = '1 1 1 1'/>
        <body name = 'body' pos = '0 0 1'>
            <joint type = 'free'/>
            <geom type="sphere" size=".3" rgba="0 1 0 1" mass='0.1'/>
        <body name = 'model' pos='0 0 0' euler='0 0 0'>

        <body name = '0' pos='0.5 -1.2246467991473532e-16 0' euler='0 0 360.0'>   
            <joint type='hinge' name='0_joint' axis='-1 0 0' range='0 35'/><geom type="box" size=".1 .2 .1" rgba="1 0 0 1"/>
        <body name = '0_0' pos='0 .2 -.1' euler='-90.0 0 0'>
            <geom type="box" size=".1 .2 .1" rgba="1 0 0 1"/>
        </body>
        </body>
        <body name = '1' pos='-0.5 6.123233995736766e-17 0' euler='0 0 180.0'>    
            <joint type='hinge' name='1_joint' axis='-1 0 0' range='0 35'/><geom type="box" size=".1 .2 .1" rgba="1 0 0 1"/>
        <body name = '1_0' pos='0 .2 -.1' euler='-90.0 0 0'>
            <geom type="box" size=".1 .2 .1" rgba="1 0 0 1"/><joint type='hinge' name='1_0_joint' axis='-1 0 0' range='0 35'/>
        </body>
        </body>
        </body>
        </body>
    </worldbody>
    <actuator>
        <motor name = '0_motor' joint = '0_joint'/><motor name = '1_0_motor' joint = '1_0_joint'/><motor name = '1_motor' joint = '1_joint'/>
    </actuator>
</mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


viewer = mujoco_viewer.MujocoViewer(model, data)
actuators = model.nu
#array of velocities

for i in range(10000):
    if viewer.is_alive:
        # when all joints have reached terminal velocity, invert
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break


viewer.close()