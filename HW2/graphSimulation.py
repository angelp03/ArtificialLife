import mujoco
import mujoco_viewer
import numpy as np
import random

#Defining graph class and functions
class Graph:
    def __init__(self):
        self.graph = {}

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, from_vertex, to_vertex):
        if from_vertex in self.graph and to_vertex in self.graph:
            self.graph[from_vertex].append(to_vertex)

    def display(self):
        for vertex in self.graph:
            print(f"{vertex}: {self.graph[vertex]}")
            

#Mutable XML template
xml = """
<mujoco>
    <option gravity = '0 0 -9.81'/>
    <worldbody>
        <light diffuse = '.5 .5 .5' pos = '0 0 6' dir = '0 0 -1'/>
        <geom type = 'plane' size = '5 5 0.1' rgba = '1 1 1 1'/>
        <body name = '0' pos = '0 0 1'>
            <joint type = 'free'/>
            {}
        </body>
    </worldbody>
    {}
</mujoco>
"""

#XML Components
box = """<geom type="box" size=".1 .2 .1" rgba="1 0 0 1"/>"""
sphere = """<geom type="sphere" size=".3" rgba="0 1 0 1" mass='0.1'/>"""
joint = """<joint type='{}' name='{}'  axis = '-1 0 0' range = '0 35'/>"""
body = """
        <body name = '{}' pos='{}' euler='{}'> 
            {}
        </body>"""
motor = "<motor name ='{}' joint='{}'/>"
actuatorXML = """
    <actuator>
        {}
    </actuator>"""

#Possible limb positions and rotations - currently hardcoded
posn = [(0.5,0,0,0,0,-90), (-0.5,0,0,0,0,90), (0,0.5,0,0,0,0), (0,-0.5,0,0,0,180)]

#Initialize graph and add 0 index value corresponding to body
entityGraph = Graph()
entityGraph.add_vertex(0)

#Select a random number of limbs to generate on body
limbs = random.randrange(2,5)

#Select random number of segments per limb
segs = random.randrange(1,5)

#XML string for limbs
limbsXML = ""

#XML string for joint motors
motorXML = ""

#Track index in posn array
posnIndex = 0

#Add edges from body to limbs and build limbs XML
for i in range(limbs):
    entityGraph.add_vertex(i+1)
    entityGraph.add_edge(0,i+1)
    segXML = ''
    #Create hierarchy of limb segments
    for j in range(segs):
        entityGraph.add_edge(i+1, i+1)
        if j == 0:
            segXML = body
        if j != segs-1:
            segXML = segXML.format(f"limb{i}-seg-{j}", "0 .2 -.1", f"{(-90/segs)} 0 0", box+body)
        else:
            segXML = segXML.format(f"limb{i}-seg-{j}", "0 .2 -.1", f"{(-90/segs)} 0 0", box)
    limbsXML += body.format(f"limb-{i+1}", f"{posn[posnIndex][0]} {posn[posnIndex][1]} {posn[posnIndex][2]}",
                        f"{posn[posnIndex][3]} {posn[posnIndex][4]} {posn[posnIndex][5]}", 
                        joint.format('hinge', i+1)+box+segXML)
    motorXML += motor.format(f"motor-limb-{i+1}", i+1)
    posnIndex += 1

#Format xml string
jumpingXML = xml.format(sphere+limbsXML, actuatorXML.format(motorXML))

#Display corresponding genotype graph
#Note: Edge to self represents 1 additional segmentation
entityGraph.display()

#Make model and data
model = mujoco.MjModel.from_xml_string(jumpingXML)
data = mujoco.MjData(model)

#Make viewer
viewer = mujoco_viewer.MujocoViewer(model, data)

#Get array of actuators
actuators = model.nu

#Create list of velocities for corresponding limb motors
velocities = np.array([500*segs for i in range(limbs)])

for i in range(10000):
    if viewer.is_alive:
        if np.all(np.abs(data.qvel) < 0.01) and i%50:
            data.ctrl[:actuators] = velocities
            velocities*=-1
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break


viewer.close()