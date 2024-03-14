import numpy as np
import mujoco
import mujoco_viewer
import math


xmlTemplate = """
<mujoco>
    <option gravity = '0 0 -9.81'/>
    <worldbody>
        <light diffuse = '.5 .5 .5' pos = '0 0 6' dir = '0 0 -1'/>
        <geom type = 'plane' size = '10 10 0.1' rgba = '1 1 1 1'/>
        <body name = 'body' pos = '0 0 1'>
            <joint type = 'free'/>
            {} 
        </body>
    </worldbody>
    <actuator>
        {}
    </actuator>
</mujoco>
"""

motorTemplate = """<motor name = '{}' joint = '{}'/>"""

bodyTemplate = """
        <body name = '{}' pos='{}' euler='{}'> 
            {}
        </body>"""

geomTemplate = """<geom type='{}' size='{}' rgba='{}'/>"""

jointTemplate = """<joint type='{}' name='{}' axis='{}' range='{}'/>"""

bodyTypes = ["sphere", "box"]
geomTypes = ["box", "capsule", "cylinder"]

posn = [(0.5,0,0,0,0, 270), (-0.5,0,0,0,0,90), (0,0.5,0,0,0,360), (0,-0.5,0,0,0,180)]

def circle_coordinates(degrees):
    angle_radians = math.radians(degrees)
    # Compute x and y coordinates
    x = 0.5 * math.cos(angle_radians)
    y = 0.5 * math.sin(angle_radians)
    return x, y

def generate_posns(max_value:int):
    divisor = (max_value+1)//2
    rot_increment = 360/(divisor*2)
    posns = []
    for j in range(divisor*2):
        degree = 360-(rot_increment*j)
        x, y = circle_coordinates(degree)
        posns.append((x,y,degree-90))
    return posns

# Minimize variety to ensure mutation function works
def generateGeom (type: str):
    if type == "sphere":
        #return geomTemplate.format(type, f"{0.7}", "0 128 0 1")
        return """<geom type="sphere" size=".3" rgba="0 1 0 1" mass='0.1'/>"""
    else:
        return """<geom type="box" size=".1 .2 .1" rgba="1 0 0 1"/>"""


# Make a simulation class in order to generate, store, and load based on unique seed
class Simulation:
    # Set initial values based on seed
    def __init__(self, seed=None) -> None:
        self.seed = seed
        self.generation = 0
        self.population = []
    
    # Create the first "parent" models, this is only done upon the creation of a simulation
    def initialize_population(self):
        if self.seed is not None:
            # Set unique seed for population
            np.random.seed(self.seed)
            max_value = np.random.choice([2, 4]) # Maximum number of limbs/segments that can be generated
            for _ in range(5): # Initialize 5 parents so no parents are removed after first evolution
                motorCount = 0
                limbCount = 0
                body = generateGeom("sphere")
                rootNode = Node("body", "sphere", f"0 0 0", euler=None, size=None, limbsPosn=[])
                limbs = '' # Body + limbs should be final insertion
                motor = '' # Track all the motors for joints
                limbPosns = generate_posns(max_value)
                for limb in range(max_value): # Generate random number of limbs
                    limbCount += 1
                    limbShape = np.random.choice(geomTypes) # Choose random limb shape
                    limbPosn = limbPosns[limb]
                    rootNode.limbsPosn.append(limbPosn)
                    limbs += bodyTemplate # Need to fill name, pos, euler, and values inside body '' + body = one limb. body to body = two limbs
                    limbNode = Node("limb", limbShape, f"{limbPosn[0]} {limbPosn[1]} 0", euler=f"0 0 {limbPosn[2]}", size=None) #Make limb node
                    # limbs ALWAYS jointed
                    limbBody = jointTemplate.format('hinge', f"{limb}_joint", "-1 0 0", "0 35") # set limb body to be joint.
                    limbBody += generateGeom(limbShape) # add shape to limb string
                    segs = np.random.randint(max_value)
                    segments = bodyTemplate if segs else None # track the next level of models branching from limb
                    for segmentation in range(segs): # Generate random number of segments per limb
                        segShape = np.random.choice(geomTypes) # choose random segment shape
                        segPosn =(0,.2,-.1)
                        segNode = Node(f"segment", segShape, f'{segPosn[0]} {segPosn[1]} {segPosn[2]}', euler=f"{(-90/segs)} 0 0", size=None) # Generate segment node
                        jointBool = np.random.choice([True, False])
                        joint = ''
                        if jointBool:
                            joint = jointTemplate.format('hinge', f"{limb}_{segmentation}_joint", "-1 0 0", "0 35")
                            motor += motorTemplate.format(f"{limb}_{segmentation}_motor", f"{limb}_{segmentation}_joint")
                            motorCount += 1
                        if segmentation != segs-1:
                            segments = segments.format(f"{limb}_{segmentation}", "0 .2 -.1", f"{(-90/segs)} 0 0", generateGeom(segShape)+joint+bodyTemplate) # File format with current segment information and body for next segment
                        else:
                            segments = segments.format(f"{limb}_{segmentation}",  "0 .2 -.1", f"{(-90/segs)} 0 0", generateGeom(segShape)+joint)
                        limbNode.add_child(segNode)
                    limbBody += segments if segments else ''
                    limbs = limbs.format(f"{limb}", f"{limbPosn[0]} {limbPosn[1]} 0",f"0 0 {limbPosn[2]}", limbBody)
                    motor += motorTemplate.format(f"{limb}_motor", f"{limb}_joint")
                    motorCount += 1
                    rootNode.add_child(limbNode, True)
                body += bodyTemplate.format("model", "0 0 0", "0 0 0", limbs)
                rootNode.phenotype = xmlTemplate.format(body, motor) #Need to add motors
                print(rootNode.phenotype)
                rootNode.motors = motorCount
                rootNode.limbs = limbCount
                self.population.append(rootNode) 
        pass
    
    def run_generation(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.generation += 1
        print("Generation:"+ str(self.generation) + '\n')
        for parent in self.population:
            if parent.limbs != 1: # Prevents from exploiting single limb and staying in local maximum
                print(parent.limbs)
                parent.evaluate_fitness()
            else:
                parent.fitness = 0
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True) # Sort population by highest fitness
        n = len(self.population)
        indices = [i for i, x in enumerate(self.population) if x.fitness == self.population[0].fitness or x.fitness == self.population[1].fitness]
        if len(indices) > 1:  # Check if there are duplicates for first and second best to ensure evolutions do no create the same copies over and over
            for i in reversed(range(1, len(indices))):
                del self.population[indices[i]]
        while n > 5: # Only the 5 most fit entities are kept
            self.population.pop() 
            n -= 1
        if n < 10: # Fill remaining population with children
            for i in range(n,10):
                self.population.append(self.population[i-n].copy_and_mutate(mutation_chance=0.5))

# Make a class for node type considering tree-based approach towards genotype creation
class Node:
    def __init__(self, classification=None, shape=None, position=None, euler=None, size=None, phenotype=None, fitness=0, motors=0, limbs=0, limbsPosn = None) -> None:
        self.classification = classification 
        self.shape = shape 
        self.position = position 
        self.euler = euler 
        self.size = size 
        self.edges = []
        self.children = []
        self.phenotype = phenotype
        self.fitness = fitness
        self.motors = motors
        self.limbs = limbs
        self.limbsPosn = limbsPosn
    
    def add_child(self, child,  jointed=True, jointPosition=None, jointType=None):
        self.children.append(child) # Add child to node
        edge_info = {'child': child, 'jointed':jointed} # Include important information for connections between nodes
        if jointed:
            edge_info['jointPosition'] = jointPosition # If joint is shared between objets include position
            edge_info['jointType'] = jointType # Include type of joint
        self.edges.append(edge_info) # Add the edge information to list of edges
        # child[i] gives ith child and edges[i] gives information of edge with child
    
    def display(self, depth=0):
        indent = ' ' * (depth * 4)
        print(f"{indent}{self.classification} (Shape: {self.shape}, Position: {self.position}, Euler: {self.euler}, Size: {self.size})")
        for child in self.children:
            child.display(depth + 1)

    def is_jointed(self, child):
        for edge in self.edges:
            if edge['child'] is child and edge['jointed']:
                return True
        return False
    
    # NEED TO REVIEW AND ADJUST MUTATIONS
    # SHOULD BE LAST CHANGE REQUIRED BEFORE JUST RUNNING SIMULATION
    # create a copy of tree and make any potential mutations
    def copy_and_mutate(self, mutation_chance=0.1):
        new_node = Node(self.classification, self.shape, self.position, self.euler, self.size, limbs=self.limbs)
        if self.classification == "body":
            new_node.limbsPosn = self.limbsPosn[:]
        if np.random.rand() < mutation_chance:
            # Perform mutation
            mutation_type = np.random.choice(["add", "subtract"]) #Change mutation not currently used
            if mutation_type == "add": # Add a new child node
                if new_node.classification == "body":
                    new_node.limbs += 1
                    posns = generate_posns(new_node.limbs + 1)
                    positions = [x for x in posns if x not in new_node.limbsPosn]
                    i = np.random.randint(0, len(positions))
                    new_node.limbsPosn.append(positions[i])
                    new_node.add_child(Node("limb", "box", f'{positions[i][0]} {positions[i][1]} 0', euler=f"0 0 {positions[i][2]}", size=None))
                if new_node.classification == "limb":
                    new_node.add_child(Node(f"segment", "box", "0 .2 -.1", euler=f"180 0 0", size=None))
            elif mutation_type == "subtract" and new_node.children: # Remove child node if possible
                if new_node.classification == "limb" and new_node.children: # Only remove from child from limb if there exists a segment
                    new_node.children.pop()
                if new_node.classification == "body" and new_node.limbsPosn:
                    new_node.limbs -= 1
                    new_node.limbsPosn.pop()
        for child in self.children:
            new_node.add_child(child.copy_and_mutate())  # recursively copy children
        # Generate random chance to mutate
        build_from_tree(new_node)
        return new_node
    
    def evaluate_fitness(self):
        fit = 0
        velocities = np.array([100 for i in range(self.motors)])
        model = mujoco.MjModel.from_xml_string(self.phenotype)
        data = mujoco.MjData(model)
        actuators = model.nu
        #Make viewer
        viewer = mujoco_viewer.MujocoViewer(model, data)
        startHeight = 0
        maxHeight = 0
        for i in range(1000):
            if viewer.is_alive:
                if i > 1 and i%500 == 0:
                    startHeight = data.qpos[2]
                    data.ctrl[:actuators] = velocities
                if i > 500:
                    maxHeight = max(maxHeight, data.qpos[2])
                mujoco.mj_step(model, data)
                viewer.render()
            else:
                break
        viewer.close()
        self.fitness = (maxHeight - startHeight)/startHeight
        print(self.fitness)

# 3-level tree, body --> limb --> segments
def build_from_tree (root: Node):
    body = generateGeom(root.shape)
    limbs = ''
    motors = ''
    motorCount = 0
    for i, limb in enumerate(root.children):
        limbs += bodyTemplate
        limbBody = jointTemplate.format('hinge',f'{i}_joint', "-1 0 0", "0 35")
        motors += motorTemplate.format(f'{i}_motor', f"{i}_joint")
        motorCount += 1
        limbBody += generateGeom(limb.shape)
        if limb.children: #if a limb has segments build segment xml string
            segments = bodyTemplate if limb.children else ''
            n = len(limb.children)
            for j, segment in enumerate(limb.children):
                joint = ''
                if limb.is_jointed(segment):
                    joint = jointTemplate.format('hinge', f"{i}_{j}_joint", "-1 0 0", "0 35")
                    motors += motorTemplate.format(f'{i}_{j}_motor', f"{i}_{j}_joint")
                    motorCount += 1
                if j != n - 1: #if not the last segment
                    segments = segments.format(f'{limb}_{segment}', "0 .2 -.1", f"{(-90/n)} 0 0", generateGeom(segment.shape)+joint+bodyTemplate)
                else:
                    segments = segments.format(f'{limb}_{segment}', "0 .2 -.1", f"{(-90/n)} 0 0", generateGeom(segment.shape)+joint)
            limbBody += segments
        limbs = limbs.format(f"{limb}", limb.position, limb.euler, limbBody)
    body += bodyTemplate.format('name', f"{0} {0} {0}", f"{0} {0} {0}", limbs)
    root.phenotype = xmlTemplate.format(body, motors)
    root.motors = motorCount

#891 -> advanced start (max limbs and segment = 4)
#890 -> simple start (max limbs and segments = 2)
population = Simulation(891)
population.initialize_population()
for _ in range(15):
    population.run_generation()