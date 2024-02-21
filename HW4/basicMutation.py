import numpy as np
import mujoco
import mujoco_viewer

# Positions and rotations hard coded to ensure observable difference in movement, structure
# Need to add euler and size randomization
# Potentially add crossing over = setting a limb to be equal to another limb from a model in the population
# Need to figure out the fitness funciton
# Need to randomize joint axis and ranges <-- Maybe
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

posn = [(0.5,0,0,0,0,-90), (-0.5,0,0,0,0,90), (0,0.5,0,0,0,0), (0,-0.5,0,0,0,180)]

# Minimize variety to ensure mutation function works
def generateGeom (type: str):
    if type == "sphere":
        #return geomTemplate.format(type, f"{0.7}", "0 128 0 1")
        return """<geom type="sphere" size=".3" rgba="0 1 0 1" mass='0.1'/>"""
    elif type == "cylinder" or type == "capsule":
        #return geomTemplate.format(type, f"{0.3} {0.3}", "128 0 0 1")
        return  """<geom type="box" size=".1 .2 .1" rgba="1 0 0 1"/>"""
    elif type == "box":
        # return geomTemplate.format(type, f"{0.3} {0.3} {0.3}", "0 0 128 1")
        return  """<geom type="box" size=".1 .2 .1" rgba="1 0 0 1"/>"""
    else:
        return ''
# Not currently being used
""" def generatePosn():
    x = np.random.rand() * np.random.choice([1,-1])
    y = np.random.rand() * np.random.choice([1,-1])
    z = np.random.rand()
    return (x, y, z) """

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
            for model in range(np.random.randint(4,7)): # Generate random number of parents
                motorCount = 0
                bodyShape = np.random.choice(bodyTypes) # Select random body type
                body = generateGeom(bodyShape)
                rootNode = Node("body", bodyShape, f"1 1 1", euler=None, size=None)
                limbs = '' # Body + limbs should be final insertion
                motor = '' # Track all the motors for joints
                for limb in range(np.random.randint(2,4)): # Generate random number of limbs
                    limbShape = np.random.choice(geomTypes) # Choose random limb shape
                    limbPosn = posn[limb]
                    limbs += bodyTemplate # Need to fill name, pos, euler, and values inside body '' + body = one limb. body to body = two limbs
                    limbNode = Node("limb", limbShape, f"{limbPosn[0]} {limbPosn[1]} {limbPosn[2]}", euler=None, size=None) #Make limb node
                    # limbs ALWAYS jointed
                    limbBody = jointTemplate.format('hinge', f"{limb}_joint", "-1 0 0", "0 35") # set limb body to be joint.
                    limbBody += generateGeom(limbShape) # add shape to limb string
                    segments = bodyTemplate # track the next level of models branching from limb
                    segs = np.random.randint(2,4)
                    for segmentation in range(segs): # Generate random number of segments per limb
                        segShape = np.random.choice(geomTypes) # choose random segment shape
                        segPosn =(0,.2,-.1)
                        segNode = Node(f"segment", segShape, f'{segPosn[0]} {segPosn[1]} {segPosn[2]}', euler=None, size=None) # Generate segment node
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
                    limbBody += segments
                    limbs = limbs.format(f"{limb}", f"{limbPosn[0]} {limbPosn[1]} {limbPosn[2]}",f"{posn[limb][3]} {posn[limb][4]} {posn[limb][5]}", limbBody)
                    motor += motorTemplate.format(f"{limb}_motor", f"{limb}_joint")
                    motorCount += 1
                    rootNode.add_child(limbNode, True)
                body += bodyTemplate.format("model", "0 0 0", "0 0 0", limbs)
                rootNode.phenotype = xmlTemplate.format(body, motor) #Need to add motors
                rootNode.motors = motorCount
                self.population.append(rootNode) 
        pass
    
    def run_generation(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.generation += 1
        for parent in self.population:
            if parent.fitness == 0: # Only reevaluate if fitness hasn't been calculated
                parent.evaluate_fitness()
        self.population = sorted(self.population, key=lambda x: x.fitness) # Sort population by highest fitness
        n = len(self.population)
        while n > 5: # Only the 5 most fit entities are kept
            self.population.pop() 
            n -= 1
        if n < 10: # Fill remaining population with children
            for i in range(n,10):
                self.population.append(self.population[i-n].copy_and_mutate())
    # Save state into simulations directory || Change so it takes in a seed and loads file based on that
    """ def save_state(self, directory='simulations'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, f"simulation_{self.seed}.json")
        # Need to consider what values are required to continue a simulation
        state = {
            'seed': self.seed
        }
        # Add important information to respective file
        with open(filename, 'w') as f:
            json.dump(state, f)
    # Load state from previous simulation
    def load_state(self, filename):
        with open(filename, 'r') as f:
            state = json.load(f)
        # Set important information
        self.seed = state['seed'] """

# Make a class for node type considering tree-based approach towards genotype creation
class Node:
    def __init__(self, classification=None, shape=None, position=None, euler=None, size=None, phenotype=None, fitness=0, motors=0) -> None:
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
    
    # create a copy of tree and make any potential mutations
    def copy_and_mutate(self, mutation_chance=0.1):
        new_node = Node(self.classification, self.shape, self.position, self.euler, self.size)
        for child in self.children:
            new_node.add_child(child.copy_and_mutate())  # recursively copy children
        # Generate random chance to mutate
        if np.random.rand() < mutation_chance:
            # Perform mutation
            mutation_type = np.random.choice(["add", "subtract", "change"]) #Change mutation not currently used
            if mutation_type == "add": # Add a new child node
                new_node.add_child(Node("segment", "box", f'0 0 0', euler=None, size=None))
            elif mutation_type == "subtract": # Remove child node if possible
                if new_node.children:
                    new_node.children.pop()
            elif mutation_type == "change": # Change features of node
                change_type = np.random.choice(["shape", "size", "position"])
                if change_type == "shape":
                    new_node.shape = np.random.choice(geomTypes)
                elif change_type == "size":
                    new_node.size = "0 0 0"
                """ elif change_type == "position":
                    new_position = generatePosn()
                    new_node.position = f"{new_position[0]} {new_position[1]} {new_position[2]}" """
        build_from_tree(new_node)
        return new_node
    
    def evaluate_fitness(self):
        fit = 0
        velocities = np.array([500 for i in range(self.motors)])
        model = mujoco.MjModel.from_xml_string(self.phenotype)
        data = mujoco.MjData(model)
        actuators = model.nu
        #Make viewer
        viewer = mujoco_viewer.MujocoViewer(model, data)
        maxHeight = 0
        for i in range(500):
            if viewer.is_alive:
                if i%10:
                    data.ctrl[:actuators] = velocities
                    velocities *= -1
                if i > 100:
                    maxHeight = max(maxHeight, data.qpos[2])
                mujoco.mj_step(model, data)
                viewer.render()
            else:
                break
        viewer.close()
        self.fitness = maxHeight

# 3-level tree, body --> limb --> segments
def build_from_tree (root: Node):
    body = generateGeom(root.shape)
    limbs = ''
    motors = ''
    motorCount = 0
    m = len(root.children)
    for i, limb in enumerate(root.children):
        limbs += bodyTemplate
        limbBody = jointTemplate.format('hinge',f'{i}_joint', "0 0 -1", "0 35")
        motors += motorTemplate.format(f'{i}_motor', f"{i}_joint")
        motorCount += 1
        limbBody += generateGeom(limb.shape)
        if limb.children: #if a limb has segments build segment xml string
            segments = bodyTemplate
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
        limbs = limbs.format(f"{limb}", f"{posn[i][0]} {posn[i][1]} {posn[i][2]}", f"{posn[i][3]} {posn[i][4]} {posn[i][5]}", limbBody)
    body += bodyTemplate.format('name', f"{0} {0} {0}", f"{0} {0} {0}", limbs)
    root.phenotype = xmlTemplate.format(body, motors)
    root.motors = motorCount


population = Simulation(891)
population.initialize_population()
parents_and_mutations = [population.population[0]]
for _ in range(4):
    parents_and_mutations.append(population.population[0].copy_and_mutate(mutation_chance=1)) #Mutation chance = 1 to ensure mutations occur
for model in parents_and_mutations:
    model.evaluate_fitness()