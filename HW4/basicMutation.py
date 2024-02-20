import numpy as np
import json
import os
import mujoco
import mujoco_viewer


xmlTemplate = """
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
    <actuator>
        {}
    </actuator>
</mujoco>
"""

motorTemplate = """<motor name = {} joint ={}/>"""

bodyTemplate = """
        <body name = '{}' pos='{}' euler='{}'> 
            {}
        </body>"""

geomTemplate = """<geom type='{}' size='{}'/>"""

jointTemplate = """<joint type='{}' name='{}' axis='{}' range='{}'/>"""

bodyTypes = ["sphere", "box"]
geomTypes = ["box", "capsule", "cylinder"]

posn = [(0.5,0,0,0,0,-90), (-0.5,0,0,0,0,90), (0,0.5,0,0,0,0), (0,-0.5,0,0,0,180)]

def generateGeom (type: str):
    if type == "sphere":
        return geomTemplate.format(type, f"{0.5}")
    elif type == "cylinder" or type == "capsule":
        return geomTemplate.format(type, f"{0.3} {0.3}")
    elif type == "box":
        return geomTemplate.format(type, f"{0.3} {0.3} {0.3}")
    else:
        return ''

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
                # Choose random body type
                bodyShape = np.random.choice(bodyTypes)
                body = generateGeom(bodyShape)
                rootNode = Node("body", bodyShape, f"1 1 1", euler=None, size=None)
                limbs = '' # Body + limbs should be final insertion
                motor = '' # Track all the motors for joints
                for limb in range(np.random.randint(2,5)): # Generate random number of limbs
                    limbShape = np.random.choice(geomTypes) # Choose random limb shape
                    limbs += bodyTemplate # Need to fill name, pos, euler, and values inside body '' + body = one limb. body to body = two limbs
                    limbNode = Node("limb", limbShape, f"0 0 0", euler=None, size=None) #Make limb node
                    # limbs ALWAYS jointed
                    limbBody = jointTemplate.format('hinge', f"limb{limb}", "-1 0 0", "0 35") # set limb body to be joint.
                    limbBody += generateGeom(limbShape) # add shape to limb string
                    segments = bodyTemplate # track the next level of models branching from limb
                    segs = np.random.randint(2,5)
                    for segmentation in range(segs): # Generate random number of segments per limb
                        segShape = np.random.choice(geomTypes) # choose random segment shape
                        segNode = Node("segment", segShape, f'0 0 0', euler=None, size=None) # Generate segment node
                        jointBool = np.random.choice([True, False])
                        joint = ''
                        if jointBool:
                            joint = jointTemplate.format('hinge', f"limb{limb}_{segmentation}", "-1 0 0", "0 35")
                        if segmentation != segs-1:
                            segments = segments.format(f"{limb}_{segmentation}", "0 0 0", "0 0 0", generateGeom(segShape)+joint+bodyTemplate) # File format with current segment information and body for next segment
                        else:
                            segments = segments.format(f"{limb}_{segmentation}", "0 0 0", "0 0 0", generateGeom(segShape)+joint)
                        limbNode.add_child(segNode)
                    limbBody += segments
                    limbs = limbs.format(f"limb:{limb}", "0 0 0", "0 0 0", limbBody, '', '', '', '')
                    motor += motorTemplate.format("name", "joint_name")
                    rootNode.add_child(limbNode, True)
                body += bodyTemplate.format("name", f"{0} {0} {0}", f"{0} {0} {90}", limbs, "")
                rootNode.phenotype = xmlTemplate.format(body, '') #Need to add motors
                self.population.append(rootNode) 
        pass
    # This can be done within the simulation class or can create a seperate function that takes simulation class as input
    def run_generation(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.generation += 1
        pass
    # Save state into simulations directory || Change so it takes in a seed and loads file based on that
    def save_state(self, directory='simulations'):
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
        self.seed = state['seed']

# Make a class for node type considering tree based approach towards genotype creation
class Node:
    def __init__(self, classification=None, shape=None, position=None, euler=None, size=None, phenotype=None) -> None:
        self.classification = classification # What is the type of object added to the model
        self.shape = shape # What is the shape of this object
        self.position = position # What is the position of this item with respect to parent
        self.euler = euler # What is the rotation of the object
        self.size = size # What is the size of the object
        self.edges = [] # What other objects stem from this object
        self.children = []
        self.phenotype = phenotype
    
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
        if np.random.rand() < mutation_chance and self.classification != "limb": ## GET RID OF SECOND CHECK ONCE LIMB CREATION SUPPORTS MUTATION
            # Perform mutation
            mutation_type = np.random.choice(["add", "subtract", "change"])
            if mutation_type == "add": # Add a new child node
                print("Node added \n")
                new_node.add_child(Node("segment", "box", f'0 0 0', euler=None, size=None))
            elif mutation_type == "subtract": # Remove child node if possible
                if new_node.children:
                    new_node.children.pop()
            # Add for final project to allow shape, position, size changes
            # elif mutation_type == "change":
                # Change some attributes of the node
                # new_node.classification = "changed_classification"
                # new_node.shape = "changed_shape"
                # Similarly, you can change other attributes as needed

        return new_node

# 3-level tree, body --> limb --> segments
def build_from_tree (root: Node):
    body = generateGeom(root.shape)
    limbs = ''
    for limb in root.children:
        limbs += bodyTemplate
        limbBody = jointTemplate.format('hinge',f'limb{limb}', '-1 0 0', "0 35")
        limbBody += generateGeom(limb.shape)
        segments = bodyTemplate
        n = len(limb.children)
        for segment in limb.children:
            joint = ''
            if limb.is_jointed(segment):
                joint = jointTemplate.format('hinge', f"limb{limb}_{segment}", "-1 0 0", "0 35")
            if limb.children.index(segment) != n - 1: #if not the last segment
                segments = segments.format(f'{limb}_{segment}', "0 0 0", "0 0 0", generateGeom(segment.shape)+joint+bodyTemplate)
            else:
                segments = segments.format(f'{limb}_{segment}', "0 0 0", "0 0 0", generateGeom(segment.shape)+joint)
        limbBody += segments
        limbs = limbs.format(f"limb:{limb}", "0 0 0", "0 0 0", limbBody, '', '', '', '')
    body += bodyTemplate.format('name', f"{0} {0} {0}", f"{0} {0} {90}", limbs,'',)
    root.phenotype = xmlTemplate.format(body, '')


# To build model, take in tree, perform DFS and build limbs fully 1 by 1
# After all limbs built, add to body and run simulation
population = Simulation(891)
population.initialize_population()
models = population.population
mutation = models[0].copy_and_mutate(mutation_chance=1)
build_from_tree(mutation)
modelArrays = [models[0], mutation]
for parent in modelArrays:
    if parent == mutation:
        print(parent.phenotype)
    model = mujoco.MjModel.from_xml_string(parent.phenotype)
    data = mujoco.MjData(model)
    #Make viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    for i in range(10000):
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()
        else:
            break
    viewer.close()