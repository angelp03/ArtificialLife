import numpy as np
import json
import os


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

motorTemplate = """<motor name = {} joint ={}/> 
                    {}"""

bodyTemplate = """
        <body name = '{}' pos='{}' euler='{}'> 
            {}
        </body>
        {}"""

geomTemplate = """<geom type='{}' size='{}'/>"""

jointTemplate = """<joint type='{}' name='{}' axis='{}' range='{}'/>"""

bodyTypes = ["sphere", "box"]
geomTypes = ["box", "capsule", "cylinder"]

def generateGeom (type: str):
    if type == "sphere":
        return geomTemplate.format(type, f"{1}")
    elif type == "cylinder" or type == "capsule":
        return geomTemplate.format(type, f"{1} {1}")
    elif type == "box":
        return geomTemplate.format(type, f"{1} {1} {1}")


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
            for model in range(np.random.randint(4,7)): # Generate random number of starting parents
                body = generateGeom(np.random.choice(bodyTypes))
                limbs = ''
                for limb in range(np.random.randint(2,5)): # Generate random number of limb
                    limbs += bodyTemplate #Add body for limb
                    limbs.format("name", f"{0} {0} {0}",generateGeom(np.random.choice(geomTypes)), "")
                body += bodyTemplate.format("name", f"{0} {0} {0}", f"{0} {0} {90}", limbs, "")



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
    def __init__(self, classification, shape=None, position=None, euler=None, size=None) -> None:
        self.classification = classification # What is the type of object added to the model
        self.shape = shape # What is the shape of this object
        self.position = position # What is the position of this item with respect to parent
        self.euler = euler # What is the rotation of the object
        self.size = size # What is the size of the object
        self.edges = [] # What other objects stem from this object
    
    def add_child(self, child,position=None, euler=None, jointed=True, jointPosition=None, jointType=None):
        self.children.append(child) # Add child to node
        edge_info = {'child': child, 'jointed':jointed, 'position': position, 'euler': euler} # Include important information for connections between nodes
        if jointed:
            edge_info['jointPosition'] = jointPosition # If joint is shared between objets include position
            edge_info['jointType'] = jointType # Include type of joint
        self.edges.append(edge_info) # Add the edge information to list of edges
        # child[i] gives ith child and edges[i] gives information of edge with child

# To build model, take in tree, perform DFS and build limbs fully 1 by 1
# After all limbs built, add to body and run simulation