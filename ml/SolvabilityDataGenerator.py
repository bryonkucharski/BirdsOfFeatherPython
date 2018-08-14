
import random
from SolvabilitySearcher import SolvabilitySearcher
import os 
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/game")
from BirdsOfAFeatherNode import BirdsOfAFeatherNode



class SolvabilityDataGenerator():

    def __init__(self,filename,startSeed,numSeeds):
        self.filename = filename
        self.startSeed = startSeed
        self.minEndSeed = self.startSeed + numSeeds
        self.minStates = sys.maxsize
        self.minSolvableStateCount = sys.maxsize
        self.minUnsolvableStates = sys.maxsize

    '''
	 * Generates comma-separated value (CSV) data where each line consists of a 
	 * String description of a Birds of a Feather state, a comma, and an integer 
	 * 0/1 indicating that the state is unsolvable/solvable.  The initial state 
	 * of a seed is always included in the data. For each solvable initial 
	 * state, the following iterative process is used to generate data through 
	 * simulated play:
	 * <ul>
	 * <li>All children of the state are generated.</li>
	 * <li>If there are no solvable children, we increment the seed.</li>
	 * <li>If all are solvable, pick one at random as the next state, generate no
	 *     output, and repeat.</li>
	 * <li>If some are unsolvable, print a CSV line for each child, pick a
	 *     solvable child at random as the next state, and repeat.</li>
	 * </ul>
	 * This process continues until we've met or exceeded one minimum stopping 
	 * condition as expressed in our public fields (e.g. minEndSeed, minStates,
	 * minSolvableStates, minUnsolvableStates). States are counted if they are
	 * output.  We do not include children that are all solvable on the output,
	 * as we wish to focus our machine learning on state decisions where the
	 * decision matters.
	 * @throws FileNotFoundException 
	'''
    def generateCSVData(self):
        random.seed(0)
        seed = self.startSeed
        stateCount = 0
        solvableStateCount = 0
        unsolvableStateCount = 0
        out = open(self.filename + ".csv", 'w').close() #resets the file if something already exists
        out = open(self.filename + ".csv", 'a') #opens file for appending
        out.write("\"state\",\"solvable\"")
        while ((seed < self.minEndSeed) and (stateCount < self.minStates) and (solvableStateCount < self.minSolvableStateCount) and (unsolvableStateCount < self.minUnsolvableStates)):
            node = BirdsOfAFeatherNode.create_initial(seed)
            seed += 1
            searcher = SolvabilitySearcher()
            solvable = searcher.search(node)
            print("seed: " + str(seed) + ", " + self.stateToCSVString(node) + ", " + str(int(solvable)))
            if not solvable:
                continue
            #iterate through children
            solvableChildren = []
            unsolveableChildren = []
            while(True):
                print("here")
                children = node.expand()
                if len(children) == 0:
                    break
                stateCount += len(children)
                solvableChildren = []
                unsolveableChildren = []
                i = 0
                for child in children:
                    print(i)
                    solvable = searcher.search(child)
                    if solvable:
                        solvableChildren.append(child)
                    else:
                        unsolveableChildren.append(child)
                    i += 1
                randomIndex = random.randint(0,len(solvableChildren)-1)
                node = solvableChildren[randomIndex]#choose random solvable child
                if len(unsolveableChildren) < 1:
                    continue
                solvableStateCount += len(solvableChildren)
                unsolvableStateCount += len(unsolveableChildren)
                for child in solvableChildren:
                    print("seed: " + str(seed) + ", " + self.stateToCSVString(child) + ", " + "1")
                    out.write(self.stateToCSVString(child) + "," + "1\n")
                for child in unsolveableChildren:
                    print("seed: " + str(seed) + ", " + self.stateToCSVString(child) + ", " + "0")
                    out.write(self.stateToCSVString(child) + "," + "0\n")
                        
        out.close()
    '''
	 * Convert the puzzle state of the given BirdsOfAFeatherNode to a row-major ordered list of
	 * two-character codes Card toString() or "--" if a grid cell is empty.
	 * @param state given BirdsOfAFeatherNode
	 * @return a row-major ordered list of two-character codes describing the state
	'''
    def stateToCSVString(self,state):
        output = []
        for r in range(0,4):
            for c in range(0,4):
                card = state.grid[r][c]
                if card == None:
                    output.append("--")
                else:
                    output.append(card.__repr__())

        return ''.join(output)

generator = SolvabilityDataGenerator("test_file",0,1)
generator.generateCSVData()
