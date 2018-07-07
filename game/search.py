import os, sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/ml")
sys.path.insert(0, r'C:\Users\kucharskib\Documents\GitRepos\EmbeddedIntelligence')

from machine_learning_classifiers import machine_learning_classifiers
from feature_adder import FeatureAdder

from BirdsOfAFeatherNode import BirdsOfAFeatherNode
import numpy as np

import pdb

closed = set()
node_count = 0
goal_node = None

classifiers = machine_learning_classifiers()



def reset_depth_first_search_no_repeats():
    global closed, node_count, goal_node
    closed = set()
    node_count = 0
    goal_node = None


def depth_first_search_no_repeats(node):
    global closed, node_count, goal_node
    #pdb.set_trace()
    node_count += 1
    #print(node_count)
    if node.is_goal():
        goal_node = node
        return True
    node_str = repr(node)
    if node_str in closed:
        #print("closed")
        return False
    for child in node.expand():
        if depth_first_search_no_repeats(child):
            return True
    closed.add(node_str)
    return False

def heuristic_search_pca_and_network(node):
    return None

def heuristic_search_pca(node):
    global node_count, closed
    node_count += 1
    #print(node_count)

    if node.is_goal():
        return True
    
    node_str = repr(node)
    if node_str in closed:
        #print("closed")
        return False
    probabilities = []
    children = []

    for child in node.expand():
        children.append(child)
        prob = calc_solveability_pca(child)
        probabilities.append(prob)

    #pdb.set_trace()
    
    if len(children) > 0:
        #combine children and prob
        zipped = zip(children,probabilities)
        #sort based on prob from highest to lowest
        Z = sorted(zipped, key=lambda x: x[1], reverse=True)
        # unzip back to seperate lists
        unzipped = [list(t) for t in zip(*Z)]

        sorted_children = unzipped[0]
        sorted_probabilites = unzipped[1]

        #sorted_children = children
       # pdb.set_trace()
    else:
        return False

    for child in sorted_children:
        if heuristic_search_pca(child):
            return True

    closed.add(node_str)
    return False

def heuristic_search_network(node):
    global node_count, closed
    node_count += 1
    #pdb.set_trace()
    #print(node_count)

    if node.is_goal():
        return True
    
    node_str = repr(node)
    if node_str in closed:
        #print("closed")
        return False
    probabilities = []
    children = []

    for child in node.expand():
        children.append(child)
        prob = calc_solveability_network(child)
        probabilities.append(prob)

    
    
    if len(children) > 0:
    #combine children and prob
        zipped = zip(children,probabilities)
        #sort based on prob from highest to lowest
        Z = sorted(zipped, key=lambda x: x[1], reverse=True)
        # unzip back to seperate lists
        unzipped = [list(t) for t in zip(*Z)]

        sorted_children = unzipped[0]
        sorted_probabilites = unzipped[1]

        #sorted_children = children
       
    else:
        return False
    for child in sorted_children:
        if heuristic_search_network(child):
            return True

    closed.add(node_str)
    return False
        

def calc_solveability_network(node):
    feat_adder = FeatureAdder()
    features = feat_adder.calc_features(node.__repr__().replace(" ", "").replace("\n",""))
    input = feat_adder.normalizeInput(features)
    prob = classifiers.predict_model(input)
    
    return float(prob)

def calc_solveability_pca(node):
    feat_adder = FeatureAdder()
    features = feat_adder.calc_features(node.__repr__().replace(" ", "").replace("\n",""))
    #print(features)
    input = feat_adder.normalizeInput(features)
    prob = pca_equation(input)
    
    return float(prob)

def pca_equation(input):
    '''
    0.4006num_cards + 0.3897num_moves + 0.346ratio_moves_per_card + 0.2491NumClubs + 0.2379NumHearts + 0.2383NumSpades + 0.2163NumDiamonds + 0.1294NumAce
     + 0.1449NumTwo + 0.1228NumThree + 0.1111NumFour + 0.1249NumFive + 0.1152NumSix + 0.1154NumSeven + 0.1285NumEight + 0.0995NumNine + 0.0934NumTen + 0.1218NumJack +
      0.1028Num Queen + 0.1566Num King - 0.0029Most Repeated Rank - 0.0095Most Repeated Suit
    '''

    return ((0.4006*input[0]) + (0.3897*input[1]) + (0.346*input[2]) + (0.2491*input[3]) + (0.2379*input[4])+ (0.2383*input[5]) + (0.2163*input[6]) + (0.1294*input[7])
            + (0.1449*input[8]) + (0.1228*input[9]) + (0.1111*input[10]) + (0.1249*input[11]) + (0.1152*input[12]) + (0.1154*input[13]) + (0.1285*input[14]) + (0.0995*input[15]) 
            + (0.0934*input[16]) + (0.1218*input[17]) + (0.1028*input[18]) + (0.1566*input[19]) + (-0.0029*input[20]) + (-0.0095*input[21]))





def test_random_solve():
    reset_depth_first_search_no_repeats()
    root = BirdsOfAFeatherNode.create_initial()  # (367297990)
    print(root)
    if root.has_separated_flock():
        print('Separated flock - search is futile.')
    elif depth_first_search_no_repeats(root):
        # successful search
        print('Goal node found in {} nodes.'.format(node_count))
        print('Solution string:', goal_node.solution_string())
    else:
        # unsuccessful search
        print('Goal node not found in {} nodes.'.format(node_count))


def experiment1():
    start_seed = 0
    num_seeds = 100
    num_solved = 0
    unsolvable = []
    for seed in range(start_seed, start_seed + num_seeds):
        print('Seed {}: '.format(seed), end='')
        node = BirdsOfAFeatherNode.create_initial(seed)
        solvable = False if node.has_separated_flock() else depth_first_search_no_repeats(node)
        if solvable:
            print('solved.')
            num_solved += 1
        else:
            unsolvable.append(seed)
            print('unsolvable seed {}.'.format(seed))
            print(node)
    print('Seeds {}-{}: {} solved, {} not solvable'.format(start_seed, start_seed + num_seeds - 1, num_solved,
                                                           num_seeds - num_solved))
    print('          Unsolvable: ', unsolvable)


def experiment2():
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []
    total_nodes = 0
    average_nodes = 0
    for seed in range(start_seed, start_seed + num_seeds):
        print('Seed {}: '.format(seed), end='')
        node = BirdsOfAFeatherNode.create_initial(seed)
        if node.has_odd_bird():
            solvable = False
            odd_birds.append(seed)
        elif node.has_separated_flock():
            solvable = False
            separated_flocks.append(seed)
        else:
            solvable = depth_first_search_no_repeats(node)
        if solvable:
            print('solved in ' + str(node_count) + ' nodes.  ')
            num_solved += 1
        else:
            unsolvable.append(seed)
            print('unsolvable seed {}.'.format(seed))
            print(node)
    print('Average Number of Nodes Across ' + str(num_solved) + 'nodes: ' + str(node_count/num_solved))
    print('Seeds {}-{}: {} solved, {} not solvable'.format(start_seed, start_seed + num_seeds - 1, num_solved,
                                                           num_seeds - num_solved))
    print('Unsolvable odd birds: ', odd_birds)
    print('    Separated flocks: ', separated_flocks)
    print('          Unsolvable: ', unsolvable)

def experiment3(train = False):
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []
    path_to_numpy = r'..\data\numpy'

    if train:
    
        classifiers.load_dataset(
                            path_to_numpy + r'\boaf-data-0-100_normalized_x_train.npy',
                            path_to_numpy + r'\boaf-data-0-100_normalized_y_train.npy',
                            path_to_numpy + r'\boaf-data-0-100_normalized_x_valid.npy',
                            path_to_numpy + r'\boaf-data-0-100_normalized_y_valid.npy')

        classifiers.CustomDeepModel(input_size = (22,),
                            num_layers = 4,
                            num_hidden_units = [512,256,30,20],
                            num_outputs = 1,
                            output_activation = 'sigmoid',
                            hidden_activation = 'relu', 
                            loss = 'binary_crossentropy',
                            optimizer = 'adam',
                            learning_rate = .005,
                            epochs = 500, 
                            batch_size = 100,
                            save_model=True,
                            model_name="../models/experiment3_model.h5"
                            )
    else:
        classifiers.load_model("../models/experiment3_model.h5")

    for seed in range(start_seed, start_seed + num_seeds):
        print('Seed {}: '.format(seed), end='')
        node = BirdsOfAFeatherNode.create_initial(seed)
        #print(node)
        if node.has_odd_bird():
            solvable = False
            odd_birds.append(seed)
        elif node.has_separated_flock():
            solvable = False
            separated_flocks.append(seed)
        else:
            solvable = heuristic_search_network(node)
        if solvable:
            print('solved in ' + str(node_count) + ' nodes. ')
           
            num_solved += 1
        else:
            unsolvable.append(seed)
            print('unsolvable seed {}.'.format(seed))
            print(node)
    print('Average Number of Nodes Across ' + str(num_solved) + 'nodes: ' + str(node_count/num_solved))
    print('Seeds {}-{}: {} solved, {} not solvable'.format(start_seed, start_seed + num_seeds - 1, num_solved,
                                                           num_seeds - num_solved))
    print('Unsolvable odd birds: ', odd_birds)
    print('    Separated flocks: ', separated_flocks)
    print('          Unsolvable: ', unsolvable)

def experiment4():
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []

    for seed in range(start_seed, start_seed + num_seeds):
        print('Seed {}: '.format(seed), end='')
        node = BirdsOfAFeatherNode.create_initial(seed)
        #print(node)
        if node.has_odd_bird():
            solvable = False
            odd_birds.append(seed)
        elif node.has_separated_flock():
            solvable = False
            separated_flocks.append(seed)
        else:
            solvable = heuristic_search_pca(node)
        if solvable:
            print('solved in ' + str(node_count) + ' nodes. ')
           
            num_solved += 1
        else:
            unsolvable.append(seed)
            print('unsolvable seed {}.'.format(seed))
            print(node)
    print('Average Number of Nodes Across ' + str(num_solved) + 'nodes: ' + str(node_count/num_solved))
    print('Seeds {}-{}: {} solved, {} not solvable'.format(start_seed, start_seed + num_seeds - 1, num_solved,
                                                           num_seeds - num_solved))
    print('Unsolvable odd birds: ', odd_birds)
    print('    Separated flocks: ', separated_flocks)
    print('          Unsolvable: ', unsolvable)



if __name__ == '__main__':
    # test_random_solve()
   # experiment1()  # TWN: ran on my laptop in 1m27.726s, whereas original distributed Java version ran in 31.553s
    #experiment2()
    #experiment3(False)
    experiment4()












