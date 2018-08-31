import os, sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/ml")

from feature_adder import FeatureAdder
import utils as ml_utils
from BirdsOfAFeatherNode import BirdsOfAFeatherNode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Card import STR_CARD_DICT
from MCTS import MonteCarloTreeSearch
from MCTSNode import MonteCarloTreeSearchNode

import pdb

closed = set()
node_count = 0
goal_node = None
seed = 0

def convert_nodestr_to_num_cards(node_str):
    nc = 0
    index = 1
    for r in range(0,4):
        for c in range(0,4):
            card_str = node_str[index-1:index+1]
            if(card_str != '--'):
                nc += 1
            index += 2 
            
    return nc
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

def heuristic_search_network(model, node):
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
        prob = calc_solveability_network(model, child)
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
        if heuristic_search_network(model,child):
            return True

    closed.add(node_str)
    return False

def heuristic_clumped(model_1615,model_1413,model_1211,model_109,node):
    ''' 
    if number_cards == 16 or 15:
        use model_1615
    elif number_cards == 14 or 13:
        use model_1413
    elif . . .
    else
        use dfs
    '''
    return None

def heuristic_search_16_dfs(model_16_only,node):
    '''
    if number of cards == 16:
        use model_16_only
    else
        use dfs
    '''

    global node_count, closed
    node_count += 1

    #pdb.set_trace()
    #print(node_count)

    if node.is_goal():
        return True

    node_str = repr(node)
    num_cards = convert_nodestr_to_num_cards(node_str)

    if node_str in closed:
        #print("closed")
        return False

    if num_cards < 16:
        print("num_cards: " + str(num_cards) +" now running dfs")
        for child in node.expand():
            if depth_first_search_no_repeats(child):
                return True
    else:
        probabilities = []
        children = []
        print("num_cards: " + str(num_cards) +" now running LR")
        for child in node.expand():
            children.append(child)
            #find prob of this child
            feat_adder = FeatureAdder()
            features = feat_adder.calc_features(node.__repr__().replace(" ", "").replace("\n",""))
            features = feat_adder.feature_engineer(features)
            input = feat_adder.normalizeNewInput(features)
            prob = model_16_only.predict(input)

            #append to all probabilities
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
            if heuristic_search_16_dfs(model_16_only,child):
                return True

    closed.add(node_str)
    return False
def heuristic_search_8_dfs(model_8_plus,node):
    '''
    if number of cards >= 8:
        use model_8_plus
    else
        use dfs
    '''
    global node_count, closed, seed
    node_count += 1

    #pdb.set_trace()
    #print(node_count)
    node_str = repr(node)
    #print(node_str)
    num_cards = convert_nodestr_to_num_cards(node.__repr__().replace(" ", "").replace("\n",""))

    if node.is_goal():
        return True
   
    if node_str in closed:
        #print("closed")
        return False

    if num_cards < 8:
        #print("num_cards: " + str(num_cards) +" now running dfs")
        for child in node.expand():
            if depth_first_search_no_repeats(child):
                return True
    else:
        probabilities = []
        children = []
        #print("num_cards: " + str(num_cards) +" now running LR")
        for child in node.expand():
            children.append(child)
            #find prob of this child
            feat_adder = FeatureAdder()
            features = feat_adder.calc_features(node.__repr__().replace(" ", "").replace("\n",""))
            features = feat_adder.feature_engineer(features)
            input = feat_adder.normalizeNewInput(features)
            prob = model_8_plus.predict(input)

            #append to all probabilities
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
            if heuristic_search_8_dfs(model_8_plus,child):
                return True

    closed.add(node_str)
    return False
def heuristic_search_8_MCTS(model_8_plus,node):
    '''
    if number_of_cards >= 8:
        use model
    else
        MCTS
    '''
    return None
def calc_solveability_network(model, node,poly_features=True):
    feat_adder = FeatureAdder()
    features = feat_adder.calc_features(node.__repr__().replace(" ", "").replace("\n",""))

    if(poly_features):
        features = feat_adder.feature_engineer(features)

    input = feat_adder.normalizeNewInput(features)
    prob = model.predict(np.array([input]))
    
    return float(prob)

def calc_solveability_pca(node, poly_features=False):
    feat_adder = FeatureAdder()
    features = feat_adder.calc_features(node.__repr__().replace(" ", "").replace("\n",""))
    #print(features)
    if(poly_features):
        features = features.feature_engineer(features)
    input = feat_adder.normalizeNewInput(features)
    prob = pca_equation(input)
    
    return float(prob)

def pca_equation(input):
    '''
    OLDDD
    0.4006num_cards + 0.3897num_moves + 0.346ratio_moves_per_card + 0.2491NumClubs + 0.2379NumHearts + 0.2383NumSpades + 0.2163NumDiamonds + 0.1294NumAce
     + 0.1449NumTwo + 0.1228NumThree + 0.1111NumFour + 0.1249NumFive + 0.1152NumSix + 0.1154NumSeven + 0.1285NumEight + 0.0995NumNine + 0.0934NumTen + 0.1218NumJack +
      0.1028Num Queen + 0.1566Num King - 0.0029Most Repeated Rank - 0.0095Most Repeated Suit
    '''

    return ((0.4006*input[0]) + (0.3897*input[1]) + (0.346*input[2]) + (0.2491*input[3]) + (0.2379*input[4])+ (0.2383*input[5]) + (0.2163*input[6]) + (0.1294*input[7])
            + (0.1449*input[8]) + (0.1228*input[9]) + (0.1111*input[10]) + (0.1249*input[11]) + (0.1152*input[12]) + (0.1154*input[13]) + (0.1285*input[14]) + (0.0995*input[15]) 
            + (0.0934*input[16]) + (0.1218*input[17]) + (0.1028*input[18]) + (0.1566*input[19]) + (-0.0029*input[20]) + (-0.0095*input[21]))

def MCTS(input_node, num_sim):
    global closed, node_count, goal_node
    node_count += 1
    '''
    if node.is_mcts_goal():
        goal_node = node
        return True

    
    node_str = repr(node)
    if node_str in closed:
        return False
    print(node)
    mcts = MonteCarloTreeSearch(node)
    best_index = mcts.best_action(num_sim)
    child = root.children[best_index]
    
    if MCTS(child,num_sim):
        return True

    closed.add(node_str)
    return False'''


    node = input_node
    while True:
        print(repr(node) + "\n")
        if node.is_mcts_goal():
            goal_node = node
            print("Goal Found")
            return True

        if node.is_terminal_node():
            print("Not Found")
            return False


        mcts = MonteCarloTreeSearch(node)
        node = mcts.best_action(num_sim)

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
    '''
    DFS provided by Professor Neller
    '''
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []
    total_nodes = 0
    average_nodes = 0

    fh = open('Experiment2DFSResults.txt', 'a') 

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
            fh.write(str(seed) + "," + str(node_count) + "\n" )
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

    fh.close()

def experiment3(train = False):
    '''
    NN Heuristic using all cards model
    '''
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []
    path_to_numpy = r'..\data\numpy'

    if train:
    
        x_train, y_train = ml_utils.load_dataset(
                            path_to_numpy + r'\boaf-data-0-100_normalized_x_train.npy',
                            path_to_numpy + r'\boaf-data-0-100_normalized_y_train.npy')

        x_valid, y_valid =  ml_utils.load_dataset(                   
                            path_to_numpy + r'\boaf-data-0-100_normalized_x_valid.npy',
                            path_to_numpy + r'\boaf-data-0-100_normalized_y_valid.npy')

        scores, model = ml_utils.CustomDeepModel(
                            x_train,y_train,x_valid,y_valid,
                            input_size = (22,),
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
        model = ml_utils.load_trained_model("../models/experiment3_model.h5")
    fh = open('Experiment3NNResults.txt', 'a') 
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
            solvable = heuristic_search_network(model, node)
        if solvable:
            print('solved in ' + str(node_count) + ' nodes. ')
            fh.write(str(seed) + "," + str(node_count) + "\n" )
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
    fh.close()
def experiment4():
    '''
    PCA using all cards pca equation !!OLD
    '''
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []
    fh = open('Experiment4PCAResults.txt', 'a') 

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
            fh.write(str(seed) + "," + str(node_count) + "\n" ) 
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

    fh.close()
def experiment5(num_sims):
    '''
    MCTS for range of nodes 
    '''
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []
    fh = open('Experiment5MCTSResults_' + str(num_sims) + '.txt', 'a') 

    for seed in range(start_seed, start_seed + num_seeds):
        print('Seed {}: '.format(seed), end='')
        node = BirdsOfAFeatherNode.create_initial(seed)
        root = MonteCarloTreeSearchNode(state = node, parent = None)
        #print(node)
        if node.has_odd_bird():
            solvable = False
            odd_birds.append(seed)
        elif node.has_separated_flock():
            solvable = False
            separated_flocks.append(seed)
        else:
            solvable = MCTS(root,num_sims)
        if solvable:
            print('solved')
            fh.write("solve: " + str(seed) + "\n" ) 
            num_solved += 1
        else:
            unsolvable.append(seed)
            fh.write("unsolved: " + str(seed) + "\n" )
            print('unsolvable seed {}.'.format(seed))
            print(node)
    #print('Average Number of Nodes Across ' + str(num_solved) + 'nodes: ' + str(node_count/num_solved))
    print('Seeds {}-{}: {} solved, {} not solvable'.format(start_seed, start_seed + num_seeds - 1, num_solved,
                                                           num_seeds - num_solved))
    print('Unsolvable odd birds: ', odd_birds)
    print('    Separated flocks: ', separated_flocks)
    print('          Unsolvable: ', unsolvable)

    fh.close()
def experiment6():
    '''
    AZAD: Heuristic with clumpe models
    '''
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []
    fh = open('Experiment6Results.txt', 'a') 
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
            #TODO: ADD CLUMPED HEURISTIC
            solvable = clumped_heuristic()
        if solvable:
            print('solved')
            fh.write("solve: " + str(seed) + "\n" ) 
            num_solved += 1
        else:
            unsolvable.append(seed)
            fh.write("unsolved: " + str(seed) + "\n" )
            print('unsolvable seed {}.'.format(seed))
            print(node)
    #print('Average Number of Nodes Across ' + str(num_solved) + 'nodes: ' + str(node_count/num_solved))
    print('Seeds {}-{}: {} solved, {} not solvable'.format(start_seed, start_seed + num_seeds - 1, num_solved,
                                                           num_seeds - num_solved))
    print('Unsolvable odd birds: ', odd_birds)
    print('    Separated flocks: ', separated_flocks)
    print('          Unsolvable: ', unsolvable)

    fh.close()
def experiment7(train=True):
    '''
    Logistic Regression with 16 cards, DFS for the rest
    '''
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []
    path_to_numpy = r'..\data\numpy'

    if train:
    
        x_train, y_train = ml_utils.load_dataset(
                            path_to_numpy + r'\16+\boaf-data-1-2500-xtrain-poly16+.npy',
                            path_to_numpy + r'\16+\boaf-data-1-2500-ytrain-poly16+.npy',)
        
        x_valid, y_valid =  ml_utils.load_dataset(                   
                            path_to_numpy + r'\16+\boaf-data-1-2500-xtest-poly16+.npy',
                            path_to_numpy + r'\16+\boaf-data-1-2500-ytest-poly16+.npy')
        
        model = LogisticRegression().fit(x_train, y_train)
        yfit = model.predict(x_train)
        ypred = model.predict(x_valid)
        print("Train accuracy is %.2f " % (accuracy_score(y_train, yfit)*100))
        print("Test accuracy is %.2f " % (accuracy_score(y_valid, ypred)*100))

    open('Experiment7_16_DFS_Results.txt', 'w').close() #resets the text file
    fh = open('Experiment7_16_DFS_Results.txt', 'a') 
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
            solvable = heuristic_search_16_dfs(model, node)
        if solvable:
            print('solved in ' + str(node_count) + ' nodes. ')
            fh.write(str(seed) + "," + str(node_count) + "\n" )
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
    fh.close()
def experiment8(train=True):
    '''
    Logistic Regression with 16 cards, DFS for the rest
    '''
    start_seed = int(input('Start seed? '))
    num_seeds = int(input('How many seeds? '))
    num_solved = 0
    unsolvable = []
    odd_birds = []
    separated_flocks = []
    path_to_numpy = r'..\data\numpy'

    if train:
    
        x_train, y_train = ml_utils.load_dataset(
                            path_to_numpy + r'\8+\boaf-data-1-2500-xtrain-poly8+.npy',
                            path_to_numpy + r'\8+\boaf-data-1-2500-ytrain-poly8+.npy',)
        
        x_valid, y_valid =  ml_utils.load_dataset(                   
                            path_to_numpy + r'\8+\boaf-data-1-2500-xtest-poly8+.npy',
                            path_to_numpy + r'\8+\boaf-data-1-2500-ytest-poly8+.npy')
        
        model = LogisticRegression().fit(x_train, y_train)
        yfit = model.predict(x_train)
        ypred = model.predict(x_valid)
        print("Train accuracy is %.2f " % (accuracy_score(y_train, yfit)*100))
        print("Test accuracy is %.2f " % (accuracy_score(y_valid, ypred)*100))

    open('Experiment8_8_DFS_Results.txt', 'w').close() #resets the text file
    fh = open('Experiment8_8_DFS_Results.txt', 'a') 
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
            solvable = heuristic_search_8_dfs(model, node)
        if solvable:
            print('solved in ' + str(node_count) + ' nodes. ')
            fh.write(str(seed) + "," + str(node_count) + "\n" )
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
    fh.close()

def plotNode(filename):
    data = open(filename,'r').read()
    lines = data.split("\n")
    xs = []
    ys = []

    for line in lines:
        if len(line) > 1:
            #epoch,total_reward = line.split(',')
            x,y = line.split(',')
            x = int(x)
            y = int(y)
           
            xs.append(x)
            ys.append(y)
    plt.title(filename)
    plt.xlabel("Seed")
    plt.ylabel("Total Number of Nodes")
    plt.plot(xs,ys, color = 'r')
    plt.show()

def plotNodes(filenames):
    i = 0
    patches = []
    for file in filenames:
        print(file)
        
        data = open(file,'r').read()
        lines = data.split("\n")
        xs = []
        ys = []
        colors = ['r','g','b']
   
        for line in lines:
            if len(line) > 1:
                #epoch,total_reward = line.split(',')
                x,y = line.split(',')
                x = int(x)
                y = int(y)
            
                xs.append(x)
                ys.append(y)

        plt.legend()
        patch = mpatches.Patch(color=colors[i], label=file)
        patches.append(patch)
        plt.xlabel("Seed")
        plt.ylabel("Total Number of Nodes")
        plt.plot(xs,ys, color = colors[i])
        i+= 1
    plt.legend(handles=patches)
    plt.show()

if __name__ == '__main__':
    # test_random_solve()
    #experiment1()  # TWN: ran on my laptop in 1m27.726s, whereas original distributed Java version ran in 31.553s
    #experiment2() #36222
    #experiment3(False) #98051
    #experiment4() #173384
    #experiment5(10)
    experiment8(True)
    #print(convert_nodestr_to_num_cards("----------------------6H--------"))
    #plotNodes('Experiment2DFSResults.txt')
    #plotNodes('Experiment3NNResults.txt')
    #plotNodes(['Experiment4PCAResults.txt']) 
    #plotNodes(['Experiment2DFSResults.txt','Experiment3NNResults.txt','Experiment4PCAResults.txt']  )
    plotNodes(['Experiment7_16_DFS_Results.txt','Experiment8_8_DFS_Results.txt'])
    
    












