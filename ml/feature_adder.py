
import os
import sys
import numpy as np
import re
import random
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/game")

from Card import STR_CARD_DICT
from BirdsOfAFeatherNode import BirdsOfAFeatherNode

class FeatureAdder():
    
    def generate_numpy_data(self):
        path = "../data/csv/boaf-data-0-100-with-normalized-features.csv"
        save_name = "../data/numpy/boaf-data-0-100_normalized_"

        #unwanted_rows = [0, 7, 9]
        unwanted_rows = [0, 23, 24, 25,26]

        with open(path, "r") as ins:
            data = []
            labels = []
            i = 0
            for line in ins:
                if(i != 0 and i != 306529 and i != 615380 and i != 923985 and i != 1229102): # first line
                    #get rid of new line, split by delim
                    list = line.strip().replace('\n','')
                    list = re.split(',',list)
                    list = np.delete(list, unwanted_rows).tolist()
                    #convert to float
                    list = [float(x) for x in list]
                    
                    data.append(list[:-1])
                    labels.append(list[-1])
                print("Current Row: " + str(i))  
                i += 1


        
        x = np.array(data)
        y = labels

        x_valid = []
        y_valid = []
        num_validations = x.shape[0] * .2

        #gets unique random numbers the size of num_validations
        indexes = random.sample(range(0,x.shape[0]-1), round(num_validations))


        for index in indexes:
            x_valid.append(x[index])
            y_valid.append(y[index])

            
        new_x_train = np.delete(x,indexes,0)
        new_y_train = np.delete(y,indexes,0)
        x_valid = np.array(x_valid, dtype='float32')


        print('-Data set -\nX Shape', x.shape, '\nY Length: ' , len(y))
        print('-Validation Data set -\nX Shape', x_valid.shape, '\nY Length: ' , len(y_valid))
        
        np.save(save_name + "x_train",x)
        np.save(save_name + "y_train",y)
        np.save(save_name + "x_valid",x_valid)
        np.save(save_name + "y_valid",y_valid)
    
    def get_min_max_values(self):
        path = "../data/csv/boaf-data-0-100-with-raw-features.csv"
        save_name = "../data/numpy/boaf-min-max-data"

        #unwanted_rows = [0, 7, 9]
        unwanted_rows = [0, 23, 24, 25,26]

        with open(path, "r") as ins:
            data = []
            i = 0
            for line in ins:
                if(i != 0 and i != 306529 and i != 615380 and i != 923985 and i != 1229102): #headers
                    #get rid of new line, split by delim
                    list = line.strip().replace('\n','')
                    list = re.split(',',list)
                    list = np.delete(list, unwanted_rows).tolist()
                    #convert to float
                    list = [float(x) for x in list]
                    
                    data.append(list[:-1])
                print("Current Row: " + str(i))  
                i += 1


        
        x = np.array(data)
        min = np.min(x, axis = 0)
        max = np.max(x, axis = 0)
       
        #print(min,max,range)

        np.save("../data/numpy/boaf-min-data", min)
        np.save("../data/numpy/boaf-max-data", max)
        

    def calc_features(self,node_str):
        

        index = 1
        numCards = 0
        numSuit = [0,0,0,0]
        numRank  = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        numCardsCol = [0,0,0,0]
        numCardsRow = [0,0,0,0]
        mostRepeatedRank = 0
        mostRepeatedSuit = 0
        totalMoves = 0


        node = BirdsOfAFeatherNode()

        node.grid = [[None, None, None, None],
                     [None, None, None, None], 
                     [None, None, None, None],
                     [None, None, None, None]]
                
        for r in range(0,4):
            for c in range(0,4):
                card_str = node_str[index-1:index+1]
                if(card_str == '--'):
                    card1 = None
                else:
                    card1 = STR_CARD_DICT[card_str]
                    node.grid[r][c] = card1
                    
                index += 2            
                
                if card1 is not None:
                    numCards += 1
                    numRank[card1.get_rank()] += 1
                    numSuit[card1.get_suit()] += 1
                    numCardsCol[c] += 1
                    numCardsRow[r] += 1
        
        #print(node.grid)
                '''
                index2 = 1
                for r2 in range(0,4):
                    for c2 in range(0,4):
                        card_str2 = node_str[index2-1:index2+1]
                        if(card_str2 == '--'):
                            card2 = None
                        else:
                            card2 = STR_CARD_DICT[card_str2]
                        index2 += 2

                        if card1 is not None and card2 is not None and card2 != card1 and (card1.get_suit() == card2.get_suit() or abs(card1.get_rank() - card2.get_rank()) <= 1):
                            if (r == r2) or (c == c2):
                                totalMoves += 1
                                
                            #print(card1.__repr__(),card2.__repr__())
                '''

        mostRepeatedSuit = np.argmax(numSuit)
        mostRepeatedRank = np.argmax(numRank)

        #[numRank[i] for i in range(0,len(numRank))] 
        results_array = np.array([numCards, len(node.expand()), len(node.expand()) / numCards ])
        results_array = np.append(results_array, [numSuit[i] for i in range(0,len(numSuit))])
        results_array = np.append(results_array, [numRank[i] for i in range(0,len(numRank))] )
        results_array = np.append(results_array,[mostRepeatedRank,mostRepeatedSuit])
        
        return results_array
   
       
    def normalizeInput(self,input):
        min = np.load("../data/numpy/boaf-min-data.npy")
        max = np.load("../data/numpy/boaf-max-data.npy")

        normalized_input = [None]*23
        
        diff = max - min
        normalized_input = (input - min) / diff
                
        return np.array(normalized_input)
        
#features = FeatureAdder()
#features.generate_numpy_data()
#features.get_min_max_values()
#feats = features.calc_features("--------3D9H------TD------2S----")
#print(feats)
#feats_norm = features.normalizeInput(feats)
#print(feats_norm)
