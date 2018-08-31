
import os
import sys
import numpy as np
import re
import random
import pdb
import pandas as pd
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/game")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from Card import STR_CARD_DICT
from BirdsOfAFeatherNode import BirdsOfAFeatherNode

class FeatureAdder():
    
    def generate_numpy_data(self):
        '''
        used to take a csv file and data generated from java and convert it into a numpy array
        '''
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
    
    def get_min_max_values(self, data_path = "../data/csv/boaf-data-0-100-with-raw-features.csv", save_name = "boaf-data-0-100",unwanted_rows = [0, 23, 24, 25,26] ):
        '''
        gets the min and max values of every row and saves it as a numpy file to use later for normalization
        '''

        #unwanted_rows = [0, 7, 9]
        #unwanted_rows = [0, 23, 24, 25,26]

        with open(data_path, "r") as ins:
            data = []
            i = 0
            for line in ins:
                list = []
                if(i != 0 and i != 306529 and i != 615380 and i != 923985 and i != 1229102): #headers from the large files
                    #get rid of new line, split by delim
                    list = line.strip().replace('\n','')
                    list = re.split(',',list)

                    numcards = float(list[16])

                    list = np.delete(list, unwanted_rows).tolist()
                    
                    #convert to float
                    list = [float(x) for x in list]

                    #add extra features
                    pair_per_card = float(list[1]/numcards)#list.append(num_pairs/num_cards)
                    moves_per_card = float(list[0]/numcards)#list.append(validMoves/num_cards)
                    list.insert(0,pair_per_card)
                    list.insert(1,moves_per_card)
                    data.append(list)
                print("Current Row: " + str(i), str(list))  
                i += 1


        poly = PolynomialFeatures(degree=3, include_bias=False)
        x = np.array(data)
        x = poly.fit_transform(x)
        
        
        print(x)
        min = np.min(x, axis = 0)
        max = np.max(x, axis = 0)
        var = np.var(x, axis = 0)
        std = np.std(x, axis = 0)
       
        print(min)
        print(max)


        np.save(("../data/numpy/min-max/" + save_name+ "-min-data"), min)
        np.save(("../data/numpy/min-max/" + save_name+ "-max-data"), max)
        np.save(("../data/numpy/min-max/" + save_name+ "-std-data"), std)
        np.save(("../data/numpy/min-max/" + save_name+ "-var-data"), var)
    def calc_features_old(self,node_str):
        '''
        old data features 
        '''
        
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
    def calc_features(self,node_str):
        '''

        **take out valid moves number of pairs to see if bettter accuracy
        **15 only cards, 14 only cards, 8 only cards, etc to get different predicition accuracies

        Given a node string, calcualtes seven NON-NORMALIZED features
        Use normalizeNewInput to normalize features
        
        0. noofPairs/num_cards
        1. num_moves/num_cards
        2. validMoves - how many cards are same suit/adj rank in the same row/col
        3. noofPairs - how many cards are the samesuit/adj ranks, dont need to be in the same row/col (will create future moves), this is bidirectional 
        4. suitMost - how many of which suit appears the most in the given state
        5. rankMost - how many of which rank appears the most in the given state
        '''
        
        noofPairs = 0
        num_cards = 0
        moves_per_card = 0
        valid_moves = 0
        suitMost = 0
        rankMost = 0

        #create new node
        node = BirdsOfAFeatherNode()

        #create empty grid of new node
        node.grid = [[None, None, None, None],
                     [None, None, None, None], 
                     [None, None, None, None],
                     [None, None, None, None]]

        #holds the freq of each card
        numSuit = [0,0,0,0]
        numRank  = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        index = 1

        #iterate through node and create cards at each position 
        for r in range(0,4):
            for c in range(0,4):
                card_str = node_str[index-1:index+1]
                if(card_str == '--'):
                    card1 = None
                else:
                    card1 = STR_CARD_DICT[card_str]
                    node.grid[r][c] = card1
                   

                if card1 is not None:
                    num_cards += 1
                    
                    #increase the frequency for that card/suit
                    numRank[card1.get_rank()] += 1
                    numSuit[card1.get_suit()] += 1

                
                
                    index2 = 1

                    #iterate through each card again to get total number of possible moves
                    for r2 in range(0,4):
                        for c2 in range(0,4):
                            card_str2 = node_str[index2-1:index2+1]
                            if(card_str2 == '--'):
                                card2 = None
                            else:
                                card2 = STR_CARD_DICT[card_str2]
                            index2 += 2


                            if  (card1 is not None and card2 is not None #not null cards
                                and card2 != card1 #not the same card
                                and (card1.get_suit() == card2.get_suit()
                                    or abs(card1.get_rank() - card2.get_rank()) <= 1)): #same suit or same/adj ranks
                                noofPairs += 1

                index += 2 

        validMoves = len(node.get_legal_moves()) #number of poss
        moves_per_card = validMoves/num_cards
        pairs_per_card = noofPairs/num_cards

        suitMost = np.max(numSuit)
        rankMost = np.max(numRank)

        results_array = np.array([pairs_per_card,moves_per_card, validMoves, noofPairs, suitMost, rankMost ])
        return results_array
    def feature_engineer(self,input):
        '''
        Given a input, creates additional polynominal features
        which_rows is an array of which rows to add poly features 
        Ex: 
        input =  [0,1,2,3,4]
        which_row = [2]
        output = [0,1,2,3,4,input[2]^2, input[2]^3, etc . . . ]
        '''


        poly = PolynomialFeatures(degree=3, include_bias=False)
        X2 = poly.fit_transform(input.reshape(1,-1))
        #print(X2)

        #normalize
        #std = np.load(std_path)
        #var = np.load(var_path)

        #num_featu,mres = len(input)
        #normalized_input = [None]*num_features

        #output = (X2 - var) / std

       #print(output[0])
       # print("Feature Engineering\nInput shape: " + str(input.reshape(1,-1).shape)+ "\nOutput shape: " + str(output.shape))

        return X2





    def normalizeNewInput(self,input,min_path="../data/numpy/min-max/boaf-data-1-2500-min-data.npy",max_path="../data/numpy/min-max/boaf-data-1-2500-max-data.npy"):
        '''
        Called with a new input to normalize, expect min and max of each feature to be stored in a numpy array
        '''

        #input array of unnormalized [pairs_per_card,moves_per_card, validMoves,noofPairs,suitMost,rankMost]
        #max and min in format of    [pairs_per_card,moves_per_card,validMoves, noofPairs,suitMost,rankMost]
        min = np.load(min_path)
        max = np.load(max_path)

        num_features = len(input)

        normalized_input = [None]*num_features
        
        diff = max - min
        normalized_input = (input - min) / diff
                
        return np.array(normalized_input)
        
features = FeatureAdder()

#features.generate_numpy_data()
# For New features, we want to keep 
# Excel (feat) column
# k (validMoves) 10
# l (noofPairs) 11
# m (suitMost) 12
# o (rankMost) 14
#so we dont want [0,1,2,3,4,5,6,7,8,9,13,16,15,17,18]
#features.get_min_max_values(data_path="../data/csv/boaf-data-1-2500-with-features.csv", save_name="boaf-data-1-2500", unwanted_rows=[0,1,2,3,4,5,6,7,8,9,13,16,15,17,18])

#feats = features.calc_features("JD2D9HJC5D7H7C5HKDKC9S5SADQCKH3H")
#print(pd.DataFrame(feats.reshape(1,-1),columns = ["noofPairs/num_cards_left", "ratio_moves_per_card", "validMoves", "noofPairs", "suitMost", "rankMost"]))

#new_feats = features.feature_engineer(feats)
#print(new_feats)

#feats_norm = features.normalizeNewInput(new_feats,min_path="../data/numpy/min-max/boaf-data-1-2500-min-data.npy",max_path="../data/numpy/min-max/boaf-data-1-2500-max-data.npy")
#print(pd.DataFrame(feats_norm.reshape(1,-1)))

