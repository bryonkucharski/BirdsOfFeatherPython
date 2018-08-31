
import sys
sys.path.insert(0, r'C:\Users\kucharskib\Documents\GitRepos\EmbeddedIntelligence')

from machine_learning_classifiers import machine_learning_classifiers
import utils as ml_utils
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA

path_to_numpy = r'C:\Users\kucharskib\Documents\GitRepos\BirdsOfFeatherPython\data\numpy'


x_train, y_train = ml_utils.load_dataset(
    path_to_numpy + r'\boaf-data-1-2500-xtrain-poly1+.npy',
    path_to_numpy + r'\boaf-data-1-2500-ytrain-poly1+.npy')

x_valid, y_valid =  ml_utils.load_dataset(
    path_to_numpy + r'\boaf-data-1-2500-xtest-poly1+.npy',
    path_to_numpy + r'\boaf-data-1-2500-ytest-poly1+.npy')
'''
ml_utils.CustomDeepModel(   x_train, y_train, x_valid, y_valid,
                            input_size = (119,),
                            num_layers = 4,
                            num_hidden_units = [512,256,30,20],
                            num_outputs = 1,
                            output_activation = 'sigmoid',
                            hidden_activation = 'relu', 
                            loss = 'binary_crossentropy',
                            optimizer = 'adam',
                            learning_rate = .005,
                            epochs = 10, 
                            batch_size = 100,
                            load_model=False,
                            save_model = False,
                            model_name = 'test.h5'
                            )
'''

ml_utils.RunCustomDeepModel(   x_train, y_train, x_valid, y_valid,
                            input_size = (83,),
                            num_layers = 3,
                            num_hidden_units = [500,250,50],
                            num_outputs = 1,
                            output_activation = 'sigmoid',
                            hidden_activation = 'relu', 
                            loss = 'binary_crossentropy',
                            optimizer = 'adam',
                            learning_rate =.005,
                            epochs = 100, 
                            batch_size = 32,
                            load_model=False,
                            save_model = False,
                            model_name = 'test.h5'
                            )


def create_model(learning_rate = .01):

    return ml_utils.custom_Deep_Model(  input_size = (7,),
                                        num_layers = 3,
                                        num_hidden_units = [500,250,100],
                                        num_outputs = 1,
                                        output_activation = 'sigmoid',
                                        hidden_activation = 'relu', 
                                        loss = 'binary_crossentropy',
                                        optimizer = 'adam',
                                        learning_rate = learning_rate)

def run():
    model = KerasClassifier(build_fn=create_model,epochs = 100, batch_size = 100,verbose=0)

    learning_rate = [.0001,.001,.01,.1,1]
    param_grid = dict(learning_rate=learning_rate)
    grid = GridSearchCV(model, param_grid, cv=2, verbose=10)
    grid_result = grid.fit(x_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
def run2():
    #set params as a dictionary
    from sklearn.model_selection import GridSearchCV
    
    #set params as a dictionary 
    param_grid = {'alpha': [0.0001, 0.01, 1, 3],
              'hidden_layer_sizes': [(100,), (100,100,100) ,(100,100,100,100,100,100), (100,100,100,100,100,100,100,100,100,100,100,100) ]}
    grid = GridSearchCV(MLPClassifier(), param_grid, cv=2, verbose=10)
    grid.fit(x_train, y_train);

#if __name__ == '__main__':
    #run()

'''
pca = PCA(n_components=2)
pca.fit_transform(x_train)

# Dump components relations with features:
print(pd.DataFrame(pca.components_,columns = ["noofPairs/num_cards_left", "num_moves_norm", "ratio_moves_per_card_norm", "validMoves_norm", "noofPairs_norm", "suitMost_norm", "rankMost_norm"],index = ['PC-1','PC-2']))
'''