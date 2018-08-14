
import sys
sys.path.insert(0, r'C:\Users\kucharskib\Documents\GitRepos\EmbeddedIntelligence')

from machine_learning_classifiers import machine_learning_classifiers
import machine_learning_utils as utils
import numpy as np

classifiers = machine_learning_classifiers()
path_to_numpy = r'C:\Users\kucharskib\Documents\GitRepos\BirdsOfFeatherPython\data\numpy'


utils.load_dataset(
    path_to_numpy + r'\boaf-data-0-100_normalized_x_train.npy',
    path_to_numpy + r'\boaf-data-0-100_normalized_y_train.npy',
    path_to_numpy + r'\boaf-data-0-100_normalized_x_valid.npy',
    path_to_numpy + r'\boaf-data-0-100_normalized_y_valid.npy')

'''
classifiers.load_dataset(
    path_to_numpy + r'\bof-1-10000_normalized_x_train.npy',
    path_to_numpy + r'\bof-1-10000_normalized_y_train.npy',
    path_to_numpy + r'\bof-1-10000_normalized_x_valid.npy',
    path_to_numpy + r'\bof-1-10000_normalized_y_valid.npy')
'''

utils.CustomDeepModel(input_size = (22,),
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
                            load_model=True,
                            save_model = True,
                            model_name = 'test.h5'
                            )




'''

classifiers.XGBoost()
#classifiers.numpy_logistic_reg(num_iterations = 1000, learning_rate = .08, confusion_title='BoF 1-100', confusion_labels=['solved', 'unsolved'])
'''