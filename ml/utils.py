from keras.models import load_model
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Lambda
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

import numpy as np

def load_dataset(x_name, y_name):
    """
    loads a .npy file x_name and y_name

    Args:
        x_name: .npy file name for x
        y_name: .npy file name for y
       
    Returns:
        x,y matricies of data
    """
    
    print('getting vector data from npy files. . .')
  
    x = np.load(x_name)
    y = np.load(y_name)

    print('-Data set -\nX Shape', x.shape, '\nY Length: ' , len(y))

    return x, y

def CustomDeepModel(x_train,y_train,x_valid,y_valid,input_size,num_layers, num_hidden_units,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate,epochs, batch_size, save_model = False,load_model = False, model_name = ''):
        '''
        Assumes data is already preprocessed with mean = 0 and std = 1
        '''
   
        x_std = x_train
        x_valid_std = x_valid
        #tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        model = custom_Deep_Model(input_size, num_layers, num_hidden_units,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate)
        if load_model:
            model.load_weights(model_name)


       # y_train = utils.one_hot(y_train, num_outputs) 
       # y_valid = utils.one_hot(y_valid, num_outputs)
        #y_train = to_categorical(y_train, num_outputs)
        #y_valid = to_categorical(y_valid, num_outputs)
        model.fit(x_std, y_train,validation_data = (x_valid_std, y_valid), epochs=epochs, batch_size=batch_size) #,callbacks=[tbCallBack])
        scores = model.evaluate(x_valid, y_valid)

        if save_model:
            model.save_weights(model_name)


        print("Baseline Error: %.2f%%" % (100-scores[1]*100))
        return scores, model

def custom_Deep_Model(input_size, num_layers, num_hidden_units,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate):
    
    """
    Custom Deep Neural Network
    Architecture will be 
    input layer size input_shape  -> 1 to num_layers hidden layers size num_hidden_units (all same size) -> output size num_outputs

    Args:
        inputs to the model
        
    Returns:
        keras model 
    """
    
    model = Sequential()
   
    if optimizer == 'adam':
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)
        #can add other Keras optimizers here
    
    for i in range(num_layers):
        if (i == 0): #input layer
            model.add(Dense(num_hidden_units[i],input_shape = input_size, activation = hidden_activation))
        else:
            model.add(Dense(num_hidden_units[i],activation = hidden_activation ))

    model.add(Dense(num_outputs, activation = output_activation))

    model.compile(loss=loss,optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return model

def load_trained_model(path):
        return load_model(path)
        