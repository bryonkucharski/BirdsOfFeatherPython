import re
import numpy as np
import random

def generate_numpy_data():
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
