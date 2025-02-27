from tslearn.datasets import UCR_UEA_datasets
from sklearn.utils import shuffle
import numpy as np
from random import randint
from time import sleep
from scipy.io import arff
from pyts.datasets import ucr_dataset_list


def datasetSelector(dataset, seed_Value, number, takeName = True, use_cache=True):
    """ Select one dataset """
    if dataset == 'ucr':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, number = doUCR(seed_Value, number, takeName = takeName, use_cache=use_cache)
    elif dataset == 'saved':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, number = load_saved_dataset(seed_Value, number)
    else:
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, number = []

    if X_train is None:
        return None, None, None, None, None, None, None, dataName, None, None

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test = np.array(y_test)
    y_test = y_test.astype(float)
    X_test = np.array(X_test)
    X_test = X_test.astype(float)
    X_train = np.array(X_train)
    X_train = X_train.astype(float)   
    y_testy = np.array(y_testy)
    y_trainy = np.array(y_trainy)

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, number




def doUCR(seed_value, number, takeName = True, retry=0, use_cache=True):
    """ Load UCR datasets and information """
    try:
        datasets = UCR_UEA_datasets(use_cache=use_cache)
        dataset_list = ucr_dataset_list()
       
        if takeName:
            
            datasetName = number
            number = dataset_list.index(datasetName)
        else:
            datasetName = dataset_list[number]
        
        X_train, y_trainy, X_test, y_testy = datasets.load_dataset(datasetName)

        if (y_trainy is None) or (y_testy is None):
            return None, None, None, None, None, None, None, datasetName, None, None 
        
        setY = list(set(y_testy))
        setY.sort()
        

        num_of_classes = len(set(y_testy))
        seqSize = len(X_train[0])

        X_train, y_trainy = shuffle(X_train, y_trainy, random_state = seed_value)
    
        y_train = []
        
        for y in y_trainy:
            y_train_puffer = np.zeros(num_of_classes)
            y_train_puffer[setY.index(y)] = 1
            y_train.append(y_train_puffer)

        y_trainy = np.argmax(y_train,axis=1) +1 
            
        y_test = []
        for y in y_testy:
            y_puffer = np.zeros(num_of_classes)
            y_puffer[setY.index(y)] = 1
            y_test.append(y_puffer)
            
        y_testy = np.argmax(y_test,axis=1) +1 
    
    except Exception as e:
        print(e)
        if retry < 5:
            sleep(randint(10,30))

            if retry == 4:
                return doUCR(seed_value, number, takeName = takeName, retry=retry+1, use_cache=False)
            else:
                return doUCR(seed_value, number, takeName = takeName, retry=retry+1) 

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, datasetName, num_of_classes, number

def split_dataframe(df, chunk_size = 10000): 
    """ split dataframe into chunks """
    
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def load_saved_dataset(seed_value=56, number='SyntheticControl'):   
    """ Load a dataset saved in datasets/ """
    datasetName = number

    data_path_train = './datasets/'+datasetName+'/'+datasetName+'_TRAIN.arff'
    data_path_test = './datasets/'+datasetName+'/'+datasetName+'_TEST.arff'

    #Load and formate data
    data_train, meta_train = arff.loadarff(data_path_train)
    data_test, meta_test = arff.loadarff(data_path_test)

    data_train = np.array(data_train.tolist())
    data_test = np.array(data_test.tolist())

    y_trainy = data_train[:,-1].astype(int)
    y_train = []
    X_train = data_train[:,:-1]
    y_testy_full = data_test[:,-1].astype(int)
    y_testy = y_testy_full
    y_test = []
    X_test = data_test[:,:-1]

    num_of_classes = len(np.unique(y_trainy))
    seqSize = len(X_train[0])

    X_train, y_trainy = shuffle(X_train, y_trainy, random_state = seed_value)

    for y in y_trainy:
        y_train_puffer = np.zeros(num_of_classes)
        y_train_puffer[y-1] = 1
        y_train.append(y_train_puffer)

    for y in y_testy:
        y_puffer = np.zeros(num_of_classes)
        y_puffer[y-1] = 1
        y_test.append(y_puffer)

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test_full = np.array(y_test)
    y_test_full = y_test_full.astype(float)
    y_test = y_test_full  
    y_test = y_test.astype(float)

    X_test = X_test.astype(float)
    X_train = X_train.astype(float)

    print("Loaded Saved Dataset: ", number)
    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, datasetName, num_of_classes, number