import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from seml.experiment import Experiment 
# from seml import Experiment # for newer versions of seml
import seml
import os
import random
import warnings

from modules import helper
from modules import dataset_selecter as ds
from modules import modelCreator
from modules import feature_extractor as fe

from datetime import datetime

from sklearn.model_selection import StratifiedKFold

ex = Experiment()

# The following three functions are not needed since seml 0.4.2
# from sacred import Experiment

# ex = Experiment()
# seml.setup_logger(ex)

# @ex.post_run_hook
# def collect_stats(_run):
#     seml.collect_exp_stats(_run)

# @ex.config
# def config():
#     overwrite = None
#     db_collection = None
#     if db_collection is not None:
#         ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    #init before the experiment!
    @ex.capture(prefix="init")
    def baseInit(self, nrFolds: int, patience: int, seed_value: int):
        self.seed_value = seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
        tf.random.set_seed(seed_value)
        tf.keras.utils.set_random_seed(seed_value)
        np.random.RandomState(seed_value)
        np.random.seed(seed_value)
        context.set_global_seed(seed_value)
        ops.get_default_graph().seed = seed_value

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        np.random.seed(seed_value)

        #save some variables for later
        self.kf = StratifiedKFold(nrFolds, shuffle=True, random_state=seed_value)
        self.fold = 0
        self.nrFolds = nrFolds
        self.patience = patience
        self.seed_value = seed_value  
        self.method = None  

        self.data_incomplete = False
        self.has_NaN = False    
        self.no_relevant_features = False

        #init gpu
        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices: 
            tf.config.experimental.set_memory_growth(gpu_instance, True)

        # Create the directories if they don't exist
        os.makedirs("./results/", exist_ok=True)
        os.makedirs("./saves/", exist_ok=True)
        


    # Load the dataset
    @ex.capture(prefix="data")
    def init_dataset(self, dataset: str, number: int, takename: bool, method: str, saveMethod: str, limit: int):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        self.number = number
        self.method = method
        self.saveMethod = saveMethod
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_trainy, self.y_testy, self.seqSize, self.dataName, self.num_of_classes, self.number = ds.datasetSelector(dataset, self.seed_value, number, takeName=takename)
        self.limit = limit
        
        if self.X_train is None:
            print(f"{self.dataName} is incomplete.")
            self.data_incomplete = True
            return

        if np.any(np.isnan(self.X_train.flatten())) or np.any(np.isnan(self.X_test.flatten())):
            print(f"Original data for {self.dataName} contains NaN values")
            self.has_NaN = True
            return
        
        if self.seqSize > self.limit:
            return
        
        # Pre-compute features
        saved = fe.save_foldwise_features(self.number, self.dataName, self.method, self.X_train, self.X_test, self.y_train, self.y_test, self.y_trainy, self.y_testy, self.kf, self.saveMethod)

        if not saved:
            print("No relevant Features extracted...skipping this method")
            self.no_relevant_features = True
            fe.delete_features(self.number, self.dataName, self.method)
            return
        
    # all inits
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.baseInit()
        self.init_dataset()

    #methods to save the results into the fullResults dict
    def fillOutDicWithNNOutFull(self, abstractionString, configString, fullResults, inputDict):
        if abstractionString not in fullResults.keys():
            fullResults[abstractionString] = dict()
        if configString not in fullResults[abstractionString].keys():
            fullResults[abstractionString][configString] = []
        fullResults[abstractionString][configString].append(inputDict)

    def fillOutDicWithNNOut(self, abstractionString, configString, fullResults, outData):
        inputDict = helper.fillOutDicWithNNOutSmall(outData)
        self.fillOutDicWithNNOutFull(abstractionString, configString, fullResults, inputDict)

    def printTime(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

    # one experiment run with a certain config set.
    @ex.capture(prefix="model")
    def trainExperiment(self, numEpochs: int, batchSize: int, skipDebugSaves: bool, saveWeights: bool,
        architecture='transformer', symbolCount=0,
        numOfAttentionLayers=None, header=None, dmodel=None, dff=None, 
        num_resblocks=None, num_channels=None, use_1x1conv=None,
        lr=None, warmup=None, 
        dropOutRate=None, strategy=None, ncoef=None):

        print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
        print('Dataname:')
        print(self.dataName)
        print('Method:')
        print(self.method)
        self.printTime()
        warnings.filterwarnings('ignore')   

        if self.data_incomplete:
            return "dataset " + self.dataName + " is incomplete"

        if self.has_NaN:
            return "dataset " + self.dataName + " contains NaN values"
        
        if self.no_relevant_features:
            return "dataset " + self.dataName + " no relevant features"
        
        fullResults = dict()

        if ncoef:
            ncoefI = ncoef[0]
            coef_div = ncoef[1]
        else:
            ncoefI = None
            coef_div = None

        #limit the lenght of the data
        if self.seqSize > self.limit:
            fullResults["Error"] = "dataset " + self.dataName + " too big: " + str(self.seqSize)
            print('TOO LONG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " too big: " + str(self.seqSize))
            fe.delete_features(self.number, self.dataName, self.method)
            return "dataset " + self.dataName + " too big: " + str(self.seqSize) #fullResults


        skip_folds = False
        # k fold train loop
        for train, test in self.kf.split(self.X_train, self.y_trainy):
            if skip_folds:
                continue

            self.fold+=1   

            abstractionString = self.method
            
            x_train1, x_val, x_test, y_train1, y_val, y_test, y_trainy1, y_valy, y_testy = fe.load_foldwise_features(self.number, self.dataName, self.method, self.fold, self.saveMethod)
            
            if np.any(np.isnan(x_train1.flatten())) or np.any(np.isnan(x_val.flatten())) or np.any(np.isnan(x_test.flatten())) or np.any(np.isnan(y_train1.flatten())) or np.any(np.isnan(y_val.flatten())) or np.any(np.isnan(y_test.flatten())):
                self.has_NaN = True
                return "Processed dataset " + self.dataName + " contains NaN values"

            if x_train1 is None:
                print("No relevant features extracted, skipping..")
                skip_folds = True
                continue 
            
            x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy1, y_testy = modelCreator.preprocessData(x_train1, x_val, x_test, y_train1, y_val, y_test, 
                                                                                                                                                y_trainy1, y_testy, self.fold, symbolsCount=symbolCount, dataName=self.dataName, 
                                                                                                                                                seqSize=self.seqSize,
                                                                                                                                                strategy=strategy,
                                                                                                                                                method=self.method,
                                                                                                                                                ncoef=ncoefI, coef_div=coef_div)
    
            out = modelCreator.doAbstractedTraining(x_train1, x_val, x_test, y_train1, y_val, y_testy, 
                                                    BATCH=batchSize, seed_value=self.seed_value, num_epochs=numEpochs, patience = self.patience, 
                                                    num_of_classes=self.num_of_classes, number=self.number, dataName=self.dataName, fold=self.fold,
                                                    architecture=architecture, symbolCount=symbolCount, 
                                                    numOfAttentionLayers=numOfAttentionLayers,header=header, dmodel=dmodel, dff=dff, 
                                                    num_resblocks=num_resblocks, num_channels=num_channels, use_1x1conv=use_1x1conv,
                                                    abstractionType=abstractionString, rate=dropOutRate, lr=lr, warmup=warmup, 
                                                    skipDebugSaves=skipDebugSaves, saveWeights=saveWeights)
            self.fillOutDicWithNNOut(abstractionString, "results", fullResults, out)
            
            saveName = modelCreator.getWeightName(self.number, self.dataName, 0,
                                                  learning = False, results = True,
                                                  architecture=architecture, abstractionType=self.method, symbols=symbolCount,
                                                  layers=numOfAttentionLayers, header=header, dmodel=dmodel, dff=dff,
                                                  num_resblocks=num_resblocks, num_channels=num_channels, use_1x1conv=use_1x1conv,
                                                  num_coef=ncoefI, dropout=dropOutRate, strategy=strategy)

            print("finished fold: " + str(self.fold))

        helper.save_obj(fullResults, str(saveName), self.saveMethod)

        return saveName
    

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.trainExperiment()