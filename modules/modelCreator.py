import numpy as np

import tensorflow as tf

# from tensorflow.python.keras.layers import Input # TODO Changed this import
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.python.keras.layers.merge import concatenate # TODO Changed this import
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping

from modules import helper
from modules import transformer
from modules.resnet import ResNet

from sklearn import metrics

from pyts.approximation import MultipleCoefficientBinning
from pyts.approximation import SymbolicAggregateApproximation
from pyts.approximation import SymbolicFourierApproximation

import math
import os

TF_VERBOSE = 0

def createModel(splits, x_train, x_val, x_test, batchSize, seed_value, num_of_classes, numOfAttentionLayers, dmodel, header, dff, rate = 0.0, lr='custom', warmup=10000):    
        """ Create the transformer model with given information """
        x_trains = np.dsplit(x_train, splits)

        x_tests = np.dsplit(x_test, splits)
        x_vals = np.dsplit(x_val, splits)
        maxLen = len(x_trains[0][0])

        flattenArray = []
        inputShapes = []
        encClasses = []
    
        for i in range(len(x_trains)):
            x_part = np.array(x_trains[i])
        
            seq_len1 = x_part.shape[1]

            sens1 = x_part.shape[2]
            input_shape1 = (seq_len1, sens1)
            left_input1 = Input(input_shape1)

            inputShapes.append(left_input1)
           
            encoded = left_input1
            input_vocab_size = 0
                
            #create transformer encoder layer 
            encClass1 = transformer.Encoder(numOfAttentionLayers, dmodel, header, dff, 5000, rate=rate, input_vocab_size = input_vocab_size + 2, maxLen = maxLen, seed_value=seed_value)
            encClasses.append(encClass1)

            maskLayer = tf.keras.layers.Masking(mask_value=-2)
            encInput = maskLayer(encoded)
            enc1, attention, fullAttention = encClass1(encInput) # TODO training = True?
            flatten1 = Flatten()(enc1)
            flattenArray.append(flatten1)
            
        # Merge nets
        if splits == 1:
            merged = flattenArray[0]
        else:
            merged = concatenate(flattenArray)
        output = Dense(num_of_classes, activation = "sigmoid")(merged)

        # Create combined model
        wdcnnt_multi = Model(inputs=inputShapes,outputs=(output))
       
        tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed_value)

        if lr == 'custom':
            learning_rate = transformer.CustomSchedule(32, warmup)
        else:
            learning_rate = lr
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.99, 
                                     epsilon=1e-9)
        
        wdcnnt_multi.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['accuracy'], run_eagerly=False)
                    # metrics=['accuracy'], run_eagerly=True) # TODO Uncomment for newer versions
        
        # print('done')
        
        return wdcnnt_multi, inputShapes, x_trains, x_tests, x_vals

def getWeightName(number, name, fold, symbols, layers=None, abstractionType=None, header=None,  dmodel=None, dff=None, learning = True, resultsPath = 'results', results=False, usedp=False, doHeaders=True, doDetails=False, posthoc=None, dropout=0.3,
                  doShapelets=False, minShapelets=None, timeLimit=None, strategy='uniform', 
                  architecture='transformer', num_resblocks=None, num_channels=None, use_1x1conv=True):
    """ building saving name for model weights """

    if (not doShapelets) and (architecture == 'transformer'):
        if usedp:
            if results:
                baseName = "./" + resultsPath + "/results-" + str(int(number)) + '_' + name + ' -a: ' + abstractionType + ' -s: ' + str(int(symbols)) + ' -l: '+ str(int(layers)) 
            else: 
                baseName = "./saves/weights-" + str(int(number)) + '_' + name + ' -a: ' + abstractionType + ' -f: ' + str(fold) + ' -s: ' + str(int(symbols)) + ' -l: ' + str(int(layers))
        else:
            if results:
                baseName = "./" + resultsPath + "/results-" + str(int(number)) + '_' + name + ' -a ' + abstractionType + ' -s ' + str(int(symbols)) + ' -l ' + str(int(layers))
            else: 
                baseName = "./saves/weights-" + str(int(number)) + '_' + name + ' -a ' + abstractionType + ' -f: ' + str(fold) + ' -s ' + str(int(symbols)) + ' -l ' + str(int(layers))
        
        if doHeaders:
            baseName = baseName + ' -h ' + str(int(header))
        if doDetails:
            baseName = baseName + ' -dm ' + str(int(dmodel))
            baseName = baseName + ' -df ' + str(int(dff))

    if doShapelets:
        baseName = "./" + resultsPath + "/results-" + str(int(number)) + '_' + name + ' -a: ' + abstractionType + ' -s: ' + str(int(symbols))
        if minShapelets >= 1:
            baseName = baseName + ' -ms ' + str(int(minShapelets)) + ' -t ' + str(int(timeLimit))
        else:
            baseName = baseName + ' -ms ' + str(minShapelets) + ' -t ' + str(int(timeLimit))

    if architecture == 'resnet':
        baseName = "./" + resultsPath + "/results-" + str(int(number)) + '_' + name + ' -a: ' + abstractionType + ' -s: ' + str(int(symbols))
        baseName = baseName + ' -rb ' + str(int(num_resblocks)) + ' -ch ' + str(int(num_channels))
        if use_1x1conv:
            baseName = baseName + ' -cv 1'
        else:
            baseName = baseName + ' -cv 0'
        baseName = baseName + ' -m ' + architecture
        
    if posthoc:
        baseName = baseName + ' -p ' + str(int(posthoc))

    if dropout != 0.3:
        baseName = baseName + ' -r ' + str(dropout)

    if (strategy != 'uniform') and (strategy is not None):
        baseName = baseName + ' -st ' + str(strategy)[0]
        
    if learning:
        return baseName + '-learning.weights.h5'
    elif results:
        return baseName + '.h5'
    else:
        return baseName + '.weights.h5'

def doAbstractedTraining(trainD, valD, testD, y_train1, y_val, y_testy, BATCH, seed_value, num_of_classes, number, dataName, fold, symbolCount, num_epochs, numOfAttentionLayers, dmodel, 
                         header, dff, skipDebugSaves=False, patience=0, useSaves=False, saveWeights=False, abstractionType=None, rate=0.0, ncoef=0,
                         lr='custom', warmup=10000, architecture='transformer',
                         num_resblocks=None, num_channels=None, use_1x1conv=True):
    """ Do training for the given model definition """

    newTrain = trainD
    newVal = valD
    newTest = testD

    if architecture == 'transformer':    
        n_model2, inputs2, x_trains2, x_tests2, x_vals2 = createModel(1, newTrain, newVal, newTest , BATCH, seed_value, num_of_classes, numOfAttentionLayers, dmodel, header, dff, rate=rate, lr=lr, warmup=warmup)
        
        if (os.path.isfile(getWeightName(number, dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, dmodel=dmodel, dff=dff, doDetails=True, learning=False, posthoc=ncoef) + '.index') and useSaves):
            n_model2.load_weights(getWeightName(number, dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, dmodel=dmodel, dff=dff, doDetails=True, learning=False, posthoc=ncoef))
        else:
            earlystop = EarlyStopping(monitor= 'val_loss', min_delta=0 , patience=patience, verbose=0, mode='auto', restore_best_weights=True)
            n_model2.fit(x_trains2, y_train1, validation_data = (x_vals2, y_val) , epochs = num_epochs, batch_size = BATCH, verbose=TF_VERBOSE, callbacks =[earlystop], shuffle = False) # TODO shuffle?
            if saveWeights:
                n_model2.save_weights(getWeightName(number, dataName, fold, symbolCount, numOfAttentionLayers, abstractionType, header, dmodel=dmodel, dff=dff, doDetails=True, learning=False, posthoc=ncoef), overwrite=True)
            
        earlyPredictor2 = tf.keras.Model(n_model2.inputs, n_model2.layers[2].output)

        predictions2 = n_model2.predict(x_vals2)
        predictions2 = np.argmax(predictions2,axis=1)+1

        # Measure this fold's accuracy on validation set compared to actual labels
        y_compare = np.argmax(y_val, axis=1)+1
        val_score2 = metrics.accuracy_score(y_compare, predictions2)
        val_pre = metrics.precision_score(y_compare, predictions2, average='macro')
        val_rec = metrics.recall_score(y_compare, predictions2, average='macro')
        val_f1= metrics.f1_score(y_compare, predictions2, average='macro')

        print(f"validation fold score with input {abstractionType}(accuracy): {val_score2}")

        # Predictions on the test set
        limit = 500
        test_predictions_loop2 = []
        for bor in range(int(math.ceil(len(x_tests2[0])/limit))):
            test_predictions_loop2.extend(n_model2.predict([x_tests2[0][bor*limit:(bor+1)*limit]]))

        attentionQ2 = None
        if (not skipDebugSaves):
            attentionQ0 = []
            attentionQ1 = []
            attentionQ2 = []

            for bor in range(int(math.ceil(len(x_trains2[0])/limit))):
                attOut = earlyPredictor2.predict([x_trains2[0][bor*limit:(bor+1)*limit]])
                attentionQ0.extend(attOut[0]) 
                attentionQ1.extend(attOut[1])

                if len(attentionQ2) == 0:
                    attentionQ2 = attOut[2]
                else:
                    for k in range(len(attentionQ2)):
                        attentionQ2[k] = np.append(attentionQ2[k], attOut[2][k], 0)


            attentionQ2 = [attentionQ0, attentionQ1, attentionQ2]

        test_predictions_loop2 = np.argmax(test_predictions_loop2, axis=1)+1

        # Measure this fold's accuracy on test set compared to actual labels
        test_score2 = metrics.accuracy_score(y_testy, test_predictions_loop2)
        test_pre = metrics.precision_score(y_testy, test_predictions_loop2, average='macro')
        test_rec = metrics.recall_score(y_testy, test_predictions_loop2, average='macro')
        test_f1= metrics.f1_score(y_testy, test_predictions_loop2, average='macro')

        print(f"test fold score with input {abstractionType}-(accuracy): {test_score2}")

        train_predictions= []
        for bor in range(int(math.ceil(len(x_trains2[0])/limit))):
            train_predictions.extend(n_model2.predict([x_trains2[0][bor*limit:(bor+1)*limit]]))
        train_predictions = np.argmax(train_predictions, axis=1)+1

        if skipDebugSaves:
            return [val_score2, val_pre, val_rec, val_f1], [test_score2, test_pre, test_rec, test_f1], [train_predictions, predictions2], test_predictions_loop2, None, None, None, None, None, None, None, None, None, y_val, y_train1
        else:
            return [val_score2, val_pre, val_rec, val_f1], [test_score2, test_pre, test_rec, test_f1], [train_predictions, predictions2], test_predictions_loop2, n_model2, x_trains2, x_tests2, x_vals2, attentionQ2, earlyPredictor2, newTrain, newVal, newTest, y_val, y_train1

    elif architecture == 'resnet':
        x_train1 = trainD
        x_val = valD
        x_test = testD

        x_train1  = np.expand_dims(x_train1, 1)
        x_val = np.expand_dims(x_val, 1)
        x_test = np.expand_dims(x_test, 1)

        # Create a ResNet model
        model = ResNet(num_resblocks=num_resblocks, num_channels=num_channels, num_classes=num_of_classes, use_1x1conv=use_1x1conv)

        optimizer = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'], run_eagerly=False)

        earlystop = EarlyStopping(monitor= 'val_loss', min_delta=0 , patience=patience, verbose=0, mode='auto', restore_best_weights=True)

        model.fit(x_train1, y_train1, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=BATCH, callbacks=[earlystop], verbose=0)

        train_predictions = model.predict(x_train1)
        train_predictions = np.argmax(train_predictions) + 1
        val_predictions = model.predict(x_val)
        val_predictions = np.argmax(val_predictions, axis=1) + 1
        test_predictions = model.predict(x_test)
        test_predictions = np.argmax(test_predictions, axis=1) + 1

        y_compare = np.argmax(y_val, axis=1)+1

        val_score = metrics.accuracy_score(y_compare, val_predictions)
        val_pre = metrics.precision_score(y_compare, val_predictions, average='macro')
        val_rec = metrics.recall_score(y_compare, val_predictions, average='macro')
        val_f1= metrics.f1_score(y_compare, val_predictions, average='macro')

        test_score = metrics.accuracy_score(y_testy, test_predictions)
        test_pre = metrics.precision_score(y_testy, test_predictions, average='macro')
        test_rec = metrics.recall_score(y_testy, test_predictions, average='macro')
        test_f1= metrics.f1_score(y_testy, test_predictions, average='macro')

        print(f"test fold score with input {abstractionType}-(accuracy): {test_score}")

        return [val_score, val_pre, val_rec, val_f1], [test_score, test_pre, test_rec, test_f1], [train_predictions, val_predictions], test_predictions, None, None, None, None, None, None, None, None, None, y_val, y_train1
        # results = model.evaluate(x_test, y_testy, batch_size=BATCH) # TODO y_testy or y_test?

def preprocessData(x_train1, x_val, x_test, y_train1, y_val, y_test, y_trainy, y_testy, 
                   binNr, symbolsCount, dataName, seqSize, useSaves = False, method='ORI', strategy='uniform', ncoef=None, coef_div=None):    
    """ Preprocess the data """
    
    processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount)
    fileExists = os.path.isfile(processedDataName +'.pkl')

    if(fileExists and useSaves):
        res = helper.load_obj(processedDataName)
        
        for index, v in np.ndenumerate(res):
            res = v
        res.keys()

        x_train1 = res['X_train']
        x_test = res['X_test']
        x_val = res['X_val']
        X_train_ori = res['X_train_ori']
        X_test_ori = res['X_test_ori']
        y_trainy = res['y_trainy']
        y_train1 = res['y_train']
        y_test = res['y_test']
        y_testy = res['y_testy']
        y_val = res['y_val']
        X_val_ori = res['X_val_ori']
             
    else:
        x_train1 = x_train1.squeeze()
        x_val = x_val.squeeze()
        x_test = x_test.squeeze()

        X_train_ori = x_train1.copy()
        X_val_ori = x_val.copy()
        X_test_ori = x_test.copy()

        if strategy == 'quantile':
            # Iteratively remove low variance features for quantile binning
            filter_low_variance = True

            while filter_low_variance:
                print('Delete low variance features')
                bins_edges = np.percentile(
                    x_train1, np.linspace(0, 100, symbolsCount + 1)[1:-1], axis=0
                ).T

                low_var_cols = np.unique(np.where(np.diff(bins_edges, axis=0) == 0)[0])
                print('Deleting: ', low_var_cols)
                x_train1 = np.delete(x_train1, low_var_cols, axis=1)
                x_val = np.delete(x_val, low_var_cols, axis=1)
                x_test = np.delete(x_test, low_var_cols, axis=1)

                # Adapt sequence size to new number of features
                seqSize -= len(low_var_cols)

                bins_edges = np.percentile(
                    x_train1, np.linspace(0, 100, symbolsCount + 1)[1:-1], axis=0
                ).T
                
                low_var_cols = np.unique(np.where(np.diff(bins_edges, axis=0) == 0)[0])
                print('Remaining: ', low_var_cols)
            
                if len(low_var_cols) == 0:
                    print('Aborting')
                    filter_low_variance = False

        if helper.do_symbolize(method):
            mcb = MultipleCoefficientBinning(n_bins=symbolsCount, strategy=strategy)

            mcb = mcb.fit(x_train1)
            x_train1 = helper.symbolizeTrans(x_train1, mcb, sinfo=mcb, bins = symbolsCount)
            x_val = helper.symbolizeTrans(x_val, mcb, sinfo=mcb, bins = symbolsCount)
            x_test = helper.symbolizeTrans(x_test, mcb, sinfo=mcb, bins = symbolsCount)

        if (method == 'SFA') or (method == 'SHAPE_SFA'):
            if ncoef > seqSize:
                ncoefI = seqSize // coef_div
            else:
                ncoefI = ncoef
                
            sfa = SymbolicFourierApproximation(n_coefs=ncoefI, n_bins=symbolsCount, strategy=strategy)
            sfa.fit(x_train1)
            sinfo = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy=strategy)
            x_train1 = helper.symbolizeTrans(x_train1, sfa, sinfo=sinfo, bins = symbolsCount)
            x_val = helper.symbolizeTrans(x_val, sfa, sinfo=sinfo, bins = symbolsCount)
            x_test = helper.symbolizeTrans(x_test, sfa, sinfo=sinfo, bins = symbolsCount)

        elif (method == 'SAX') or (method == 'SHAPE_SAX'):
            sax = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy=strategy)     
            sax.fit(x_train1)
            sinfo = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy=strategy)
            x_train1 = helper.symbolizeTrans(x_train1, sax, sinfo=sinfo, bins = symbolsCount)
            x_val = helper.symbolizeTrans(x_val, sax, sinfo=sinfo, bins = symbolsCount)
            x_test = helper.symbolizeTrans(x_test, sax, sinfo=sinfo, bins = symbolsCount)


        x_train1 = np.expand_dims(x_train1, axis=2)
        x_val = np.expand_dims(x_val, axis=2)
        x_test = np.expand_dims(x_test, axis=2)   
        X_test_ori = np.expand_dims(X_test_ori, axis=2)   
        X_train_ori = np.expand_dims(X_train_ori, axis=2) 
        X_val_ori = np.expand_dims(X_val_ori, axis=2) 

    return x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy, y_testy