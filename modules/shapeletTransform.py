import numpy as np
import pandas as pd
from modules import helper
import math
from scipy.interpolate import interp1d
import time
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform


def evaluateShapelets(pipeline, X_test, y_testy, basedata):
    """ Evaluate shapelets results """
    results = dict()

    x_test = X_test
    x_test = np.array(x_test).squeeze()

    dftest = pd.DataFrame({"dim_0": [pd.Series(x) for x in x_test]})

    test_x = dftest
    test_y = y_testy

    preds = pipeline.predict(test_x)
    baselinePreds = pipeline.predict(basedata)

    results['Test Accuracy'] = metrics.accuracy_score(y_testy , preds)
    results['Test Precision'] = metrics.precision_score(y_testy, preds, average='macro')
    results['Test Recall'] = metrics.recall_score(y_testy, preds, average='macro')
    results['Test F1'] = metrics.f1_score(y_testy, preds, average='macro')
    results['Train Predictions'] = baselinePreds
    results['Test Predictions'] = preds
    correct = sum(preds == test_y)
    results['correct'] = correct
    results['maxCorrect'] = len(test_y)
    
    return results

def trainShapelets(x_train, y_train, min_shapelet_length, seed, reduceTrainy=True, time_limit_in_minutes=1):
    """ Train Shapelets """
    x_train = np.array(x_train).squeeze()
    if reduceTrainy:
        predictions = np.argmax(y_train, axis=1) + 1
    else:
        predictions = y_train

    # Convert to sktime-compatible format
    dftrain = pd.DataFrame({"dim_0": [pd.Series(x) for x in x_train]})
    train_x, train_y = dftrain, predictions

    pipeline = Pipeline(
        [
            (
                "st",
                RandomShapeletTransform(
                    min_shapelet_length=min_shapelet_length,
                    time_limit_in_minutes=time_limit_in_minutes,
                    random_state=seed,
                    n_jobs=-1
                ),
            ),
            ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
        ]
    )

    pipeline.fit(train_x, train_y)

    return pipeline, train_x

