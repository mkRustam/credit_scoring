# explore lightgbm number of trees effect on performance
from numpy import mean
from numpy import std
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot
import pandas as pd
from constants import *
import numpy as np


def fillMedian(data):
    cols = {}
    for i in data.columns:
        if data[i].isnull().sum() > 0:
            cols[i] = np.nanmedian(data[i])
    for i in cols.keys():
        data[i].fillna(cols[i], inplace=True)


# get the dataset
def get_dataset():
    train = pd.read_csv(pathToTrain)
    train = train.drop('Unnamed: 0', axis=1)
    fillMedian(train)
    X = train.drop('SeriousDlqin2yrs', axis=1)
    y = train['SeriousDlqin2yrs']
    return X, y


# get a list of models to evaluate
def get_models():
    models = dict()
    trees = [10, 50, 100, 200, 500]
    for n in trees:
        models[str(n)] = LGBMClassifier(n_estimators=n)
    return models


# get a list of models to evaluate
def get_models_xgb():
    models = dict()
    trees = [10, 50, 100, 200, 500]
    for n in trees:
        models[str(n)] = xgb.XGBClassifier(min_child_weight=10.0, n_estimators=n, nthread=-1,
                                           objective='binary:logistic',
                                           max_depth=5,
                                           eval_metric='auc',
                                           max_delta_step=1.8,
                                           colsample_bytree=0.4,
                                           subsample=0.8,
                                           eta=0.025,
                                           gamma=0.65,
                                           num_boost_round=391, seed=42)
    return models


# get a list of models to evaluate
def get_models_rf():
    models = dict()
    trees = [10, 50, 100, 200, 500]
    for n in trees:
        models[str(n)] = RandomForestClassifier(max_depth=7, max_features=0.5, criterion='entropy',
                                 n_estimators=n, n_jobs=-1, random_state=42)
    return models


# evaluate a give model using cross-validation
def evaluate_model(model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models_rf()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
