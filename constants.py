import os

path = "data/"
pathToTrain = os.path.join(path, "cs-training.csv")
pathToTest = os.path.join(path, "cs-test.csv")
pathToSingle = os.path.join(path, "single.csv")
pathToEntry = os.path.join(path, "sampleEntry.csv")
pathToResult = os.path.join(path, "ans.csv")

pathToModels = "models/"
modelNameLGBM = os.path.join(pathToModels, "LGBM")
modelNameLGBM2 = os.path.join(pathToModels, "LGBM2")
modelNameXGB = os.path.join(pathToModels, "XGB")
modelNameXGB2 = os.path.join(pathToModels, "XGB2")
modelNameRand = os.path.join(pathToModels, "Rand")
modelNameRand2 = os.path.join(pathToModels, "Rand2")
