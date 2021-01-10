import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from constants import *
import datetime

sns.set()

clfLGBM = LGBMClassifier(n_estimators=200, nthread=-1, seed=42)
clfLGBM2 = LGBMClassifier(n_estimators=180, nthread=-1, seed=42)
clfXGB = xgb.XGBClassifier(min_child_weight=10.0, n_estimators=250, nthread=-1,
                           objective='binary:logistic',
                           max_depth=5,
                           eval_metric='auc',
                           max_delta_step=1.8,
                           colsample_bytree=0.4,
                           subsample=0.8,
                           eta=0.025,
                           gamma=0.65,
                           num_boost_round=391, seed=42)

clfXGB2 = xgb.XGBClassifier(min_child_weight=10.0, n_estimators=330, nthread=-1,
                            objective='binary:logistic',
                            max_depth=5,
                            eval_metric='auc',
                            max_delta_step=1.8,
                            colsample_bytree=0.4,
                            subsample=0.8,
                            eta=0.025,
                            gamma=0.65,
                            num_boost_round=391, seed=42)

clfRand = RandomForestClassifier(max_depth=7, max_features=0.5, criterion='entropy',
                                 n_estimators=160, n_jobs=-1, random_state=42)
clfRand2 = RandomForestClassifier(max_depth=7, max_features=0.5, criterion='entropy',
                                  n_estimators=200, n_jobs=-1, random_state=42)


def fillMedian(data):
    cols = {}
    for i in data.columns:
        if (data[i].isnull().sum() > 0):
            cols[i] = np.nanmedian(data[i])
    for i in cols.keys():
        data[i].fillna(cols[i], inplace=True)


def isExists(fileName):
    return os.path.exists(fileName)


def log(text):
    print(str(datetime.datetime.now().time()) + ": " + text)


def printAcc(clf, name, x_test, y_test):
    log("[" + name + "] Acc: " + str(scoring(clf, x_test, y_test)))


def loadModels():
    lgmbOk = isExists(modelNameLGBM) and isExists(modelNameLGBM2)
    xgbOk = isExists(modelNameXGB) and isExists(modelNameXGB2)
    rfOk = isExists(modelNameRand) and isExists(modelNameRand2)
    if lgmbOk and xgbOk and rfOk:
        log("Загрузка моделей...")
        global clfLGBM
        clfLGBM = joblib.load(modelNameLGBM)
        global clfLGBM2
        clfLGBM2 = joblib.load(modelNameLGBM2)
        global clfRand
        clfRand = joblib.load(modelNameRand)
        global clfRand2
        clfRand2 = joblib.load(modelNameRand2)
        global clfXGB
        clfXGB = joblib.load(modelNameXGB)
        global clfXGB2
        clfXGB2 = joblib.load(modelNameXGB2)
        return True
    else:
        return False


def trainModels():
    log("Тренировка моделей...")

    train = pd.read_csv(pathToTrain)
    train = train.drop('Unnamed: 0', axis=1)
    fillMedian(train)
    X = train.drop('SeriousDlqin2yrs', axis=1)
    y = train['SeriousDlqin2yrs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

    # LGBM
    if clfLGBM is not None and clfLGBM2 is not None:
        log("LGBM Тренировка...")
        clfLGBM.fit(X_train, y_train)
        clfLGBM2.fit(X_train, y_train)
        # acc
        printAcc(clfLGBM, "LGBM", X_test, y_test)
        printAcc(clfLGBM2, "LGBM2", X_test, y_test)
        # save
        log("LGBM Сохранение...")
        joblib.dump(clfLGBM, modelNameLGBM)
        joblib.dump(clfLGBM2, modelNameLGBM2)

    # XGB
    if clfXGB is not None and clfXGB2 is not None:
        log("XGB Тренировка...")
        clfXGB.fit(X_train, y_train)
        clfXGB2.fit(X_train, y_train)
        # acc
        printAcc(clfXGB, "XGB", X_test, y_test)
        printAcc(clfXGB2, "XGB2", X_test, y_test)
        # save
        log("XGB Сохранение...")
        joblib.dump(clfXGB, modelNameXGB)
        joblib.dump(clfXGB2, modelNameXGB2)

    # RF
    if clfRand is not None and clfRand2 is not None:
        log("RF Тренировка...")
        clfRand.fit(X_train, y_train)
        clfRand2.fit(X_train, y_train)
        # acc
        printAcc(clfRand, "RF", X_test, y_test)
        printAcc(clfRand2, "RF2", X_test, y_test)
        # save
        log("RF Сохранение...")
        joblib.dump(clfRand, modelNameRand)
        joblib.dump(clfRand2, modelNameRand2)


def scoring(clf, X, y):
    fpr, tpr, thresholds = roc_curve(y, clf.predict_proba(X)[:, 1])
    roc_auc = auc(fpr, tpr)
    return roc_auc


def predict(test):
    log("Анализ...")
    predClfLGBM1 = clfLGBM.predict_proba(test)[:, 1]
    predClfLGBM2 = clfLGBM2.predict_proba(test)[:, 1]
    predClfXGB = clfXGB.predict_proba(test)[:, 1]
    predClfXGB2 = clfXGB2.predict_proba(test)[:, 1]
    predClfRand = clfRand.predict_proba(test)[:, 1]
    predClfRand2 = clfRand2.predict_proba(test)[:, 1]

    pred = (predClfLGBM1 + 2 * predClfLGBM2 + predClfXGB + predClfXGB2 + predClfRand + predClfRand2) / 7
    return pred


def inputMode():
    RevolvingUtilizationOfUnsecuredLines = float(input("Общий баланс по кредитным картам и личным кредитным линиям,\n"
                                                       "за исключением долга по недвижимости и без рассрочки,\n"
                                                       "например автокредитов, деленный на сумму кредитных лимитов: "))
    age = int(input("Возраст: "))
    NumberOfTime30_59DaysPastDueNotWorse = int(input("Количество раз, когда заемщик был просрочен на 30-59 дней,\n"
                                                     "но не хуже за последние 2 года: "))
    DebtRatio = float(input("Ежемесячные выплаты по долгам, алименты, расходы на проживание, разделенные на месячный\n"
                            "валовой доход: "))
    MonthlyIncome = float(input("Ежемесячный доход: "))
    NumberOfOpenCreditLinesAndLoans = int(input("Количество открытых займов (рассрочка, например, автокредит или\n"
                                                "ипотека) и кредитных линий (например, кредитные карты): "))
    NumberOfTimes90DaysLate = int(input("Количество просроченных платежей заемщика на 90 дней или более: "))
    NumberRealEstateLoansOrLines = int(input("Количество ипотечных кредитов и ссуд на недвижимость, включая кредитные\n"
                                             "линии под залог собственного капитала: "))
    NumberOfTime60_89DaysPastDueNotWorse = int(input("Количество раз, когда заемщик просрочил платеж на 60-89 дней,\n"
                                                     "но не хуже за последние 2 года: "))
    NumberOfDependents = int(input("Количество иждивенцев в семье, исключая их самих (супруга, дети и т. Д.): "))

    df = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': [RevolvingUtilizationOfUnsecuredLines],
        'age': [age],
        'NumberOfTime30-59DaysPastDueNotWorse': [NumberOfTime30_59DaysPastDueNotWorse],
        'DebtRatio': [DebtRatio],
        'MonthlyIncome': [MonthlyIncome],
        'NumberOfOpenCreditLinesAndLoans': [NumberOfOpenCreditLinesAndLoans],
        'NumberOfTimes90DaysLate': [NumberOfTimes90DaysLate],
        'NumberRealEstateLoansOrLines': [NumberRealEstateLoansOrLines],
        'NumberOfTime60-89DaysPastDueNotWorse': [NumberOfTime60_89DaysPastDueNotWorse],
        'NumberOfDependents': [NumberOfDependents]
    })

    log("Риск: " + format(predict(df).item(0) * 100, '.2f') + "%")


def fileMode():
    filename = str(input("Путь к файлу: "))
    test = pd.read_csv(filename)
    test = test.drop(['Unnamed: 0', 'SeriousDlqin2yrs'], axis=1)
    fillMedian(test)

    pred = [format(p * 100, '.2f') for p in predict(test)]
    log("Сохранение результата...")
    s = pd.read_csv(filename)
    s['Risk'] = pred
    s.to_csv(pathToResult, index=False)


if __name__ == '__main__':
    if not loadModels():
        trainModels()

    mode = str(input("[1] Файл\n[2] Ввод\nВыберите режим: "))
    if mode == "1":
        fileMode()
    else:
        inputMode()
