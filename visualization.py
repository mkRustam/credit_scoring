import lightgbm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
# Отображение корреляции между всеми колонками
from IPython.core.display import display
from sklearn.tree import export_graphviz


def data_heatmap(X):
    corr = X.corr()
    sns.heatmap(corr,
                annot=True,
                xticklabels=corr.columns,
                yticklabels=corr.columns)

    plt.show()


def rf_buildTree(clfRF):
    # Extract single tree
    estimator = clfRF.estimators_[5]

    # Export as dot file
    export_graphviz(estimator,
                    out_file='tree.dot',
                    rounded=True, proportion=False,
                    precision=2, filled=True)
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    # Image(filename='tree.png')


def rf_features(clfRF, data):
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('SeriousDlqin2yrs', axis=1)
    feats = {}
    for feature, importance in zip(data.columns, clfRF.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale=5)
    sns.set(style="whitegrid", color_codes=True, font_scale=1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Importance', fontsize=25, weight='bold')
    plt.ylabel('Features', fontsize=25, weight='bold')
    plt.title('Feature Importance', fontsize=25, weight='bold')
    display(plt.show())
    display(importances)


def lgbm_features(clf):
    ax = lightgbm.plot_importance(clf)
    plt.show()


def lgbm_tree(clf):
    ax = lightgbm.plot_tree(clf)
    plt.show()


def xgb_features(clf):
    ax = xgb.plot_importance(clf)
    plt.show()

def xgb_tree(clf):
    ax = xgb.plot_tree(clf)
    plt.show()