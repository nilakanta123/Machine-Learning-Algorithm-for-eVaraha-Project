import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from sklearn.model_selection import GridSearchCV, cross_validate
import matplotlib.pyplot as plt

# @@@@@@@@@ Function to find performance of the model using 10 folds cross-validation @@@@@@
def modelfit(alg, feature, label, performCV=True, printFeatureImportance=True, cv_folds=10):
    alg.fit(feature, label)
    
    print ("\n~~ Model Report ~~~~~~~~")
    scoring = {'accuracy': 'accuracy',
    'precision' : 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro'}

    if performCV:
        prec_scores = cross_validate(alg, feature, label, cv=cv_folds, scoring = scoring)
        # print(prec_scores)
        print("ACCURACY : %0.3f" % prec_scores['test_accuracy'].mean())
        print("PRECISION: %0.3f" % prec_scores['test_precision'].mean())
        print("RECALL: %0.3f" % prec_scores['test_recall'].mean())
        print("f1: %0.3f" % prec_scores['test_f1'].mean())

    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, X.columns).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
    return alg

def plotCvResult(gridresult):
    x=[]
    xx = "%s, %s, %0.5f" % (gridresult.best_params_['kernel'],gridresult.best_params_['C'],gridresult.best_params_['gamma'])
    for i in gridresult.cv_results_['params']:
        x.append("%s, %s, %0.5f" % (i['kernel'],i['C'],i['gamma']))

    ax = plt.axes()
    ax.plot(x, gridresult.cv_results_['mean_train_score'], color='blue', marker='x', markersize=7, label='train accuracy')
    ax.plot(x, gridresult.cv_results_['mean_test_score'], color='green', marker='o', markersize=5,label='test accuracy')
    ax.plot(xx, gridresult.best_score_, color='red', marker='o', markersize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(80))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.title('Best Param is %r with accuracy %0.5f' % (gridresult.best_params_,gridresult.best_score_))
    plt.xlabel('Parameters combination')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5,1])
    plt.show()

# @@@@@@@@ READING DATA @@@@@@@
df = pd.read_csv('am.csv')
# print(df)
df.drop(['Probable_agent'], axis=1, inplace=True)

# @@@@@@@@ LABEL ENCODING CLASSES @@@@@@@
from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(df['Probable_disease'])
df['Probable_disease']=labelEncoder.transform(df['Probable_disease'])

# @@@@@@@@ SET FEATURE AND TARGET @@@@@@@@@@@
from sklearn.utils import shuffle
X, y = shuffle(df.iloc[:,:-1],df.Probable_disease, random_state=13)

from sklearn.svm import SVC

# # (1) TRAINING MODEL on Default Parameter
# base_model = SVC()
# modelfit(base_model,X,y,printFeatureImportance=False)
# # ACCURACY : 0.660
# # PRECISION: 0.559
# # RECALL: 0.628
# # f1: 0.574

# # (2) CHOICE PARAMETER metric
# tuned_parameters = {
# 'kernel': ['sigmoid','rbf','linear'],
# 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
# 'C': [0.001, 0.10, 0.1, 10, 25, 26, 50, 51, 100, 1000]}
# gsearch1 = GridSearchCV(estimator = SVC(),
#   scoring='accuracy',param_grid=tuned_parameters, cv=10)
# gsearch1.fit(X,y)
# plotCvResult(gsearch1)
# print("BEST PARAM : ",gsearch1.best_params_)
# print("ACCURACY : ", gsearch1.best_score_)
# # BEST PARAM :  {'C': 26, 'gamma': 0.01, 'kernel': 'rbf'}
# # ACCURACY :  0.9693654266958425

tune_model = SVC(C=26, gamma=0.01, kernel='rbf')
modelfit(tune_model,X,y,printFeatureImportance=False)
