import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, validation_curve, train_test_split, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score
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
def plotCvResult(gridresult,param_list,param_name):
    plt.figure(figsize=(15, 10))
    plt.plot(param_list,gridresult.cv_results_['split1_train_score'],color='blue',marker='o',markersize=7,label='training accuracy')   
    plt.plot(param_list,gridresult.cv_results_['split1_test_score'],color='green',marker='x',markersize=7,label='test accuracy')    
    plt.xlabel(param_name)
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

from sklearn.neighbors import KNeighborsClassifier

# # (1) TRAINING MODEL on Default Parameter
# base_model = KNeighborsClassifier()
# modelfit(base_model,X,y,printFeatureImportance=False)

# # (2) TUNNING PARAMETER k
# param_test1 = {
#     'n_neighbors' : [x for x in range(5,21)]
# }
# gsearch1 = GridSearchCV(estimator = KNeighborsClassifier(),
#   scoring='accuracy',param_grid=param_test1, cv=10)
# gsearch1.fit(X,y)
# plotCvResult(gsearch1,param_test1['n_neighbors'],'n_neighbors')
# print("BEST PARAM : ",gsearch1.best_params_)
# print("ACCURACY : ", gsearch1.best_score_)
# # BEST PARAM :  {'n_neighbors': 11}
# # ACCURACY :  0.936542669584245

# (3) FINAL MODEL FITTING
# tuned_model = KNeighborsClassifier(n_neighbors=11)
# modelfit(tuned_model,X,y,printFeatureImportance=False)

