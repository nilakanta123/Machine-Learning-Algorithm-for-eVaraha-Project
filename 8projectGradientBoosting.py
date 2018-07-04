import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
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
        feat_imp.plot(kind='bar', title='Feature Importances using GradientBoosting')
        plt.ylabel('Feature Importance Score')
        plt.tight_layout()
        plt.show()
    return feat_imp

def plotGridCvResult(gridresult):
    x=[]
    xx = "%d, %d, %d, %r, %r, %r" % (gridresult.best_params_['n_estimators'],gridresult.best_params_['min_samples_split'],
      gridresult.best_params_['min_samples_leaf'],gridresult.best_params_['max_features'],gridresult.best_params_['max_depth'],
      gridresult.best_params_['bootstrap'])
    for i in gridresult.cv_results_['params']:
        x.append("%d, %d, %d, %r, %r, %r" % (i['n_estimators'],i['min_samples_split'],i['min_samples_leaf'],i['max_features'],i['max_depth'],i['bootstrap']))

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
def plotCvResult(gridresult,param_name):
   import matplotlib.pyplot as plt
   x = [i[param_name] for i in gridresult.cv_results_['params']]
   plt.plot(x,gridresult.cv_results_['mean_train_score'],color='blue',marker='o',markersize=5,label='train accuracy')
   plt.plot(x,gridresult.cv_results_['mean_test_score'],color='green',marker='x',markersize=5,label='test accuracy')  
   plt.plot(gridresult.best_params_[param_name],gridresult.best_score_,color='red',marker='+',markersize=10)    
   plt.xlabel(param_name)
   plt.ylabel('Accuracy')
   plt.title("Best parameter is %r with score %0.3f"% (gridresult.best_params_,gridresult.best_score_))
   plt.legend(loc='lower right')
   # plt.ylim([0.5,1])
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

from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

# # (1) DEFAULT BASELINE MODEL
# baseline_model = GradientBoostingClassifier(random_state=10)
# modelfit(baseline_model,X, y)

# # (2) TUNNING n_estimators
# param_test1 = {'n_estimators':range(30,101,10)}
# gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=2, min_samples_leaf=1,
#     max_depth=3, max_features='sqrt', subsample=1.0), param_grid = param_test1, scoring='accuracy',n_jobs=4, iid=False,
#     cv=5)
# gsearch1.fit(X,y)
# plotCvResult(gsearch1,'n_estimators')
# print("BEST PARAMERER : ",gsearch1.best_params_)
# print("BEST SCORE : ",gsearch1.best_score_)
# # BEST PARAMERER : {'n_estimators': 80}
# # BEST SCORE : 0.9585602007104355

# # (3) TUNNING max_depth AND min_samples_split
# param_test2 = {'max_depth':range(1,10,2), 'min_samples_split':range(2,20,4)}
# gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, min_samples_leaf=1,
#   max_features='sqrt', subsample=1.0), param_grid = param_test2, scoring='accuracy',n_jobs=4, iid=False, cv=5)
# gsearch2.fit(X,y)
# print("BEST PARAMERER : ",gsearch2.best_params_)
# print("BEST SCORE : ",gsearch2.best_score_)
# # showing graphically
# x=[]
# for i in gsearch2.cv_results_['params']:
#     x.append("%d, %d" % (i['max_depth'],i['min_samples_split']))
# xx = "%d, %d" % (gsearch2.best_params_['max_depth'],gsearch2.best_params_['min_samples_split'])
# ax = plt.axes()
# ax.plot(x, gsearch2.cv_results_['mean_train_score'], color='blue', marker='x', markersize=7, label='train accuracy')
# ax.plot(x, gsearch2.cv_results_['mean_test_score'], color='green', marker='o', markersize=5,label='test accuracy')
# ax.plot(xx, gsearch2.best_score_, color='red', marker='o', markersize=10)
# ax.xaxis.set_major_locator(plt.MaxNLocator(80))
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.title('Best Param is %r with accuracy %0.5f' % (gsearch2.best_params_,gsearch2.best_score_))
# plt.xlabel('Parameters combination')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.ylim([0.5,1])
# plt.show()
# # BEST PARAMERER :  {'max_depth': 1, 'min_samples_split': 14}
# # BEST SCORE :  0.9637651091735598


# # (4) TUNNING min_sample_lift
# param_test3 = {'min_samples_leaf':range(1,61,10)}
# gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(max_depth=1, min_samples_split=14, n_estimators=80, learning_rate=0.1,
#   max_features='sqrt', subsample=1.0), param_grid = param_test3, scoring='accuracy',n_jobs=4,iid=False, cv=5)
# gsearch3.fit(X,y)
# plotCvResult(gsearch3,'min_samples_leaf')
# print("BEST PARAMERER : ",gsearch3.best_params_)
# print("BEST SCORE : ",gsearch3.best_score_)
# # BEST PARAMERER :  {'min_samples_leaf': 11}
# # BEST SCORE :  0.9677157264575105


# # (5) TUNNING max_features
# max_features = [x for x in range(8,20)]
# max_features.append('sqrt')
# param_test4 = {'max_features':max_features}
# gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(min_samples_leaf=11, max_depth=1, min_samples_split=14, n_estimators=80,
#   learning_rate=0.1, max_features='sqrt', subsample=1.0), param_grid = param_test4, scoring='accuracy',n_jobs=4,iid=False, cv=5)
# gsearch4.fit(X,y)
# plotCvResult(gsearch4,'max_features')
# print("BEST PARAMERER : ",gsearch4.best_params_)
# print("BEST SCORE : ",gsearch4.best_score_)
# # BEST PARAMERER :  {'max_features': 13}
# # BEST SCORE :  0.969197207938992


# # (6) TUNNING subsample
# param_test5 = {'subsample':[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]}
# gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(max_features=13, min_samples_leaf=11, max_depth=1, min_samples_split=14, n_estimators=80,
#   learning_rate=0.1, subsample=1.0), param_grid = param_test5, scoring='accuracy',n_jobs=4,iid=False, cv=5)
# gsearch5.fit(X,y)
# plotCvResult(gsearch5,'subsample')
# print("BEST PARAMERER : ",gsearch5.best_params_)
# print("BEST SCORE : ",gsearch5.best_score_)
# # EST PARAMERER :  {'subsample': 0.9}
# # BEST SCORE :  0.969197207938992

# # (7) DECREASE learning_rate AND INCREASE n_estimators
tune_model=GradientBoostingClassifier(n_estimators=160,learning_rate=0.05, subsample=0.9, max_features=13, min_samples_leaf=11, max_depth=1, min_samples_split=14)
modelfit(tune_model,X, y)