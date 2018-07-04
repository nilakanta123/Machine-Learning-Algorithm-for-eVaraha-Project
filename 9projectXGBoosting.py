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

def plotCvResult2(gridresult,param1,param2):
  x=[]
  for i in gridresult.cv_results_['params']:
    x.append("%r, %r" % (i[param1],i[param2]))
  xx = "%r, %r" % (gridresult.best_params_[param1],gridresult.best_params_[param2])
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
  # plt.ylim([0.5,1])
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

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt

# # (1) DEFAULT BASELINE MODEL
# base_model=XGBClassifier(objective= 'multi:softprob',num_class=65)
# modelfit(base_model,X, y)
# # ACCURACY : 0.947
# # PRECISION: 0.913
# # RECALL: 0.928
# # f1: 0.918

# # (2) TUNNING max_depth AND min_child_weight
# param_test1 = { 'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2) }
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, gamma=0, subsample=1,
#   colsample_bytree=1, objective= 'multi:softprob',num_class=65, scale_pos_weight=1 ), param_grid = param_test1, scoring='accuracy',
#   n_jobs=4,iid=False, cv=5)
# gsearch1.fit(X,y)
# plotCvResult2(gsearch1,'max_depth','min_child_weight')
# print("BEST PARAMERER : ",gsearch1.best_params_)
# print("BEST SCORE : ",gsearch1.best_score_)
# BEST PARAMERER :  {'max_depth': 3, 'min_child_weight': 1}
# BEST SCORE :  0.9494667262836277

# # (3) TUNNING gamma
# param_test2 = {'gamma':[i/10.0 for i in range(0,10)]}
# gsearch2 = GridSearchCV(estimator = XGBClassifier(max_depth=3, min_child_weight =1, learning_rate =0.1, n_estimators=100, gamma=0, subsample=1,
#     colsample_bytree=1, objective= 'multi:softprob',num_class=65, scale_pos_weight=1 ), param_grid = param_test2, scoring='accuracy',
#     n_jobs=4,iid=False, cv=5)
# gsearch2.fit(X,y)
# plotCvResult(gsearch2,'gamma')
# print("BEST PARAMERER : ",gsearch2.best_params_)
# print("BEST SCORE : ",gsearch2.best_score_)
# # BEST PARAMERER :  {'gamma': 0.1}
# # BEST SCORE :  0.95280214621059691

# # (4) TUNNING subsample AND colsample_bytree
# param_test3 = {
#  'subsample':[i/100.0 for i in range(85,105,5)],
#  'colsample_bytree':[i/100.0 for i in range(30,55,5)]
# }
# gsearch3 = GridSearchCV(estimator = XGBClassifier(gamma=0.1, max_depth=3, min_child_weight =1, learning_rate =0.1, n_estimators=100,
#     objective= 'multi:softprob',num_class=65, scale_pos_weight=1 ), param_grid = param_test3, scoring='accuracy',
#     n_jobs=4,iid=False, cv=5)
# gsearch3.fit(X,y)
# plotCvResult2(gsearch3,'subsample','colsample_bytree')
# print("BEST PARAM : ",gsearch3.best_params_)
# print("ACCURACY : ", gsearch3.best_score_)
# # BEST PARAM :  {'colsample_bytree': 0.45, 'subsample': 1.0}
# # ACCURACY :  0.9548021462105968

# # (4) TUNNING REGULARIZATION PARAMETERS reg_alpha AND reg_lambda
# param_test4 = {
#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
#  'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
# }
# gsearch4 = GridSearchCV(estimator = XGBClassifier(colsample_bytree=0.45, subsample=1.0, gamma=0.1, max_depth=3, min_child_weight =1, learning_rate =0.1, n_estimators=100,
#     objective= 'multi:softprob',num_class=65, scale_pos_weight=1 ), param_grid = param_test4, scoring='accuracy',
#     n_jobs=4,iid=False, cv=5)
# gsearch4.fit(X,y)
# plotCvResult2(gsearch4,'reg_alpha','reg_lambda')
# print("BEST PARAM : ",gsearch4.best_params_)
# print("ACCURACY : ", gsearch4.best_score_)
# # BEST PARAM :  {'reg_alpha': 1e-05, 'reg_lambda': 1}
# # ACCURACY :  0.9548021462105968

# (5) TRAINING FINAL MODEL WITH HALF LEARNING RATE AND DOUBLE THE # OF ESTIMATORS
final_model=XGBClassifier(learning_rate =0.05, n_estimators=200, colsample_bytree=0.45, subsample=1.0, gamma=0.1, max_depth=3, min_child_weight =1,
  objective= 'multi:softprob',num_class=65, scale_pos_weight=1 )
modelfit(final_model,X, y)
# ACCURACY : 0.950
# PRECISION: 0.914
# RECALL: 0.929
# f1: 0.919