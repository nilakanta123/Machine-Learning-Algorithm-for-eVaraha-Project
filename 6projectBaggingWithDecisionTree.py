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

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()

# # (1) Training Base model
# model = BaggingClassifier(base_estimator=cart, random_state=13)
# result = modelfit(model, X, y, printFeatureImportance=False)
# # ACCURACY : 0.959
# # PRECISION: 0.933
# # RECALL: 0.943
# # f1: 0.936

# # (2) n_estimators
# # 'n_estimators':[85,87,90,93,95,97]
# param_test1 = {
#  'n_estimators':[x for x in range(50,120,5)]
# }
# gsearch1 = GridSearchCV(estimator = BaggingClassifier(base_estimator=cart, random_state=13),
# 	scoring='accuracy',param_grid=param_test1, cv=5)
# gsearch1.fit(X,y)
# plotCvResult(gsearch1,'n_estimators')
# # print("BEST PARAM : ",gsearch1.best_params_)
# # print("ACCURACY : ", gsearch1.best_score_)
# # BEST PARAM :  {'n_estimators': 90}
# # ACCURACY :  0.9606126914660832
model = BaggingClassifier(base_estimator=cart, n_estimators=90, random_state=13)
result = modelfit(model, X, y, printFeatureImportance=False)