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
        feat_imp.plot(kind='bar', title='Feature Importances using RandomForestClassifier')
        plt.ylabel('Feature Importance Score')
        plt.tight_layout()
        plt.show()
    return feat_imp

def plotCvResult(gridresult):
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


from sklearn.ensemble import RandomForestClassifier

# # (1) USE RANDOM GRID SEARCH FOR BEST HYPERPARAMETERS
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# # Fit the random search model
# rf_random.fit(X, y)
# print(rf_random.cv_results_)
# plotCvResult(rf_random)
# print("BEST PARAMETERS : ",rf_random.best_params_)


# (2) NOW TRAIN THE TUNED MODEL TO FIND ITS PERMORMANCE
model = RandomForestClassifier(n_estimators=1600,max_features='auto',max_depth=10,min_samples_split=5,min_samples_leaf=1,bootstrap=True)
fi = modelfit(model,X,y)
