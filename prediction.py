import pandas as pd
import numpy as np
import warnings

import xgboost as xgb
from xgboost import plot_importance

from pandas import DataFrame

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.exceptions import ConvergenceWarning

from numpy import sort
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category = ConvergenceWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

#dictionary containing various scores
resultsDict = {}

#auxilary methods
def generateCSV(probability, testDataset):
    print()
    print('************ CSV FILE ***************')
    print()

    idValues = testDataset['Id']
    finalDataset = {
        'Id' : idValues,
        'Category' : probability
    }

    df = DataFrame(finalDataset, columns = ['Id', 'Category'])
    df.to_csv (r'export_dataframe.csv', index = None, header = True)
def calculateAccuracyAndConfusion(model, X, y, name = ""):
    print()
    print('****************** ACCURACY *******************')
    print()

    kf = RepeatedKFold(n_splits = 3, n_repeats = 5, random_state = None) 
    results = cross_val_score(model, X, y, cv=kf)
    resultsDict[name] = results
    #Accuracy measure 
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))

    print()
    print('****************** CONFUSION MATRIX FOR EACH REPEATED K FOLD *******************')
    print()

    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        print(confusion_matrix(y_test, model.predict(X_test)))
def mle(y):
    print()
    print('************** MAXIMUM LIKELIHOOD ***************')
    print()

    maximumLikelihoodEstimate = y.value_counts(normalize = 1)
    print('Maximum likelihood ', maximumLikelihoodEstimate)
def bestParamCalculator(X, y, model, params, applyScalar = False, applyNormalizer = False):

    kf = RepeatedKFold(n_splits = 3, n_repeats = 5, random_state = None) 
    random = RandomizedSearchCV(estimator = model, param_distributions = params, cv = kf, verbose = 2, random_state = 42, n_jobs = -1, scoring='neg_mean_squared_error')
    if applyScalar:
        scaler = RobustScaler().fit(X)
        X = scaler.transform(X)   

    if applyNormalizer:
        normalizer = Normalizer().fit(X)
        X = normalizer.transform(X)

    random.fit(X, y)
    return random.best_estimator_

def showPerformancePlot():
    fig = plt.figure()
    fig.suptitle('Algorithms Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(list(resultsDict.values()))
    ax.set_xticklabels(list(resultsDict.keys()))
    plt.savefig('performance.png')
    plt.show()

#Covered in class
def logistic(X, y, test, applyScalar = False):
    print()
    print('************** KCROSS VALIDATION AND LOGISTIC REGRESSION ***************')
    print()
    print('+++++++ FIND VALUE OF C +++++++')
    print()

    n = np.arange(-2,3)
    r = pow(float(10),n)
    bestCValue = 1.0
    maxAccuracy = 0

    kf = RepeatedKFold(n_splits = 3, n_repeats = 5, random_state = None) 
    for c in r:
        maxScoreValues = []
    
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index] 
            y_train, y_test = y[train_index], y[test_index]

            if applyScalar:
                scaler = RobustScaler().fit(X_train) #Scale cause of different variations in data set
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            lr = LogisticRegression(C = c).fit(X_train, y_train)
            # print('\n'"Training Accuracy of L1 LogRess with C=%f:%f" % (c, lr_l1.score(X_train,y_train)))
            # print('\n'"Test Accuracy of L1 LogRegss with C=%f: %f" % (c, lr_l1.score(X_test,y_test)))
            maxScoreValues.append(lr.score(X_test, y_test))

        average = sum(maxScoreValues)/len(maxScoreValues)
        if average > maxAccuracy:
            maxAccuracy = average
            bestCValue = c

    print()
    print('+++++++ LOGISTIC +++++++')
    print()

    lr = LogisticRegression(C = bestCValue)
    
    if applyScalar:
        X = RobustScaler().fit(X).transform(X)
    
    lr.fit(X, y)
    calculateAccuracyAndConfusion(lr, X, y, 'Logistic Regression')
    probability = lr.predict(test)
    
    return probability

#dont use any scaling/normalization as the train dataset kept causing "can't deal with negative values" 
# when applying robust and normalizer caused values to be too small
def multinomial(X, y, testDataset, applyNormalizer = False):
    print()
    print('**************** MULTINOMIAL NB *******************')
    print()

    n = np.arange(-2,3)
    r = pow(float(10),n)

    params = {
        'alpha' : r
    }

    bestModel = bestParamCalculator(X, y, MultinomialNB(), params, applyNormalizer)
    probability = bestModel.predict(testDataset)
    calculateAccuracyAndConfusion(bestModel, X, y, 'Multinomial NB')
    return probability
def mlpClassifier(X, y, testDataset, applyScalar = False):

    print()
    print('**************** MLP Classifier *******************')
    print()

    n = np.arange(-7,1)
    r = pow(float(10),n)
    params = {
    #30 because roughly these many features we have and lastly confirming with default value
    'hidden_layer_sizes': [(30,30,30), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': r,
    'learning_rate': ['constant','adaptive'],
    }
    bestModel = bestParamCalculator(X, y, MLPClassifier(max_iter=100), params, applyScalar)
    probability = bestModel.predict(testDataset)
    calculateAccuracyAndConfusion(bestModel, X, y, 'MLPClassifier')
    return probability

#Extra ones 
def gaussian(X, y, testDataset, applyScalar = False):
   
    print()
    print('**************** GAUSSIAN NB ****************')
    print()

    n = np.arange(-20,-5)
    r = pow(float(10),n)
    params = {
        'var_smoothing' : r
    }

    model = GaussianNB()
    bestModel = bestParamCalculator(X, y, model, params, applyScalar)
    probability = bestModel.predict(testDataset)
    calculateAccuracyAndConfusion(model, X, y, 'Gaussian NB')
    return probability
def randomForest(X, y, testDataset, applyScalar = False):
    
    print()
    print('************ RANDOM FOREST WITH RANDOMIZED SEARCH ***************')
    print()

    n_estimators = [int(x) for x in np.linspace(start = 5, stop = 200, num = 5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]
    min_samples_split = [5, 10]

    random_grid = { 
                'n_estimators': n_estimators, 
                'max_features': max_features, 
                'max_depth': max_depth, 
                'min_samples_split': min_samples_split
    }

    bestModel = bestParamCalculator(X, y, RandomForestClassifier(), random_grid, applyScalar)
    probability = bestModel.predict(testDataset)
    calculateAccuracyAndConfusion(bestModel, X, y, 'Random Forest')
    
    return probability
def knn(X, y, testDataset, applyScalar = False):
    
    print()
    print('************ K NEAREST NEIGHBOURS ***************')
    print()

    random_grid = { 
                'n_neighbors': range(1,10), 
                'algorithm': ['auto', 'kd_tree'], 
                'weights' : ['distance', 'uniform']
    }

    bestModel = bestParamCalculator(X, y, KNeighborsClassifier(), random_grid, applyScalar)
    
    probability = bestModel.predict(testDataset)
    calculateAccuracyAndConfusion(bestModel, X, y, 'K Nearest Neighbors')
    
    return probability
def applyXGBoostFeatureSelection(X, y, testDataset, applyScalar = False):
    
    print()
    print('************ XG BOOST ***************')
    print()

    xgbClassifier = xgb.XGBClassifier(n_estimators=2000,learning_rate=0.3)
    if applyScalar:
        scaler = RobustScaler().fit(X) #Scale cause of different variations in data set
        X = scaler.transform(X)    
    xgbClassifier.fit(X, y)

    # plot_importance(xgbClassifier)
    # plt.show()

    thresholds = sort(xgbClassifier.feature_importances_)

    bestThresholdValue = 1.0
    maxAccuracy = 0
    
    print()
    print('++++++++++ Update dataset after selecting feature from threshold value ++++++++++')
    print()

    for value in thresholds:

        kf = RepeatedKFold(n_splits = 3, n_repeats = 5, random_state = None) 
        selection = SelectFromModel(xgbClassifier, threshold=value, prefit=True)
        selected_dataset = selection.transform(X)
        selection_model = xgb.XGBClassifier()
        selection_model.fit(selected_dataset, y)

        results = cross_val_score(selection_model, selected_dataset, y, cv=kf, scoring="accuracy")
        print("Thresh=%.3f, Accuracy: %.2f%%" % (value, results.mean() * 100.0))
        averageScore = results.mean()
        if averageScore > maxAccuracy:
            maxAccuracy = averageScore
            bestThresholdValue = value
        
    
    finalClassifier = xgb.XGBClassifier(n_estimators=2000,learning_rate=0.3)
    
    print()
    print('++++++++ Select features from best accuracy threshold value +++++++')
    print()

    selection = SelectFromModel(xgbClassifier, threshold=bestThresholdValue, prefit=True)
    selected_dataset = selection.transform(X)
    
    # feature_idx = selection.get_support()
    # trainDataset = pd.read_csv('./cs6501-final/train.csv')
    # feature_name = trainDataset.iloc[:, :-1].columns[feature_idx]
    ## selected features
    # print(feature_name)

    finalClassifier.fit(selected_dataset, y)
    calculateAccuracyAndConfusion(finalClassifier, selected_dataset, y, 'XG Boost Classifier')
    return finalClassifier.predict(selection.transform(testDataset))

trainDataset = pd.read_csv('./cs6501-final/train.csv')
testDataset = pd.read_csv('./cs6501-final/test.csv')
y = trainDataset['Category']
X = trainDataset.iloc[:, :-1].values #without category

mle(y)

print()
print('============== FINDING PREDICTIONS AND TRAINING ================')
print()

# probabilityF = randomForest(X, y, testDataset, True)
# probabilityG = gaussian(X, y, testDataset, True)
# probabilityM = multinomial(X, y, testDataset)
# probabilityMC = mlpClassifier(X, y, testDataset, True)
# probabilityK = knn(X, y, testDataset, True)
# probabilityL = logistic(X, y, testDataset, True)
probabilityA = applyXGBoostFeatureSelection(X, y, testDataset, True)

# showPerformancePlot()

# generateCSV(probability, testDataset)