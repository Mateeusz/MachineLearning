import pandas as pd

import numpy
from scipy.stats import ks_2samp
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from numpy import mean
from src.ksType import ksType
from src.Instance import Instance
from ReliefF import ReliefF

# Configuration
dataFile = 'resources/breast-cancer-wisconsin.data'

skipRowsWithInvalidFeature = True   # Skip rows with '?' in features
defaultValueOfInvalidFeature = 1    # Replace '?' with value if skipRowsWithInvalidFeature is False

# Data settings
columnId = 0            # Index of column with id
columnCancerClass = 10  # Index of column with cancer class
columnFirstFeature = 1  # Index of column with first feature

quantityOfFeatures = 9  # Quantity of features in file

# Algorithms settings
kNN = [1, 5, 10]                            # kNN settings
metricTypes = ['euclidean', 'manhattan']    # Distance metrics


def main():
    instances = loadDataFromFile(dataFile, skipRowsWithInvalidFeature = False, defaultValueOfInvalidFeature = 1)

    rankingKolmogorov = kolmogorovTest(instances, quantityOfFeatures)
    rankingKolmogorovIds = [x.getParamID() for x in rankingKolmogorov]

    rankingKBest = kBestTest(instances, quantityOfFeatures)
    rankingKBestIds = [x.getParamID() for x in rankingKBest]

    rankingReliefF = reliefFTest(instances, quantityOfFeatures)
    rankingReliefFIds = [x.getParamID() for x in rankingReliefF]

    print("===============")
    print(rankingKolmogorovIds)
    print("===============")
    print(rankingKBestIds)
    print("===============")
    print(rankingReliefFIds)

    # teachingData, testData = divideInstances(instances)
    # print("rankingKolmogorov")
    # for feature in range(0, rankingKolmogorov.__len__()):
    #     print(rankingKolmogorov[feature].getParamID() + 1, '\t', rankingKolmogorov[feature].getPValue(), '\t', rankingKolmogorov[feature].getStatistic())
    #
    # print("rankingKBest")
    # for feature in range(0, rankingKBest.__len__()):
    #     print(rankingKBest[feature].getParamID() + 1, '\t', rankingKBest[feature].getPValue(), '\t', rankingKBest[feature].getStatistic())


    # teachingData, testData = splitInstances(instances, rankingIds)
    # score, matrix = kNNAlgorithm(5, teachingData, testData, rankingIds, 'euclidean')
    # score1, matrix2 = naiveBayesAlgorithm(teachingData, testData, rankingIds)

    # print(matrix)


    scoresKNN = crossValidationKNN(kNN, metricTypes, instances, rankingKBest)
    scoresNB = crossValidationNB(instances, rankingKBest)
    scoresDT = crossValidationDT(instances, rankingKBest)
    scoresLSVC = crossValidationLinearSVM(instances, rankingKBest)
    scoresRBF = crossValidationRBF_SVM(instances, rankingKBest)

    kscoresKNN = crossValidationKNN(kNN, metricTypes, instances, rankingKolmogorov)
    kscoresNB = crossValidationNB(instances, rankingKolmogorov)
    kscoresDT = crossValidationDT(instances, rankingKolmogorov)
    kscoresLSVC = crossValidationLinearSVM(instances, rankingKolmogorov)
    kscoresRBF = crossValidationRBF_SVM(instances, rankingKolmogorov)

    rscoresKNN = crossValidationKNN(kNN, metricTypes, instances, rankingReliefF)
    rscoresNB = crossValidationNB(instances, rankingReliefF)
    rscoresDT = crossValidationDT(instances, rankingReliefF)
    rscoresLSVC = crossValidationLinearSVM(instances, rankingReliefF)
    rscoresRBF = crossValidationRBF_SVM(instances, rankingReliefF)
    print("scoresKNN k")
    print(kscoresKNN)
    print("scoresKNN")
    print(scoresKNN)
    print("scoresKNN R")
    print(rscoresKNN)

    print("scoresNB k")
    print(kscoresNB)
    print("scoresNB")
    print(scoresNB)
    print("scoresNB R")
    print(rscoresNB)

    print("scoresDT k")
    print(kscoresDT)
    print("scoresDT")
    print(scoresDT)
    print("scoresDT R")
    print(rscoresDT)

    print("linearSVC k")
    print(kscoresLSVC)
    print("linearSVC")
    print(scoresLSVC)
    print("linearSVC R")
    print(rscoresLSVC)

    print("RBF SVC k")
    print(kscoresRBF)
    print("RBF SVC")
    print(scoresRBF)
    print("RBF SVC R")
    print(rscoresRBF)

def loadDataFromFile(fileName, skipRowsWithInvalidFeature = False, defaultValueOfInvalidFeature = 1):
    instances = []

    file = open(fileName, 'r').read()
    lines = file.split('\n')

    for line in lines:
        if '?' in line and skipRowsWithInvalidFeature is True:
            continue

        row = line.split(',')

        if row.__len__() == 11:
            instance = Instance(row[columnId], row[columnFirstFeature:quantityOfFeatures + 1], row[columnCancerClass], defaultValueOfInvalidFeature)
            instances.append(instance)

    return instances

def read_data(col_min, col_max):
    data = pd.read_csv(dataFile, encoding='utf-8', sep=',', header=None).iloc[:, col_min:col_max]
    if col_max:
        return data
    else:
        return data.to_numpy().ravel()

def kolmogorovTest(instances, quantityOfFeatures):
    featuresRanking = []
    dataDir = {
        '2': {},
        '4': {}
    }

    for feature in range(0, quantityOfFeatures):
        dataDir['4'][feature] = []
        dataDir['2'][feature] = []
        for instance in instances:
            dataDir[instance.getCancerType()][feature].append(instance.getFeatureValues()[feature])

        statistic, pValue = ks_2samp(dataDir['4'][feature], dataDir['2'][feature])
        featuresRanking.append(ksType(feature, statistic, pValue))

    return sorted(featuresRanking, key=ksType.getStatistic, reverse=True)

def kBestTest(instances, quantityOfFeatures):
    X_clf = []
    y_clf = []
    for instance in instances:

        X_clf.append(instance.getFeatureValues())
        y_clf.append(instance.getCancerType())

    test = SelectKBest(score_func=f_classif, k=quantityOfFeatures)
    fit = test.fit(X_clf, y_clf)
    featuresRanking = []

    for i in range(quantityOfFeatures):
        featuresRanking.append(ksType(i, fit.scores_[i], 0))

    return sorted(featuresRanking, key=ksType.getStatistic, reverse=True)

def reliefFTest(instances, quantityOfFeatures):
    X_clf = []
    y_clf = []

    for instance in instances:
        X_clf.append(instance.getFeatureValues())
        y_clf.append(instance.getCancerType())

    X_clf = numpy.array(X_clf).astype(numpy.int)
    y_clf = numpy.array(y_clf).astype(numpy.int)

    fs = ReliefF(n_neighbors=5)
    fs.fit(X_clf, y_clf)
    rank = fs.top_features
    scores = fs.feature_scores

    featuresRanking = []
    print(rank, scores)
    for i in range(quantityOfFeatures):
        featuresRanking.append(ksType(i, scores[i], 0))

    return sorted(featuresRanking, key=ksType.getStatistic, reverse=True)

def prepareData(data, features):
    finalData = []
    finalDataLabels = []

    for instance in data:
        featureSet = []
        finalDataLabels.append(instance.getCancerClass())

        for feature in features:
            featureSet.append(float(instance.getFeature(feature)))

        finalData.append(featureSet)

    return finalData, finalDataLabels

def splitInstances(instances, features):
    return train_test_split(instances, test_size = 0.5)

def kNNAlgorithm(k, teachingData, testData, features, metric):
    teachingDataSet, teachingDataLabels = prepareData(teachingData, features)
    testDataSet, testDataLabels = prepareData(testData, features)

    classifier = KNeighborsClassifier(n_neighbors = k, metric = metric)
    classifier.fit(teachingDataSet, teachingDataLabels)

    predictions = classifier.predict(testDataSet)
    score = accuracy_score(testDataLabels, predictions)
    matrix = confusion_matrix(testDataLabels, predictions, labels = ['B', 'M'])

    print(score)
    return score, matrix

def naiveBayesAlgorithm(teachingData, testData, features):
    teachingDataSet, teachingDataLabels = prepareData(teachingData, features)
    testDataSet, testDataLabels = prepareData(testData, features)

    classifier = GaussianNB()
    classifier.fit(teachingDataSet, teachingDataLabels)

    predictions = classifier.predict(testDataSet)
    score = accuracy_score(testDataLabels, predictions)
    matrix = confusion_matrix(testDataLabels, predictions, labels = ['B', 'M'])

    print(score)
    return score, matrix

def decisionTreeAlgorithm(teachingData, testData, features):
    teachingDataSet, teachingDataLabels = prepareData(teachingData, features)
    testDataSet, testDataLabels = prepareData(testData, features)

    classifier = DecisionTreeClassifier(max_depth=5)
    classifier.fit(teachingDataSet, teachingDataLabels)

    predictions = classifier.predict(testDataSet)
    score = accuracy_score(testDataLabels, predictions)
    matrix = confusion_matrix(testDataLabels, predictions, labels = ['B', 'M'])

    print(score)
    return score, matrix

def linearSVMAlgorithm(teachingData, testData, features):
    teachingDataSet, teachingDataLabels = prepareData(teachingData, features)
    testDataSet, testDataLabels = prepareData(testData, features)

    classifier = SVC(kernel="linear", C=0.025)
    classifier.fit(teachingDataSet, teachingDataLabels)

    predictions = classifier.predict(testDataSet)
    score = accuracy_score(testDataLabels, predictions)
    matrix = confusion_matrix(testDataLabels, predictions, labels = ['B', 'M'])

    print(score)
    return score, matrix


def RBF_SVM_Algorithm(teachingData, testData, features):
    teachingDataSet, teachingDataLabels = prepareData(teachingData, features)
    testDataSet, testDataLabels = prepareData(testData, features)

    classifier = SVC(gamma=2, C=1)
    classifier.fit(teachingDataSet, teachingDataLabels)

    predictions = classifier.predict(testDataSet)
    score = accuracy_score(testDataLabels, predictions)
    matrix = confusion_matrix(testDataLabels, predictions, labels=['B', 'M'])

    print(score)
    return score, matrix

def crossValidationKNN(kValues, metrics, instances, features):
    scores = {}

    for m in metrics:
        scores[m] = {}

        for k in kValues:
            scores[m][k] = []

            for i in range(0, features.__len__()):
                featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
                data, dataLabels = prepareData(instances, featuresIds)

                knnClassifier = KNeighborsClassifier(n_neighbors = k, metric = m)
                rkf = RepeatedKFold(n_splits = 2, n_repeats = 5)
                score = cross_val_score(estimator = knnClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
                scores[m][k].append(round(mean(score) * 100, 2))

    return scores

def crossValidationNB(instances, features):
    scores = []

    for i in range(0, features.__len__()):
        featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
        data, dataLabels = prepareData(instances, featuresIds)

        naiveBayesClassifier = GaussianNB()
        rkf = RepeatedKFold(n_splits = 2, n_repeats = 5)
        score = cross_val_score(estimator = naiveBayesClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
        scores.append(round(mean(score) * 100, 2))


    return scores

def crossValidationDT(instances, features):
    scores = []

    for i in range(0, features.__len__()):
        featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
        data, dataLabels = prepareData(instances, featuresIds)

        decisionTreeClassifier = DecisionTreeClassifier(max_depth=5)
        rkf = RepeatedKFold(n_splits = 2, n_repeats = 5)
        score = cross_val_score(estimator = decisionTreeClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
        scores.append(round(mean(score) * 100, 2))


    return scores

def crossValidationLinearSVM(instances, features):
    scores = []

    for i in range(0, features.__len__()):
        featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
        data, dataLabels = prepareData(instances, featuresIds)

        linearSVMClassifier = SVC(kernel="linear", C=0.025)
        rkf = RepeatedKFold(n_splits = 2, n_repeats = 5)
        score = cross_val_score(estimator = linearSVMClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
        scores.append(round(mean(score) * 100, 2))


    return scores

def crossValidationRBF_SVM(instances, features):
    scores = []

    for i in range(0, features.__len__()):
        featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
        data, dataLabels = prepareData(instances, featuresIds)

        RBF_SVMClassifier = SVC(gamma=2, C=1)
        rkf = RepeatedKFold(n_splits = 2, n_repeats = 5)
        score = cross_val_score(estimator = RBF_SVMClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
        scores.append(round(mean(score) * 100, 2))


    return scores

main()
