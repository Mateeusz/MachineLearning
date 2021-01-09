import pandas as pd

import numpy
from scipy.stats import ks_2samp
from sklearn.feature_selection import SelectKBest, f_classif, chi2


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

#
datafiles = []
    # , 'breast-cancer-wisconsin.data', 'dataR2.csv', 'dataset_37_diabetes.csv', 'divorce.csv', 'glass.data', 'haberman.data', 'Immunotherapy.data', 'iris.data',
    #          'lung-cancer.data','mammographic_masses.data', 'messidor_features.arff', 'php4fATLZ.csv', 'pop_failures.dat', 'sobar-72.csv',
#          'sonar.all-data', 'SPECT.test', 'SPECT.train',
    #          'SPECTF.test', 'SPECTF.train', 'wine.data']

# Configuration
dataFile = 'resources/breast-cancer-wisconsin.data'

skipRowsWithInvalidFeature = True   # Skip rows with '?' in features
defaultValueOfInvalidFeature = 1    # Replace '?' with value if skipRowsWithInvalidFeature is False

quantityOfFeatures = 0  # Quantity of features in file
rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=2323)



def main(fileNumber):
    instances = loadDataFromFile(skipRowsWithInvalidFeature = True, defaultValueOfInvalidFeature = 1, fileNumber=fileNumber)
    rankingKolmogorov = kBestchi2Test(instances, quantityOfFeatures)
    rankingKolmogorovIds = [x.getParamID() for x in rankingKolmogorov]

    rankingKBest = kBestANOVATest(instances, quantityOfFeatures)
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


    scoresKNN = crossValidationKNN(instances, rankingKBest)
    scoresNB = crossValidationNB(instances, rankingKBest)
    scoresDT = crossValidationDT(instances, rankingKBest)
    scoresLSVC = crossValidationLinearSVM(instances, rankingKBest)
    scoresRBF = crossValidationRBF_SVM(instances, rankingKBest)

    kscoresKNN = crossValidationKNN(instances, rankingKolmogorov)
    kscoresNB = crossValidationNB(instances, rankingKolmogorov)
    kscoresDT = crossValidationDT(instances, rankingKolmogorov)
    kscoresLSVC = crossValidationLinearSVM(instances, rankingKolmogorov)
    kscoresRBF = crossValidationRBF_SVM(instances, rankingKolmogorov)

    rscoresKNN = crossValidationKNN(instances, rankingReliefF)
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

def loadDataFromFile(skipRowsWithInvalidFeature = False, defaultValueOfInvalidFeature = 1, fileNumber=0):
    instances = []

    fileName = 'resources/'+datafiles[fileNumber]
    print('CREATING FOR :' + fileName)
    file = open(fileName, 'r').read()
    lines = file.split('\n')

    metadata = lines[0].split(',')

    lineLen = int(metadata[0])
    columnCancerClass = int(metadata[1])  # Index of column with cancer class
    columnFirstFeature = int(metadata[2])  # Index of column with first feature
    global quantityOfFeatures
    quantityOfFeatures = int(metadata[3])  # Quantity of features in file
    amountOfClasses = int(metadata[4])

    for line in lines[1:]:
        if '?' in line and skipRowsWithInvalidFeature is True:
            continue

        row = line.split(',')

        if row.__len__() == lineLen:
            instance = Instance(line, row[columnFirstFeature:quantityOfFeatures+1], row[columnCancerClass], defaultValueOfInvalidFeature)
            instances.append(instance)

    return instances

def kBestANOVATest(instances, quantityOfFeatures):
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

def kBestchi2Test(instances, quantityOfFeatures):
    X_clf = []
    y_clf = []
    for instance in instances:

        X_clf.append(instance.getFeatureValues())
        y_clf.append(instance.getCancerType())

    X_clf = numpy.array(X_clf).astype(numpy.float)
    y_clf = numpy.array(y_clf).astype(numpy.float)

    test = SelectKBest(score_func=chi2, k=quantityOfFeatures)
    print(X_clf)
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

    X_clf = numpy.array(X_clf).astype(numpy.float)
    y_clf = numpy.array(y_clf).astype(numpy.float)

    fs = ReliefF(n_neighbors=5)
    fs.fit(X_clf, y_clf)
    rank = fs.top_features
    scores = fs.feature_scores

    featuresRanking = []
    for i in range(quantityOfFeatures):
        featuresRanking.append(ksType(i, scores[i], 0))

    return sorted(featuresRanking, key=ksType.getStatistic, reverse=True)

def prepareData(data, features):
    finalData = []
    finalDataLabels = []

    for instance in data:
        featureSet = []
        finalDataLabels.append(instance.getCancerType())

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

def crossValidationKNN(instances, features):
    scores = []

    for i in range(0, features.__len__()):
        featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
        data, dataLabels = prepareData(instances, featuresIds)

        knnClassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'manhattan')
        score = cross_val_score(estimator = knnClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
        scores.append(round(mean(score) * 100, 2))

    return scores

def crossValidationNB(instances, features):
    scores = []

    for i in range(0, features.__len__()):
        featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
        data, dataLabels = prepareData(instances, featuresIds)

        naiveBayesClassifier = GaussianNB()
        score = cross_val_score(estimator = naiveBayesClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
        scores.append(round(mean(score) * 100, 2))


    return scores

def crossValidationDT(instances, features):
    scores = []

    for i in range(0, features.__len__()):
        featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
        data, dataLabels = prepareData(instances, featuresIds)

        decisionTreeClassifier = DecisionTreeClassifier(max_depth=5)
        score = cross_val_score(estimator = decisionTreeClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
        scores.append(round(mean(score) * 100, 2))


    return scores

def crossValidationLinearSVM(instances, features):
    scores = []

    for i in range(0, features.__len__()):
        featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
        data, dataLabels = prepareData(instances, featuresIds)

        linearSVMClassifier = SVC(kernel="linear", C=0.025)
        score = cross_val_score(estimator = linearSVMClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
        scores.append(round(mean(score) * 100, 2))


    return scores

def crossValidationRBF_SVM(instances, features):
    scores = []

    for i in range(0, features.__len__()):
        featuresIds = [feature.getParamID() for feature in features[0:i + 1]]
        data, dataLabels = prepareData(instances, featuresIds)

        RBF_SVMClassifier = SVC(gamma=2, C=1)
        score = cross_val_score(estimator = RBF_SVMClassifier, X = data, y = dataLabels, scoring = 'accuracy', cv = rkf)
        scores.append(round(mean(score) * 100, 2))


    return scores


for fileNumber in range(0, datafiles.__len__()):
    main(fileNumber)


# TODO:
# nie ma stanu losowego, repeated key w foldzie wszedzie random ale taki sam
# https://metsi.github.io/
# brak merytoryki, brak danych konkretnych
