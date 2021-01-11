import pandas as pd
from tabulate import tabulate
import numpy
from scipy.stats import ks_2samp
from sklearn import clone
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, chi2
from scipy.stats import ttest_ind
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
datafiles = ['dataR2.csv']
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

clfs = {
    'kNN' : KNeighborsClassifier(n_neighbors = 5, metric = 'manhattan'),
    'GNB': GaussianNB(),
    'CART': DecisionTreeClassifier(max_depth=5),
    'SVC' : SVC(kernel="linear", C=0.025),
    'RBF_SVMC' : SVC(gamma=2, C=1),
}


def main(fileNumber):
    instances = loadDataFromFile(skipRowsWithInvalidFeature = True, defaultValueOfInvalidFeature = 1, fileNumber=fileNumber)
    rankingKolmogorov = kBestchi2Test(instances)
    rankingKBest = kBestANOVATest(instances, quantityOfFeatures)
    rankingReliefF = reliefFTest(instances)

    scores = classification(instances, rankingKBest)
    kscores = classification(instances, rankingKolmogorov)
    rscores = classification(instances, rankingReliefF)

    # print(scores)
    # print(kscores)
    # print(rscores)
    finalScores = [scores, kscores, rscores]
    finalScores = numpy.array(finalScores)

    numpy.save('results', finalScores)
    finalScores = numpy.load('results.npy')
    # print("Folds:\n", finalScores)

    alfa = .05
    selectors = ['kBest', 'perCentil', 'ReleifF']
    for k in range(selectors.__len__()):
        t_statistic = numpy.ones((len(clfs), len(clfs)))
        p_value = numpy.ones((len(clfs), len(clfs)))
        for i in range(len(clfs)):
            for j in range(len(clfs)):
                t_statistic[i, j], p_value[i, j] = ttest_ind(finalScores[k][i], finalScores[k][j], equal_var=False, nan_policy='omit')

        print('=======', selectors[k], '=======')
        print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

        headers = ['kNN', 'GNB', 'CART', 'SVC', 'RBF_SVMC']
        names_column = numpy.array([['kNN'], ['GNB'], ['CART'], ['SVC'], ['RBF_SVMC']])
        t_statistic_table = numpy.concatenate((names_column, t_statistic), axis=1)
        t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
        p_value_table = numpy.concatenate((names_column, p_value), axis=1)
        p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
        print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)



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

    X_clf = numpy.array(X_clf).astype(numpy.longdouble)
    y_clf = numpy.array(y_clf).astype(numpy.longdouble)

    test = SelectKBest(score_func=f_classif, k=quantityOfFeatures)
    return test.fit_transform(X_clf, y_clf)

def kBestchi2Test(instances):
    X_clf = []
    y_clf = []
    for instance in instances:

        X_clf.append(instance.getFeatureValues())
        y_clf.append(instance.getCancerType())

    X_clf = numpy.array(X_clf).astype(numpy.longdouble)
    y_clf = numpy.array(y_clf).astype(numpy.longdouble)

    test = SelectPercentile(score_func=chi2, percentile=30)

    return test.fit_transform(X_clf, y_clf)

def reliefFTest(instances):
    X_clf = []
    y_clf = []

    for instance in instances:
        X_clf.append(instance.getFeatureValues())
        y_clf.append(instance.getCancerType())

    X_clf = numpy.array(X_clf).astype(numpy.longdouble)
    y_clf = numpy.array(y_clf).astype(numpy.longdouble)

    fs = ReliefF(n_neighbors=5)
    return fs.fit_transform(X_clf, y_clf)


def classification(instances, features):
    scores = numpy.zeros((len(clfs), 5 * 2))
    X_clf = features
    y_clf = []

    for instance in instances:
        y_clf.append(instance.getCancerType())

    X_clf = numpy.array(X_clf).astype(numpy.longdouble)
    y_clf = numpy.array(y_clf).astype(numpy.longdouble)

    for fold_id, (train, test) in enumerate(rkf.split(X_clf, y_clf)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X_clf[train], y_clf[train])
            y_pred = clf.predict(X_clf[test])
            scores[clf_id, fold_id] = accuracy_score(y_clf[test], y_pred)

    return scores


for fileNumber in range(0, datafiles.__len__()):
    main(fileNumber)


# TODO:
# nie ma stanu losowego, repeated key w foldzie wszedzie random ale taki sam
# https://metsi.github.io/
# brak merytoryki, brak danych konkretnych
