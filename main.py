import numpy
from sklearn import clone
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import rankdata, ranksums
from tabulate import tabulate
from src.Instance import Instance
from ReliefF import ReliefF

# Configuration
datasets = ['australian', 'balance', 'breastcan', 'breastcancoimbra', 'cryotherapy',
            'diabetes', 'divorce', 'glass2', 'haberman', 'hayes',
            'heart', 'Immunotherapy', 'iris', 'liver', 'monkone',
            'monkthree', 'sobar-72', 'soybean', 'wine', 'wisconsin']

skipRowsWithInvalidFeature = True   # Skip rows with '?' in features
defaultValueOfInvalidFeature = 1    # Replace '?' with value if skipRowsWithInvalidFeature is False

quantityOfFeatures = 0              # Quantity of features in file
rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=2323)


clfs = {
    'kNN' : KNeighborsClassifier(n_neighbors = 5, metric = 'manhattan'),
    'GNB': GaussianNB(),
    'CART': DecisionTreeClassifier(max_depth=5),
    'SVC' : SVC(kernel="linear", C=0.025),
    'RBF_SVMC' : SVC(gamma=2, C=1),
}
scores = numpy.zeros((len(clfs), datasets.__len__(), 2 * 5))


def main():
    for data_id, datafile in enumerate(datasets):
        instances = loadDataFromFile(skipRowsWithInvalidFeature = True, defaultValueOfInvalidFeature = 1, fileNumber=data_id)
        selectPercentileRanking = selectPercentileTest(instances)
        rankingKBest = kBestANOVATest(instances, quantityOfFeatures)
        rankingReliefF = reliefFTest(instances)

        # scores = classification(instances, rankingKBest)
        scores = classification(instances, selectPercentileRanking)
        # scores = classification(instances, rankingReliefF)
    #
    numpy.save('kresults', scores)
    scores = numpy.load('kresults.npy')
    print(scores.shape)
    mean_scores = numpy.mean(scores, axis=2).T
    print("\nMean scores:\n", mean_scores)

    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = numpy.array(ranks)
    print("\nRanks:\n", ranks)

    mean_ranks = numpy.mean(ranks, axis=0)
    print("\nMean ranks:\n", mean_ranks)

    alfa = .05
    w_statistic = numpy.zeros((len(clfs), len(clfs)))
    p_value = numpy.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])


    headers = list(clfs.keys())
    names_column = numpy.expand_dims(numpy.array(list(clfs.keys())), axis=1)
    w_statistic_table = numpy.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = numpy.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = numpy.zeros((len(clfs), len(clfs)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(numpy.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    significance = numpy.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(numpy.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)


def loadDataFromFile(skipRowsWithInvalidFeature = False, defaultValueOfInvalidFeature = 1, fileNumber=0):
    instances = []

    fileName = 'datasets/'+datasets[fileNumber]+'.csv'
    print('CREATING FOR :' + fileName)
    file = open(fileName, 'r').read()
    lines = file.split('\n')

    metadata = lines[0].split(',')

    lineLen = int(metadata[0])
    columnCancerClass = int(metadata[1])    # Index of column with cancer class
    columnFirstFeature = int(metadata[2])   # Index of column with first feature
    global quantityOfFeatures
    quantityOfFeatures = int(metadata[3])   # Quantity of features in file
    amountOfClasses = int(metadata[4])

    for line in lines[1:]:
        if '?' in line and skipRowsWithInvalidFeature is True:
            continue

        row = line.split(',')

        if row.__len__() == lineLen:
            instance = Instance(line, row[columnFirstFeature:quantityOfFeatures+1], row[columnCancerClass], defaultValueOfInvalidFeature)
            instances.append(instance)

    return instances

def kBestANOVATest(instances, quantityOfFeatures):      #Selector  kBestANOVA
    X_clf = []
    y_clf = []
    for instance in instances:

        X_clf.append(instance.getFeatureValues())
        y_clf.append(instance.getCancerType())

    X_clf = numpy.array(X_clf).astype(numpy.longdouble)
    y_clf = numpy.array(y_clf).astype(numpy.longdouble)

    test = SelectKBest(score_func=f_classif, k=quantityOfFeatures)
    return test.fit_transform(X_clf, y_clf)

def selectPercentileTest(instances):                    #Selector PercentileChi2
    X_clf = []
    y_clf = []
    for instance in instances:

        X_clf.append(instance.getFeatureValues())
        y_clf.append(instance.getCancerType())

    X_clf = numpy.array(X_clf).astype(numpy.longdouble)
    y_clf = numpy.array(y_clf).astype(numpy.longdouble)

    test = SelectPercentile(score_func=chi2, percentile=30)

    return test.fit_transform(X_clf, y_clf)

def reliefFTest(instances):                              #Selector reliefF
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


main()

