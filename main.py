import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from statistics import mean, stdev

from Regression.logisticRegression import LogisticRegression

from Regression.utils import plotDataDistributionBreastCancer, plotDataHistogramBreastCancer, \
    plotClassificationDataBrestCancer, plotPredictionsBrestCancer, plotTrainIrisFlowers, plotTestIrisFlowers

from Regression.readToFile import readDataIrisFlowers


# ---------------------------------------------------- Brest Cancer ----------------------------------------------------

def normalisation(trainData, testData):
    scalar = StandardScaler()

    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        # fit only on training data
        scalar.fit(trainData)

        # apply same transformation to train data
        normalisedTrainData = scalar.transform(trainData)

        # apply same transformation to test data
        normalisedTestData = scalar.transform(testData)

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        # fit only on training data
        scalar.fit(trainData)

        # apply same transformation to train data
        normalisedTrainData = scalar.transform(trainData)

        # apply same transformation to test data
        normalisedTestData = scalar.transform(testData)

    return normalisedTrainData, normalisedTestData


def runBreastCancer():
    data = load_breast_cancer()

    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']

    featureNames = list(data['feature_names'])
    feature1 = [feat[featureNames.index('mean radius')] for feat in inputs]
    feature2 = [feat[featureNames.index('mean texture')] for feat in inputs]

    inputs = [[feat[featureNames.index('mean radius')], feat[featureNames.index('mean texture')]] for feat in inputs]

    # plot the data distribution breast cancer
    # plotDataDistributionBreastCancer(inputs, outputs, outputNames, feature1, feature2)

    # plot the data histogram breast cancer
    # plotDataHistogramBreastCancer(feature1, 'mean radius')
    # plotDataHistogramBreastCancer(feature2, 'mean texture')
    # plotDataHistogramBreastCancer(outputs, 'cancer class')

    # plot the classification data breast cancer
    # plotClassificationDataBrestCancer(feature1, feature2, outputs, outputNames, None)

    # split data into train and test subsets
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]

    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # normalise the features
    trainInputs, testInputs = normalisation(trainInputs, testInputs)

    feature1train = [ex[0] for ex in trainInputs]
    feature2train = [ex[1] for ex in trainInputs]
    feature1test = [ex[0] for ex in testInputs]
    feature2test = [ex[1] for ex in testInputs]

    # plot the normalised data brest cancer
    # plotClassificationDataBrestCancer(feature1train, feature2train, trainOutputs, outputNames,
    # 'normalised train data')

    # using sklearn (variant 1)
    # classifier = linear_model.LogisticRegression()

    # using developed code (variant 2)
    classifier = LogisticRegression()

    # train the classifier (fit in on the training data)
    classifier.fit(trainInputs, trainOutputs)

    # parameters of the linear repressor
    w0, w1, w2 = classifier.intercept_, classifier.coefficient_[0], classifier.coefficient_[1]
    print('\nClassification model brest cancer: y(feat1, feat2) = ', w0, ' + ', w1, ' * feat1 + ', w2, ' * feat2')

    # makes predictions for test data (variant 1)
    # computedTestOutputs = [w0 + w1 * el[0] + w2 * el[1] for el in testInputs]

    # makes predictions for test data (by tool) (variant 2)
    computedTestOutputs = classifier.prediction(testInputs)

    # plot the predictions brest cancer
    # plotPredictionsBrestCancer(feature1test, feature2test, outputs, testOutputs, computedTestOutputs,
    #                            "real test data", outputNames)

    # compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        if t1 != t2:
            error += 1
    error = error / len(testOutputs)
    print("Classification error (manual) brest cancer: ", error)

    error = 1 - accuracy_score(testOutputs, computedTestOutputs)
    print("Classification error (tool) brest cancer: ", error)


runBreastCancer()


# ---------------------------------------------------- Iris Flowers ----------------------------------------------------

# statistical normalisation (centered around mend and standardisation)
def statisticalNormalisation(features):
    # meanValue = sum(features) / len(features)
    meanValue = mean(features)

    # stdDevValue = (1 / len(features) * sum([ (feat - meanValue) ** 2 for feat in features])) ** 0.5
    stdDevValue = stdev(features)

    normalisedFeatures = [(feat - meanValue) / stdDevValue for feat in features]

    return meanValue, stdDevValue, normalisedFeatures


def normaliseTestData(features, meanValue, stdDevValue):
    return [(feat - meanValue) / stdDevValue for feat in features]


def computeError(computedOutputs, testOutputs):
    error = 0
    for t1, t2 in zip(computedOutputs, testOutputs):
        error += (t1 - t2) ** 2
    return error / len(testOutputs)


def runIrisFlowers():
    nonShuffledInputs, nonShuffledOutputs = readDataIrisFlowers()
    indexes = random.sample(range(len(nonShuffledInputs)), len(nonShuffledInputs))

    inputs = [nonShuffledInputs[i] for i in indexes]
    outputs = [nonShuffledOutputs[i] for i in indexes]

    count = 0
    for i in inputs:
        count = count + 1

    indexes2 = [i for i in range(len([inputs]))]

    trainSample = np.random.choice(2, int(0.8 * count), replace=True)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # using developed code (variant 2)
    classifier = LogisticRegression()

    # train the classifier (fit in on the training data)
    # classifier.fit(trainInputs, trainOutputs)

    # parameters of the linear repressor
    # w0, w1, w2 = classifier.intercept_, classifier.coefficient_[0], classifier.coefficient_[1]
    # print('\nClassification model iris flowers: y(feat1, feat2) = ', w0, ' + ', w1, ' * feat1 + ', w2, ' * feat2')

    # makes predictions for test data (by tool) (variant 2)
    # computedTestOutputs = classifier.prediction(testInputs)

    # error = 0.0
    # for t1, t2 in zip(computedTestOutputs, testOutputs):
    #     if t1 != t2:
    #         error += 1
    # error = error / len(testOutputs)
    # print("Classification error (manual) iris flowers: ", error)

    # error = 1 - accuracy_score(testOutputs, computedTestOutputs)
    # print("Classification error (tool) iris flowers : ", error)

    trainInputs = []
    trainOutputs = []

    testInputs = []
    testOutputs = []

    RLForOne = None
    RLForTwo = None
    RLForThree = None

    sum = 99999
    start = 0
    finish = 0

    # cross with validation
    for i in range(5):

        finish = start + len(outputs) / 5

        nonNormalisedTestInputs = inputs[int(start):int(finish)]
        nonNormalisedTrainInputs = inputs[:int(start)] + inputs[int(finish):]

        testOutputs = outputs[int(start):int(finish)]
        trainOutputs = outputs[:int(start)] + outputs[int(finish):]

        mean1, stDev1, features1 = statisticalNormalisation([el[0] for el in nonNormalisedTrainInputs])
        mean2, stDev2, features2 = statisticalNormalisation([el[1] for el in nonNormalisedTrainInputs])
        trainInputs = [[feat1, feat2] for feat1, feat2 in zip(features1, features2)]

        testFeatures1 = normaliseTestData([el[0] for el in nonNormalisedTestInputs], mean1, stDev1)
        testFeatures2 = normaliseTestData([el[1] for el in nonNormalisedTestInputs], mean2, stDev2)
        testInputs = [[feat1, feat2] for feat1, feat2 in zip(testFeatures1, testFeatures2)]

        regressionOne = LogisticRegression()
        regressionTwo = LogisticRegression()
        regressionThree = LogisticRegression()

        trainOutputsForOne = [1 if el == "Iris-setosa" else 0 for el in trainOutputs]
        trainOutputsForTwo = [1 if el == "Iris-versicolor" else 0 for el in trainOutputs]
        trainOutputsForThree = [1 if el == "Iris-virginica" else 0 for el in trainOutputs]

        testOutputsForOne = [1 if el == "Iris-setosa" else 0 for el in testOutputs]
        testOutputsForTwo = [1 if el == "Iris-versicolor" else 0 for el in testOutputs]
        testOutputsForThree = [1 if el == "Iris-virginica" else 0 for el in testOutputs]

        regressionOne.fit(trainInputs, trainOutputsForOne)
        regressionTwo.fit(trainInputs, trainOutputsForTwo)
        regressionThree.fit(trainInputs, trainOutputsForThree)

        computedOutputsForOne = regressionOne.prediction(testInputs)
        computedOutputsForTwo = regressionTwo.prediction(testInputs)
        computedOutputsForThree = regressionThree.prediction(testInputs)

        errorForOne = computeError(computedOutputsForOne, testOutputsForOne)
        errorForTwo = computeError(computedOutputsForTwo, testOutputsForTwo)
        errorForThree = computeError(computedOutputsForThree, testOutputsForThree)

        if sum > errorForOne + errorForTwo + errorForThree:
            sum = errorForOne + errorForTwo + errorForThree
            TrainInputs = trainInputs
            TestInputs = testInputs
            TrainOutputs = trainOutputs
            TestOutputs = testOutputs
            RLForOne = regressionOne
            RLForTwo = regressionTwo
            RLForThree = regressionThree

        start = finish

    # plot the train iris flowers
    # plotTrainIrisFlowers(TrainInputs, TrainOutputs)

    # plot the test iris flowers
    # plotTestIrisFlowers(TestInputs, TestOutputs, RLForOne, RLForTwo, RLForThree)


runIrisFlowers()
