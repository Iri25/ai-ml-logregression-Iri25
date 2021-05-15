import matplotlib.pyplot as plt


# ------------------------------------------------- Plot Brest Cancer -------------------------------------------------

def plotDataDistributionBreastCancer(inputs, outputs, outputNames, feature1, feature2):
    labels = set(outputs)
    noData = len(inputs)
    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if outputs[i] == crtLabel]
        y = [feature2[i] for i in range(noData) if outputs[i] == crtLabel]
        plt.scatter(x, y, label=outputNames[crtLabel])
    plt.xlabel('mean radius')
    plt.ylabel('mean texture')
    plt.legend()
    plt.show()


def plotDataHistogramBreastCancer(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def plotClassificationDataBrestCancer(feature1, feature2, outputs, outputNames, title=None):
    labels = set(outputs)
    noData = len(feature1)

    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if outputs[i] == crtLabel]
        y = [feature2[i] for i in range(noData) if outputs[i] == crtLabel]
        plt.scatter(x, y, label=outputNames[crtLabel])

    plt.xlabel('mean radius')
    plt.ylabel('mean texture')
    plt.legend()
    plt.title(title)
    plt.show()


def plotPredictionsBrestCancer(feature1, feature2, outputs, realOutputs, computedOutputs, title, labelNames):
    labels = list(set(outputs))
    noData = len(feature1)

    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] == crtLabel]
        y = [feature2[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] == crtLabel]
        plt.scatter(x, y, label=labelNames[crtLabel] + ' (correct)')

    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] != crtLabel]
        y = [feature2[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] != crtLabel]
        plt.scatter(x, y, label=labelNames[crtLabel] + ' (incorrect)')

    plt.xlabel('mean radius')
    plt.ylabel('mean texture')
    plt.legend()
    plt.title(title)
    plt.show()


# ------------------------------------------------- Plot Iris Flowers -------------------------------------------------

def plotTrainIrisFlowers(trainInputs, trainOutputs):
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.title("normalised train data")
    for i in range(len(trainOutputs)):
        if trainOutputs[i] == "Iris-setosa":
            plt.plot([trainInputs[i][0]], [trainInputs[i][1]], "ro")
        if trainOutputs[i] == "Iris-versicolor":
            plt.plot([trainInputs[i][0]], [trainInputs[i][1]], "go")
        if trainOutputs[i] == "Iris-virginica":
            plt.plot([trainInputs[i][0]], [trainInputs[i][1]], "bo")
    plt.show()


# max((RLForOne, RLForTwo, RLForThree).predictOneSampleValue) de TestOutputs[i] => correct prediction
def plotTestIrisFlowers(testInputs, testOutputs, RLForOne, RLForTwo, RLForThree):
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.title("normalised test data")
    for i in range(len(testOutputs)):

        firstPrediction = RLForOne.predictOneSampleValue(testInputs[i])
        secondPrediction = RLForTwo.predictOneSampleValue(testInputs[i])
        thirdPrediction = RLForThree.predictOneSampleValue(testInputs[i])

        if testOutputs[i] == "Iris-setosa":
            if firstPrediction > secondPrediction and firstPrediction > thirdPrediction:
                plt.plot([testInputs[i][0]], [testInputs[i][1]], "r*")
            else:
                plt.plot([testInputs[i][0]], [testInputs[i][1]], "ro")

        if testOutputs[i] == "Iris-versicolor":
            if secondPrediction > firstPrediction and secondPrediction > thirdPrediction:
                plt.plot([testInputs[i][0]], [testInputs[i][1]], "g*")
            else:
                plt.plot([testInputs[i][0]], [testInputs[i][1]], "go")

        if testOutputs[i] == "Iris-virginica":
            if thirdPrediction > secondPrediction and thirdPrediction > firstPrediction:
                plt.plot([testInputs[i][0]], [testInputs[i][1]], "b*")
            else:
                plt.plot([testInputs[i][0]], [testInputs[i][1]], "bo")
    plt.show()
