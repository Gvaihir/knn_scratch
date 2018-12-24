# Import module

import pandas as pd
irisIn = pd.read_csv('/Users/antonogorodnikov/Documents/Work/Python/knn_scratch/iris.data.txt', header=None)

trainSet = irisIn.sample(frac=0.66, axis=0)
testSet = irisIn[~irisIn.index.isin(trainSet.index)]
trainSet.reset_index(drop=True, inplace=True)
testSet.reset_index(drop=True, inplace=True)


print('Train set: {}\n'
      'Test set: {}'.format(len(trainSet),
                            len(testSet)))


# distance measurement
import math

# use euclidean distance for distance measurement
def euclidDist(sample1, sample2, features):
    distance = 0
    for i in range(features):
        distance += (sample1[i] - sample2[i])**2
    return math.sqrt(distance)


# getting neighbors

def kNearestNeighbors(testSample, trainingSet, k):
    distances = []
    length = len(testSample) - 1
    for i in range(len(trainingSet)):
        dist = euclidDist(testSample, trainingSet.iloc[i], features=length)
        distances.append(trainingSet.loc[i].append(pd.Series(dist, index=['dist'])))
        distDF = pd.DataFrame(distances)
        distDF.sort_values(by='dist', inplace=True)
        neighbors = distDF.iloc[0:k]
    classVotes = pd.DataFrame({'class': irisIn[4].unique(), 'vote': [0] * len(irisIn[4].unique())})
    for i in range(len(neighbors)):
        response = neighbors.iloc[i][-2]
        classVotes.loc[classVotes['class'] == response, 'vote'] += 1
    responseDF = classVotes.loc[classVotes['vote'] == classVotes['vote'].max(), 'class']
    response = responseDF.iloc[0]
    return response


# true positive rate

def tpr(testSet):
    correct = 0
    for i in range(len(testSet)):
        if testSet.iloc[i,4] is testSet.loc[i,'prediction']:
            correct += 1
    return correct/float(len(testSet))




# deploy
def main():
    testSet['prediction'] = testSet.apply(kNearestNeighbors, trainingSet=trainSet, k=5, axis=1)


tpr(testSet)

print(timeit.timeit(lambda: main(),number=1))

