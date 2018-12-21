# Import module

import pandas as pd
irisIn = pd.read_csv('/Users/antonogorodnikov/Documents/Work/Python/knn_scratch/iris.data.txt', header=None)

trainSet = irisIn.sample(frac=0.66, axis=0).reset_index(drop=True)
testSet = irisIn[~irisIn.index.isin(trainSet.index)].reset_index(drop=True)

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

def kNearesNeighbors(trainingSet, testSample, k):
    distances = []
    length = len(testSample) - 1
    for i in range(len(trainingSet)):
        dist = euclidDist(testSample, trainingSet.iloc[i], features=length)
        distances.append(trainingSet.loc[i].append(pd.Series(dist, index=['dist'])))
        distDF = pd.DataFrame(distances)
        distDF.sort_values(1, inplace=True)
    return distDF.iloc[0:k]


# measuring response

def getResponse(neighbors):
    classVotes = pd.DataFrame({'class': irisIn[4].unique(), 'vote': [0]*len(irisIn[4].unique())})
    for i in range(len(neighbors)):
        response = neighbors.iloc[i][-2]
        classVotes.loc[classVotes['class'] == response, 'vote'] += 1
    responseDF = classVotes.loc[classVotes['vote'] == classVotes['vote'].max(), 'class']
    response = responseDF.iloc[0]
    return response





