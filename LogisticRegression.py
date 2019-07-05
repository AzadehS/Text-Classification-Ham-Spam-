
"""
Created on Sun Feb 25 17:43:54 2018

@author: AzadehSamadian
"""

import numpy as np
import nltk
import os, sys
import nltk


to_filter = [',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']


def readFile(emailPath):
    print(emailPath)
    corpus= ''
    emails = []
    labels = [] #non-spam = 0, spam = 1
    spamEmailFiles = [x for x in os.listdir(emailPath+'/spam/')] #creates list of email filenames
    for e in spamEmailFiles:
        with open(emailPath+'/spam/' + e, 'r') as email:
            text = email.read().lower().replace('\n', ' ')
            
            text = unicode(text, errors='ignore')
            
            corpus += text.replace('\n', ' ') +'\n'
            for ch in to_filter:
                text = [x.replace(ch, ' ') for x in text]
                text = ''.join(text).split()
            labels.append(1)
            emails.append(text)
    
    hamEmailFiles = [x for x in os.listdir(emailPath+'/ham/')] 
    
    for e in hamEmailFiles:
        with open(emailPath+'/ham/' + e, 'r') as email:
            text = email.read().lower().replace('\n', ' ')
            
            text = unicode(text, errors='ignore')

            corpus += text.replace('\n', ' ') +'\n'
            for ch in to_filter:
                text = [x.replace(ch, ' ') for x in text]
                text = ''.join(text).split()
            labels.append(0)
            emails.append(text)
            
    return corpus, labels


def extractFeatures(corpus, vocabulary):
    
    featureMatrix=[]
    
    for line in corpus.split('\n'):
        if line == '':
            continue
        Feature = [0] * len(vocabulary)
        for word in nltk.tokenize.word_tokenize(line):         
            try:
                Feature[vocabulary.index(word.lower())] +=1
            except Exception:
                continue
        #FeatureNorm = Feature / np.linalg.norm(Feature)
        featureMatrix.append(Feature)
    return featureMatrix 


#Compute the sigmoid function
def sigmoid(X):
    sig = 1.0 / (1.0 + np.exp(-1.0 * X))
    return sig

# Train the model and find the weight
def train(trainFeatureMatrix, label, Lambda, eta, iteration):
    
    x, y = np.asarray(trainFeatureMatrix).shape
    data = np.ones ((x,y+1))
    data[:, 1:] = trainFeatureMatrix
    
    m,n = np.asarray(data).shape
    init_weight = [0] * n
    init_weight[0] = 1
     
    weight, final_cost = gradientAscent(data, label, init_weight, Lambda, eta, iteration)
        
    return weight, final_cost


def weightUpdate(data, label, init_weight, Lambda, eta):
    m,n = np.asarray(data).shape
    error = 0
    error_x = np.zeros((m,n))
    error_sum = [0] * n
    for l in range (m):
        error = label[l] - sigmoid( np.dot(data[l], init_weight) )
        error_x[l] =  np.multiply( data[l], error)
    error_sum = np.sum (error_x, axis=0) 
    weight = np.add( np.multiply( init_weight, -1.0 * eta * Lambda ) , np.multiply( eta , error_sum ))
    newWeight = np.add (weight , init_weight)
    
    return newWeight, np.sum(error_sum)
    
def gradientAscent (data, label, init_weight, Lambda, eta, iteration):
    
    i=0
    flag = False
    weight=[]
    weight.append(init_weight)
    for i in range(iteration):
    #while True:
        #i+=1
        #print (str (i))
        
        init_weight, error_sum = weightUpdate(data, label, init_weight, Lambda, eta)
        weight.append(init_weight)
        
        #print (str (error_sum))
        
        for j in range (len(init_weight)):
            if  np.absolute( weight[-1][j] - weight[-2][j]) > 0.001: 
                flag = False
                break
            else: 
                flag = True
        
        if flag: 
            break
    return init_weight, error_sum




def test(testFeatureMatrix, testLabels, weight):
    
    x, y = np.asarray(testFeatureMatrix).shape
    testData = np.ones ((x,y+1))
    testData[:, 1:] = testFeatureMatrix
    
    errorCount=0
    count=0
    for testVector, TrueLbl in zip(testData, testLabels):
        count +=1
        sig = sigmoid(np.dot(testVector, weight))
        
        if sig >= 0.5 : 
            predictedLbl = 1
        else: 
            predictedLbl = 0
        
        if predictedLbl != TrueLbl:
            errorCount +=1
    
    return 1- ( errorCount / float(count))


if __name__ == "__main__":


    #-------Input : address -------------

    # please put your file's address here
    #address = "/A_Azadeh_UTD/Courses/Spring18/ML/HomeWorks/HW/HW2/Solution/dataSet1"
    address = "/A_Azadeh_UTD/Courses/Spring18/ML/HomeWorks/HW/HW2/Solution/elearning/AzadehSamadian/dataSet1"

    train70EmailPath = address + "/train/train70/"
    print(train70EmailPath)
    validationEmailPath = address + "/train/validation30"
    trainEmailPath = address + "/train"
    testEmailPath = address + "/test"
    """
    
    if len(sys.argv) < 2:
        sys.exit()
    else:

        trainEmailPath = sys.argv[1]
        testEmailPath = sys.argv[2]

    train70EmailPath=trainEmailPath+'train70'
    validationEmailPath = trainEmailPath+'validation30'

    eta = 0.01
    iteration = 80
    """

    #train 70%
    trainCorpus, trainLabels= readFile(train70EmailPath)
    word_features = nltk.FreqDist(w.lower() for w in nltk.tokenize.word_tokenize(trainCorpus))
    vocabulary = word_features.keys()
    train70FeatureMatrix= extractFeatures(trainCorpus, vocabulary)


    # validation part (30% of training data)
    validationCorpus, validationLabels = readFile(validationEmailPath)
    validationFeatureMatrix = extractFeatures(validationCorpus, vocabulary)

    maxAccuracy = 0
    bestLambda = 100
    for Lambda in range(0,10):

        weight, final_cost= train(train70FeatureMatrix, trainLabels, Lambda, eta, iteration)
        accuracy = test(validationFeatureMatrix, validationLabels, weight)
        if accuracy >= maxAccuracy:
            maxAccuracy = accuracy
            bestLambda = Lambda

    print('Best Lambda is: {}.'.format(bestLambda))


    # train all data based on the best Lambda
    trainCorpus, trainLabels = readFile(trainEmailPath)
    word_features = nltk.FreqDist(w.lower() for w in nltk.tokenize.word_tokenize(trainCorpus))
    vocabulary = word_features.keys()
    trainFeatureMatrix = extractFeatures(trainCorpus, vocabulary)
    weight, final_cost = train(trainFeatureMatrix, trainLabels, bestLambda, eta, iteration)

    # test part
    testCorpus, testLabels= readFile(testEmailPath)
    testFeatureMatrix= extractFeatures(testCorpus, vocabulary)

    accuracy = test(testFeatureMatrix, testLabels, weight)

    print('Accuracy: {}.'.format(accuracy))

