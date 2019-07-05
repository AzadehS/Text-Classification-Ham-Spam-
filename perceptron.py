
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:43:54 2018

@author: Azadeh
"""

import numpy as np
import nltk
import os, sys
from numpy import array, dot, random



to_filter = [',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']

def readFile( emailPath):
    
    corpus= ''
    emails = []
    labels = [] #non-spam = 0 and spam = 1
    #print("hello!!")
    #print(emailPath+'/spam/')

    spamEmailFiles = [x for x in os.listdir(emailPath+'/spam/')] #creates list of email filenames
    
    for e in spamEmailFiles:
        with open(emailPath+'/spam/' + e, 'r') as email:
            text = email.read().lower().replace('\n', ' ')
            #print(emailPath + '/spam/')

            text = unicode(text, errors='ignore')
            #text = str(text, errors='ignore')

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
    
    featureMatrix = []
    
    for line in corpus.split('\n'):
        if line == '':
            continue
        Feature = [0] * len(vocabulary)
        for word in nltk.tokenize.word_tokenize(line):         
            try:
                Feature[vocabulary.index(word.lower())] +=1
            except Exception:
                continue
        featureMatrix.append(Feature)
    
    return featureMatrix 

def perceptron(trainFeatureMatrix, trainLabels, eta, iteration):
    
    unit_step = lambda x: 0 if x < 0 else 1
    
    x, y = np.asarray(trainFeatureMatrix).shape
    
    trainData = np.ones ((x,y+1))
    trainData[:, 1:] = trainFeatureMatrix
    
    weight = random.rand(y+1) 
    
    for i in xrange(iteration):
        for trainVector, TrueLbl in zip(trainData, trainLabels):
            result = dot(weight, trainVector)
            error = TrueLbl - unit_step(result)
            weight += eta * error * trainVector
    
    return weight


def test(testFeatureMatrix, testLabels, weight):
    
    unit_step = lambda x: 0 if x < 0 else 1
    
    x, y = np.asarray(testFeatureMatrix).shape
    testData = np.ones ((x,y+1))
    testData[:, 1:] = testFeatureMatrix
    
    errorCount=0
    count=0
    for testVector, TrueLbl in zip(testData, testLabels):
        count +=1
        
        result = dot(testVector, weight)
        predictedLbl= unit_step(result)
        
        if predictedLbl != TrueLbl:
            errorCount +=1
    
    return 1- ( errorCount / float(count))



if __name__ == "__main__":



    # -------Input : address ------------- 
    eta = 0.01

    # please put your file's address here
    address = "/A_Azadeh_UTD/Courses/Spring18/ML/HomeWorks/HW/HW2/Solution/dataSet1"

    train70EmailPath = address + "/train/train70"
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

    """




    #train70
    trainCorpus, trainLabels= readFile(train70EmailPath)
    word_features = nltk.FreqDist(w.lower() for w in nltk.tokenize.word_tokenize(trainCorpus))
    vocabulary = word_features.keys()
    trainFeatureMatrix = extractFeatures(trainCorpus, vocabulary)


    #validatio30
    validationCorpus, validationLabels = readFile(validationEmailPath)
    validFeatureMatrix = extractFeatures(validationCorpus, vocabulary)


    unit_step = lambda x: 0 if x < 0 else 1

    x, y = np.asarray(trainFeatureMatrix).shape

    trainData = np.ones((x, y + 1))
    trainData[:, 1:] = trainFeatureMatrix

    weight = random.rand(y + 1)

    maxAccuracy = 0
    bestIteration = -100

    for i in range(0,200):
        for trainVector, TrueLbl in zip(trainData, trainLabels):
            result = dot(weight, trainVector)
            error = TrueLbl - unit_step(result)
            weight += eta * error * trainVector
        accuracy = test(validFeatureMatrix, validationLabels, weight)
        if accuracy >= maxAccuracy:
            maxAccuracy = accuracy
            bestIteration = i

    print('Best Iteration: {}.'.format(bestIteration))

    iteration = bestIteration


    # train all data
    trainCorpus, trainLabels = readFile(trainEmailPath)
    word_features = nltk.FreqDist(w.lower() for w in nltk.tokenize.word_tokenize(trainCorpus))
    vocabulary = word_features.keys()
    trainFeatureMatrix = extractFeatures(trainCorpus, vocabulary)

    # test part
    testCorpus, testLabels = readFile(testEmailPath)
    testFeatureMatrix = extractFeatures(testCorpus, vocabulary)

    weight = perceptron(trainFeatureMatrix, trainLabels, eta, iteration)
    accuracy = test(testFeatureMatrix, testLabels, weight)

    print('Accuracy: {}.'.format(accuracy))

