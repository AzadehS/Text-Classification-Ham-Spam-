
"""
Created on Sun Feb 25 17:43:54 2018

@author: Azadeh
"""

import numpy as np
import nltk
import os, sys
from math import log
from itertools import *

to_filter = [',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']

def readFile( emailPath):
    
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
            labels.append('spam')
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
            labels.append('ham')
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
        featureMatrix.append(Feature)
    
    return featureMatrix 


def selectFeatures(separated):
    
    N=[len(separated['ham']),len(separated['spam'])]
    Nall= N[0] + N[1]
    A=[0,0]
    B=[0,0]
    C=[0,0]
    D=[0,0]
    IG = [0] * len(separated['ham'][0])
    
    for col in range(len(separated['ham'][0])):
        for cl in separated:
            
            if (cl =='ham'): clnum = 0
            else: clnum=1
            
            A[clnum]= np.sum (separated[cl], axis=0)[col]
            
            for row in (separated[cl]):
                if (row[col]==0):
                    C[clnum] +=1
        
        B[0], B[1]= A[1], A[0]
        D[0], D[1] = C[1] , C[0]
        
        IG[col] = -1.0 * (N[0] / float(Nall)) * np.log(N[0] / float(Nall)) -1.0 * (N[1] / float(Nall)) * np.log(N[1] / float(Nall)) + \
          ( (float(A[0])/Nall) + (float(A[1])/Nall) ) * ( (A[0] / float(A[0]+B[0])) * np.log(A[0] / float(A[0]+B[0])) + (A[1] / float(A[1]+B[1])) * np.log(A[1] / float(A[1]+B[1])) ) + \
          ( (float(C[0])/Nall) + (float(C[1])/Nall) ) * ( (C[0] / float(C[0]+D[0])) * np.log(C[0] / float(C[0]+D[0])) + (C[1] / float(C[1]+C[1])) * np.log(C[1] / float(C[1]+D[1])) ) 
          
    return IG


def classSepration(featureMatrix, labels):
	separated = {}
	for i in range(len(featureMatrix)):
		vector = featureMatrix[i]
		if (labels[i] not in separated):
			separated[labels[i]] = []
		separated[labels[i]].append(vector)
	return separated

def train(separated):
    prior={}
    termCount ={}
    
    h, w = len(separated['spam'][0]) , len(separated)
    condprob = [[0 for x in range(w)] for y in range(h)] 
    
    n =  sum ( len(v) for v in separated.itervalues())
    for c in separated.keys():
        prior[c]=len(separated[c]) / float(n)        
        termCount[c] = np.sum(separated[c], axis=0) #counting the frequency of the all word in training data
        
        allTermCount = np.sum(termCount[c]) # sum of the frequency of the all word in training data
        
        if c=='ham': cl=0
        else: cl=1
        
        for t in range(len(termCount[c])):
            condprob[t][cl] = (termCount[c][t] + 1) / (float (allTermCount) + len(termCount[c]) )
            
    return prior, condprob


def test(testFeatureMatrix, testLabels, prior, condprob):
    
    errorCount=0
    count=0
    for testVector, TrueLbl in zip(testFeatureMatrix, testLabels):
        count +=1
        score = [0] * 2
        for c in prior: 
            if c=='ham': 
                cl=0
                score[cl] = log(prior[c])
            else: 
                cl=1
                score[cl] = log(prior[c])
            
            
            for t in range( len(testVector)):
                score[cl] += testVector[t] * log(condprob[t][cl])
        
        if score[0] > score[1] : 
            predictedLbl = 'ham'
        else: 
            predictedLbl = 'spam'
        
        if predictedLbl != TrueLbl:
            errorCount +=1
    
    return 1- ( errorCount / float(count))
            

if __name__ == "__main__":

    """
    #-------Input : address ------------- 

    # please put your file's address here
    address = "/A_Azadeh_UTD/Courses/Spring18/ML/HomeWorks/HW/HW2/HW2/dataSet2"

    trainEmailPath = address + "/train"
    testEmailPath = address + "/test"
    """



    if len(sys.argv) < 2:
        sys.exit()
    else:

        trainEmailPath = sys.argv[1]
        testEmailPath = sys.argv[2]


    trainCorpus, trainLabels= readFile(trainEmailPath)
    word_features = nltk.FreqDist(w.lower() for w in nltk.tokenize.word_tokenize(trainCorpus))
    vocabulary = word_features.keys()
    

    trainFeatureMatrix= extractFeatures(trainCorpus, vocabulary)
    trainSeparated= classSepration(trainFeatureMatrix, trainLabels)
    

    prior, condprob= train(trainSeparated)
        
    # test part 
    testCorpus, testLabels= readFile(testEmailPath)
    testFeatureMatrix= extractFeatures(testCorpus, vocabulary)

    accuracy = test(testFeatureMatrix, testLabels, prior, condprob)

    print('Accuracy: {}.'.format(accuracy))

