import nltk
import numpy as np
import wordtovec

# parsin dataset.txt
dataset = []
with open('dataset.txt') as fp:
    fp.readline()
    for line in iter(fp.readline, ''):
        if len(line) > 2:
            data = line.split('"')
            sentence = data[1]
            features = data[0].split(',')
            features = features[:-1]
            features.append(sentence)
            dataset.append(features)

tokenList = []
wordToIdx = {}
idxToWord = {}

idx = 0
for data in dataset:
    review = data[-1]
    sentences = nltk.sent_tokenize(review)
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordToIdx:
                wordToIdx[token] = idx
                idxToWord[idx] = token
                idx += 1
        tokenList.append(tokens)
    data[-1] = tokenList
    tokenList = []

sourceWordIdx = []
targetWordIdx = []
windowSize = 2
for data in dataset:
    tokenList = data[-1]
    for i in range(len(tokenList)):
        tokenSen = tokenList[i]
        for j in range(len(tokenSen)):
            for k in range(windowSize):
                if j + k < len(tokenSen):
                    sourceWordIdx.append(wordToIdx[tokenSen[j]])
                    targetWordIdx.append(wordToIdx[tokenSen[j+k]])
                if j - k > 0:
                    sourceWordIdx.append(wordToIdx[tokenSen[j]])
                    targetWordIdx.append(wordToIdx[tokenSen[j-k]])

source = np.asarray(sourceWordIdx)
target = np.asarray(targetWordIdx)
lexiconSize = len(wordToIdx)
hiddenUnitNumber = 300
result = wordtovec.trainNetwork(source, target, lexiconSize, hiddenUnitNumber)

thefile = open('results.txt', 'w');
for i in range(lexiconSize):
    word = idxToWord[i]
    thefile.write("%s    "%word)
    for j in range(hiddenUnitNumber):
        idx = j * lexiconSize + i
        value = result[idx];
        thefile.write("%.3f "%value)
    thefile.write("\n")
