import nltk
from nltk.stem.lancaster import LancasterStemmer

positiveFile = open('positive-words.txt', 'rU')
negativeFile = open('negative-words.txt', 'rU')
positiveList = positiveFile.read().strip().split('\n')
negativeList = negativeFile.read().strip().split('\n')
negationList = ['no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never', "n't"]
intensifierList = ['so', 'too', 'very', 'really', 'awful', 'bloody', 'dead',\
'dreadfully', 'extremely', 'fucking', 'hella', 'most', 'precious', 'quite',\
'real', 'remarkably', 'terribly', 'moderately', 'wicked', 'bare', 'rather',\
'somewhat', 'fully', 'ass', 'super']

def computeFeatureVector(sentence):
    features = str(containPositive(sentence)) +str(containNegation(sentence)) + str(containIntensifier(sentence))
    return features
emissionProb = computeEmissionProbability(trainingData)
emissionProb


def containIntensifier(sentence):
    intensifierList = ['so', 'too', 'very', 'really', 'awful', 'bloody', 'dead',\
    'dreadfully', 'extremely', 'fucking', 'hella', 'most', 'precious', 'quite',\
    'real', 'remarkably', 'terribly', 'moderately', 'wicked', 'bare', 'rather',\
    'somewhat', 'fully', 'ass', 'super']
    for w in sentence:
        if w[0].lower() in intensifierList:# and w[1] == 'RB':
            return 1
    return 0

def containNegation(sentence):
    negationList = ['no', 'not', 'none', 'nobody', 'nothing', 
    'neither', 'nowhere', 'never', "n't", 'hardly', 'barely', 'scarcely']
    for w in sentence:
        if w[0].lower() in negationList:
            return 1
    return 0

def PosNegNegation(sentence):
    positiveFile = open('positive-words.txt', 'rU')
    negativeFile = open('negative-words.txt', 'rU')
    positiveList = positiveFile.read().strip().split('\n')
    negativeList = negativeFile.read().strip().split('\n')
    positiveFile.close()
    negativeFile.close()
    pos = 0
    neg = 0
    if containNegation:
        for w in sentence:
            if w[0] in positiveList:
                pos += 1
                if 'JJ' in w[1]:
                    return -1
            elif w[0] in negativeList:
                neg += 1
                if 'JJ' in w[1]:
                    return 1
        if pos > neg:
            return 1
        elif neg > pos:
            return -1
        else:
            return 0

def containPositive(sentence):
    positiveFile = open('positive-words.txt', 'rU')
    negativeFile = open('negative-words.txt', 'rU')
    positiveList = positiveFile.read().strip().split('\n')
    negativeList = negativeFile.read().strip().split('\n')
    positiveFile.close()
    negativeFile.close()
    pos = 0
    neg = 0
    for token in sentence:
        if token[0].lower() in positiveList:
            if 'JJ' in token[1]:
                pos += 1
            pos += 1
        elif token[0].lower() in negativeList:
            if 'JJ' in token[1]:
                neg += 1
            neg += 1
    if pos > neg:
        return 1
    elif neg > pos:
        return -1
    else:
        return 0

#Check if the sentence is longer than k words
#Input: a sentence, an integer k
#Output: 1 if sentence is longer than k, 0 otherwise
def isLong(sentence, k):
    if len(sentence) > k:
        return 1
    else:
        return 0

#Check if the sentence contains any word that is all capitalized
#Input: a sentence
#Output: 1 if there is at least one, 0 otherwise
def capitalization(sentence):
    for token in sentence:
        if token.isupper():
            return 1
    return 0
    
#Check if the sentence is ending with one or more '!' or '?'
#Input: a sentence
#Output: 1 if it contains with one, 0 otherwise
def punctuation(sentence):
    words = [s[0] for s in sentence]
    count = 0
    for s in words:
        if s == '!' or s == '?':
            count += 1
    if count > 1:
        return 1
    else:
        return 0
    if '!' in words:
        return 1
    else:
        return 0

#Read file into training data and validation data
#Input: the name string of the file
#output: a tuple, the first element is the training data set
#       the second element is the validation data set
#       each data set is a dictionary whose keys are the titles
#       and values are tuples of (sentiments, sentences)
#       dic['reviewName'] = > ([sentiment labels],[])
#       sentiments are in integer form
def getTraining(fileName):
    f = open(fileName, 'rU')
    allData = f.readlines()
    allTitles = []
    st = LancasterStemmer()
    for l in allData:
        if len(l.split('_')) == 3:
            allTitles.append(l)
    trainingDataSize = int(len(allTitles) * 0.8)
    validationDataSize = len(allTitles) - trainingDataSize
    trainingData = {}
    validationData = {}
    count = 0
    i = 0
    while i < len(allData) and len(trainingData) < trainingDataSize:
        tags = []
        sentences = []
        while allData[i] != '\n':
            if allData[i] in allTitles:
                key = allData[i]
                i += 1
            else:
                tag = allData[i].split('\t')[0]
                if tag == 'pos':
                    numtag = 1
                elif tag == 'neg':
                    numtag = -1
                else:
                    numtag = 0
                sentence = allData[i].split('\t')[1]
                tokens = nltk.word_tokenize(sentence)
                tokens = nltk.pos_tag(tokens)
                tags.append(numtag)
                sentences.append(tokens)
                i += 1
        trainingData[key] = (tags, sentences)
        i += 1
        
        while i < len(allData) and len(validationData) < validationDataSize:
            tags = []
            sentences = []
            while allData[i] != '\n' and allData[i] not in trainingData.keys():
                if allData[i] in allTitles:
                    key = allData[i]
                    i += 1
                else:
                    tag = allData[i].split('\t')[0]
                    if tag == 'pos':
                        numtag = 1
                    elif tag == 'neg':
                        numtag = -1
                    else:
                        numtag = 0
                    sentence = allData[i].split('\t')[1]
                    tags.append(numtag)
                    sentence = allData[i].split('\t')[1]
                    tokens = nltk.word_tokenize(sentence)
                    tokens = nltk.pos_tag(tokens)
                    sentences.append(tokens)
                    i += 1
            validationData[key] = (tags, sentences)
            i += 1
    f.close()
    return (trainingData, validationData)

#Given the training data, compute what is the probability each sentiment
#follows another
#Input: the training data set
#Output: a dictionary of 12 probabilities
def computeTransitionProbability(trainingData):
    (Pos, Neg, Neu, PosToPos, PosToNeg, PosToNeu, NegToPos, NegToNeg, NegToNeu,\
    NeuToPos, NeuToNeg, NeuToNeu) = (0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    (StartPos, StartNeg, StartNeu) = (0.0, 0.0, 0.0)
    for key in trainingData:
        sentiments = trainingData[key][0]
        if sentiments[0] == 1:
            StartPos += 1
        elif sentiments[0] == 0:
            StartNeu += 1
        else:
            StartNeg += 1
        for i in range(len(sentiments)-1):
            thisWord = sentiments[i]
            nextWord = sentiments[i+1]
            if thisWord == 1:
                Pos += 1
                if nextWord == 1:
                    PosToPos += 1
                elif nextWord == 0:
                    PosToNeu += 1
                else:
                    PosToNeg += 1
            elif thisWord == 0:
                Neu += 1
                if nextWord == 1:
                    NeuToPos += 1
                elif nextWord == 0:
                    NeuToNeu += 1
                else:
                    NeuToNeg += 1
            else:
                Neg += 1
                if nextWord == 1:
                    NegToPos += 1
                elif nextWord == 0:
                    NegToNeu += 1
                else:
                    NegToNeg += 1
        #if sentiments[-1] == 1:
            #Pos += 1
        #elif sentiments[-1] == 0:
            #Neu += 1
        #else:
            #Neg += 1
    reviewNum = len(trainingData)
    (StartPos, StartNeu, StartNeg, PosToPos, PosToNeg, PosToNeu, NegToPos, NegToNeg, NegToNeu,\
    NeuToPos, NeuToNeg, NeuToNeu) = (StartPos / reviewNum, StartNeu / reviewNum, StartNeg / reviewNum,\
    PosToPos / Pos, PosToNeg / Pos, PosToNeu / Pos, NegToPos / Neg, NegToNeg / Neg, NegToNeu / Neg, \
    NeuToPos / Neu, NeuToNeg / Neu, NeuToNeu / Neu)
    return {'sTo1': StartPos, 'sTo0': StartNeu, 'sTo-1': StartNeg, \
    '1To1': PosToPos, '1To-1': PosToNeg, '1To0': PosToNeu,'-1To1': NegToPos,\
    '-1To-1': NegToNeg, '-1To0': NegToNeu,'0To1': NeuToPos, '0To-1': NeuToNeg, \
    '0To0': NeuToNeu}

#Given a sentence, compute its emission probability
#Input: the training data set
#Output: a dictionary whose keys are the feature vectors and values are
#        triples (Positive probability, Neutual probability, Negative probability)
def computeEmissionProbability(trainingData):
    EmissionProbability = {}
    for key in trainingData:
        (sentiments, sentences) = trainingData[key]
        for i in range(len(sentiments)):
            sentiment = sentiments[i]
            sentence = sentences[i]
            featureVector = computeFeatureVector(sentence)
            if featureVector in EmissionProbability:
                (Pos, Neu, Neg) = EmissionProbability[featureVector]
                if sentiment == 1:
                    Pos += 1
                elif sentiment == 0:
                    Neu += 1
                else:
                    Neg += 1
                EmissionProbability[featureVector] = (Pos, Neu, Neg)
            else:
                if sentiment == 1:
                    (Pos, Neu, Neg) = (1.0, 0.0, 0.0)
                elif sentiment == 0:
                    (Pos, Neu, Neg) = (0.0, 1.0, 0.0)
                else:
                    (Pos, Neu, Neg) = (0.0, 0.0, 1.0)
                EmissionProbability[featureVector] = (Pos, Neu, Neg)
    for feature in EmissionProbability:
        (Pos, Neu, Neg) = EmissionProbability[feature]
        total = Pos + Neu + Neg
        EmissionProbability[feature] = (Pos / total, Neu / total, Neg / total, total)
    return EmissionProbability

#Input: a list of sentences, a set of transition probabilities and a
#       set of emission probabilities
#Output: a list of sentiment in order
def Verterbi(review, transitionProb, emissionProb):
    #Initialization
    (sToPos, sToNeu, sToNeg) = (transitionProb['sTo1'],\
    transitionProb['sTo0'], transitionProb['sTo-1'])
    sentence = review[0]
    firstFeatureVector = computeFeatureVector(sentence)
    PosProb = emissionProb[firstFeatureVector][0] * sToPos
    NeuProb = emissionProb[firstFeatureVector][1] * sToNeu
    NegProb = emissionProb[firstFeatureVector][2] * sToNeg
    sentimentList = []
    maxProb = max(PosProb, NeuProb, NegProb)
    if maxProb == PosProb:
        sentimentList.append(1)
    elif maxProb == NeuProb:
        sentimentList.append(0)
    else:
        sentimentList.append(-1)
    for t in range(1, len(review)):
        sentence = review[t]
        (prevPos, prevNeu, prevNeg) = (PosProb, NeuProb, NegProb)
        featureVector = computeFeatureVector(sentence)
        (ePos, eNeu, eNeg, numOfAppearance) = emissionProb[featureVector]
        PosProb = max(prevPos * transitionProb['1To1'], prevNeu * transitionProb['0To1'],\
        prevNeg * transitionProb['-1To1']) * ePos
        NeuProb = max(prevPos * transitionProb['1To0'], prevNeu * transitionProb['0To0'],\
        prevNeg * transitionProb['-1To0']) * eNeu
        NegProb = max(prevPos * transitionProb['1To-1'], prevNeu * transitionProb['0To-1'],\
        prevNeg * transitionProb['-1To-1']) * eNeg
        maxProb = max(PosProb, NeuProb, NegProb)
        if maxProb == PosProb:
            sentimentList.append(1)
        elif maxProb == NeuProb:
            sentimentList.append(0)
        else:
            sentimentList.append(-1)
    return sentimentList

(trainingData, validationData) = getTraining('training_data.txt')
transitionProb = computeTransitionProbability(trainingData)

sentiment = Verterbi(validationData[validationData.keys()[0]][1], transitionProb, emissionProb)        

def computeTransitionWithReview(trainingData):
    #Generate 3 data sets, each with one review sentiment
    posTraining = {}
    neuTraining = {}
    negTraining = {}
    for title in trainingData:
        titleSen = title.split('_')[1]
        if titleSen == 'pos':
            posTraining[title] = trainingData[title]
        elif titleSen == 'neu':
            neuTraining[title] = trainingData[title]
        else:
            negTraining[title] = trainingData[title]
            
    (posResult, neuResult, negResult) = ({}, {}, {})
    posResult = computeTransitionProbability(posTraining)
    neuResult = computeTransitionProbability(neuTraining)
    negResult = computeTransitionProbability(negTraining)
    
    return (posResult, neuResult, negResult)

validationResults = {}
errorSum = 0.0
total = 0
for title in validationData:
    if title.split('_')[1] == 'pos':
        sentiments = Verterbi(validationData[title][1], transitionProb[0], emissionProb)
    elif title.split('_')[1] == 'neu':
        sentiments = Verterbi(validationData[title][1], transitionProb[1], emissionProb)
    else:
        sentiments = Verterbi(validationData[title][1], transitionProb[2], emissionProb)
    errorSum += sum([abs(sentiments[i] - validationData[title][0][i]) for i in range(len(sentiments))])
    total += len(sentiments)
    validationResults[title] = sentiments
accuracy = 1 - errorSum / total
accuracy

testResults = []
def getTesting(fileName):
    f = open(fileName, 'rU')
    allData = f.readlines()
    allTitles = []
    testingData = {}
    for l in allData:
        if len(l.split('_')) == 3:
            allTitles.append(l)
    i = 0
    while i < len(allData):
        sentences = []
        while allData[i] != '\n':
            if allData[i] in allTitles:
                key = allData[i]
                i += 1
            else:
                sentence = allData[i].split('\t')[1]
                tokens = nltk.word_tokenize(sentence)
                tokens = nltk.pos_tag(tokens)
                sentences.append(tokens)
                i += 1
        testingData[key] = sentences
        i += 1
    return (allTitles, testingData)
    
(titles, testingData) = getTesting('test_data_no_true_labels.txt')

for title in titles:
    if title.split('_') == 'pos':
        testSentiments = Verterbi(testingData[title], transitionProb[0], emissionProb)
    elif title.split('_') == 'neu':
        testSentiments = Verterbi(testingData[title], transitionProb[1], emissionProb)
    else:
        testSentiments = Verterbi(testingData[title], transitionProb[2], emissionProb)
    testResults.append(testSentiments)

output = open('output.txt', 'w')
output.write('Id,answer\n')
testId = 0
for r in testResults:
    for i in range(len(r)):
        outStr = str(testId) + ',' + str(r[i]) + '\n'
        output.write(outStr)
        testId += 1
output.close()