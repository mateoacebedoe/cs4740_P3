import nltk

# features we can consider for our sentiment tagger 

#@param sentence - string representing sentence from training/validatin set
#@spec returns a tokenized sentence with only words that fall in the RB, JJ, VB categories
#        according to nltk's pos-tagging
def strip( sentence ):
    essence= []
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    targets= ["RB","JJ","VB","VBD","VBN","NN"]
    for word,tag in tagged:
        if tag in targets:
            essence.append(word.replace("n't","not"))
    return essence

#@param sentence - list of words
#@poswords - list of positive words
#@negwords- list of negative words
#@return 1, 0 or -1 -> sentiment polarity of sentence base on poswords and negwords
def getPolarity( sentence, poswords, negwords ):
    polarities= []
    
    #initialize each word
    for word in sentence:
        #print word
        word= word.lower()
        if word in poswords:
            polarities.append(1)
        elif word in negwords:
            polarities.append(-1)
        else:
            polarities.append(0)
    
    counts= [0,0,0]
    for res in polarities:
        #print res
        counts[res+1]+= 1
    return counts.index( max( counts ) ) - 1, polarities
    
# Karl
#@param reviewsDic - mapping: reviewName => (sentiments, sentences)
#@param threshold - threshold for determining pos/neg sentiment
#@return a list of positive words and another of negative words
def getPosNegWordList( reviewsDic, poswords, negwords, threshold ):
    posWords= poswords
    negWords= negwords
    posCounts= dict()
    negCounts= dict()
    allCounts= dict() #mapping: word => total number of occurrences of word
    for reviewName in reviewsDic:
        (sentiments, sentences)= reviewsDic[reviewName]
        i= 0
        length= len(sentiments)
        while i < length:
            sentiment, sentence= sentiments[i], sentences[i]
            tokens= sentence
            for word in tokens:
                try:
                    allCounts[word]+= 1
                except KeyError:
                    allCounts[word]= 1
                try:
                    if sentiment == 1:
                        posCounts[word]+= 1
                    elif sentiment == -1:
                        negCounts[word]+= 1
                except KeyError:
                    if sentiment == 1:
                        posCounts[word]= 1
                    elif sentiment == -1:
                        negCounts[word]= 1
            i+=1
        
    for word in posCounts:
        count= posCounts[word]
        if count / allCounts[word] > threshold:
            posWords.append(word.lower())
    
    for word in negCounts:
        count= negCounts[word]
        if count / allCounts[word] > threshold:
            negWords.append(word.lower())
            
    return posWords,negWords
    
# Karl
#@param sentence - target sentence
#@param threshold - 
#@return 1 on true, 0 on false
def isLong(sentence, threshold):
    if len(sentence) > threshold:
        return 1
    else:
        return 0

# Karl
#@param sentence - target sentence
#@return 1 if sentence has an all CAPS word,
#              0 otherwise
def containsAllCaps(sentence):
    for word in sentence:
        if word.isupper():
            return 1
            
    return 0

# Karl
#@param sentence- target sentece
#@return (1,num pos words) or 0,0 if sentence has no positive words
def containsPos(sentence, poswords):
    success= 0
    numPositives= 0
    for word in sentence:
        if word.lower() in poswords:
            success= 1
            numPositives+= 1
    return success, numPositives

# Karl
#@param sentence- target sentence
#@return (1,num neg words) or 0,0 if sentence has no negative words
def containsNeg(sentence, negwords):
    success= 0
    numNegatives= 0
    for word in sentence:
        if word.lower() in negwords:
           success=1
           numNegatives+= 1
    return success,numNegatives

# Karl
#@param sentence - target sentence
#@return 1 if ! is in sentence, 0 otherwise
def containsExclamation(sentence):
    if "!" in sentence:
        return 1
    return 0