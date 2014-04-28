import nltk

# features we can consider for our sentiment tagger 

#@param sentence - string representing sentence from training/validatin set
#@spec returns a tokenized sentence with only words that fall in the RB, JJ, VB categories
#        according to nltk's pos-tagging
def strip( sentence ):
    essence= []
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    for word,tag in tagged:
        if tag == "RB" or tag == "JJ" or tag == "VB":
            essence.append(word)
    return essence