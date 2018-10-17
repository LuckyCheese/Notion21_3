import json
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def getdata(path): #get the data
    text = []
    with open(path,'r') as load_f:
     load_dict = json.load(load_f)
    for i in range(0,len(load_dict['result']['docs'])):
        text.append(load_dict['result']['docs'][i]['content'])
    return text # return a list
def singleadd(CurrentList, TotalList):#Distinct the data and merage
    for i in range(0,len(CurrentList)):
        num = 0
        for j in range(0,len(TotalList)):
            if CurrentList[i] == TotalList[j]:
                num = 1
                break
            else:
                continue
        if num == 0:
            TotalList.append(CurrentList[i])

    return TotalList  # return a list
def SentenceToken(raw):#setencetokenize
    sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    for i in range(0,len(raw)):
        sentences.append(sent_tokenizer.tokenize(raw[i]))
    return  sentences
def RemovePunct(line): #remove punctuation
    #identify = str.maketrans('', '')
    delEStr = string.punctuation +string.digits  #symbols
    CleanLine = []
    for i in range(0,len(line)):
        for j in range(0,len(line[i])):
                temp = re.sub(r'http://[a-zA-Z0-9.?/&=:]*','',line[i][j])
                CleanLine.append(re.sub(r'[^a-zA-Z]',' ',temp))
                # if temp != '':
                #     CleanLine.append(temp)

    return CleanLine
def WordToken(sentence):#
    word=[]
    for i in range(0,len(sentence)):

            wordsInStr = nltk.word_tokenize(sentence[i])
            word.append(wordsInStr)

    return word
def CountWord(WordList):
    WordCount = {}
    for i in range(0,len(WordList)):
        for j in range(0,len(WordList[i])):
            if WordCount.__contains__( WordList[i][j]):
                WordCount[WordList[i][j]] +=1
            else:
                WordCount[WordList[i][j]] = 1
    pass # 统计词频
    return WordCount
def RemoveStopWord(WordList):
    cleanWords=[]
    # filtered_words = [word for word in WordList if word not in stopwords.words('english')]
    for words in WordList:
        cleanWords+= [[w.lower() for w in words if w.lower() not in stopwords.words('english') ]]
    return cleanWords
def Tf_Idf(sentence):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentence)
    word = vectorizer.get_feature_names()
    print(word)
    print(X.toarray())
    transformer = TfidfTransformer()
    print (transformer)
    tfidf = transformer.fit_transform(X)
    print (tfidf.toarray())
path_r3_1 = "/Users/luca/Documents/capstone/r3/60da7349ce56c8d1267f0f5d93429878.json"
TotalTextList = []
CurrentTextList = getdata(path_r3_1)
TotalTextList = singleadd(CurrentTextList,TotalTextList)
SentenceList = SentenceToken(TotalTextList)

CleanList = RemovePunct(SentenceList)

Tf_Idf(CleanList)
WordList= WordToken(CleanList)

NoStopWord = RemoveStopWord(WordList)
WordCount = CountWord(NoStopWord)
print(WordCount)

