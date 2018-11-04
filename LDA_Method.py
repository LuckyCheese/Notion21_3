import os
import json
from keras.preprocessing.text import Tokenizer
import re
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from nltk.tokenize import RegexpTokenizer
import gensim
from gensim import models
from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.models import Doc2Vec
from gensim.models import TfidfModel
import pyLDAvis
import pyLDAvis.gensim
from nltk.stem import WordNetLemmatizer
from pprint import pprint
from nltk.tag.stanford import StanfordNERTagger
from gensim.models import CoherenceModel
import time
import pymongo
import py2neo
from py2neo import Node, Relationship
from py2neo import Graph
from nltk.tokenize import  sent_tokenize


def load_data(dir,round,args):
    StartTime=time.time()
   
    data = []
    data_args = []
    for i in range(1,6):
        with open(os.path.join(dir,'r%d/%d.json'%(round,i)),'r') as d:
            tem_data = json.load(d)
        data.extend(tem_data['result']['docs'])
    for i in range(len(data)):
        data_args.append(data[i][args])
        data_args = list(set(data_args))
    EndTime=time.time()
    print("Running Time:",np.round(EndTime-StartTime,2),'s')
    return data_args

def getMyStopWordList(text,st,Type):
    myStops=[]
    ctext=re.sub("[^a-zA-Z]"," ", text)
    t=st.tag(ctext.split())
    for name in t:
        if(name[1]==Type):
            myStops.append(name[0].lower())
    return myStops
def review_to_wordlist(text, myStops,remove_stopwords=True):
             
       
    t = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)#remove http url
    t = re.sub(r'www.\S+','',t)#remove www url
    t = re.sub("[\\w.]*@[\\w.]+", "", t)# remove email and twitter account
    t = re.sub("\\S+.com\\S+",'',t) # remove .com url
    t = re.sub("\\S+.com",'',t) # remove .com url
    t = re.sub("\\S+.COM",'',t) # remove .com url
    t = re.sub("(Politics|Business|World|National|Sport|Entertainment|Lifestyle|Money|Environment|World Cup 2018) (Show|Hide) subsections", "", t)
    t = re.sub("Normal text sizeLarger text sizeVery large text size", "", t)
    t = re.sub("WAtoday", "", t)
    t = re.sub("The Sydney Morning Herald", "", t)
    t = re.sub("(Exclusive|Opinion) Normal text", "", t)
    t = re.sub("[^a-zA-Z]"," ", t)
    t = re.sub("\xa0", " ", t) #remove noise from parsing
    t = re.sub("\ufeff", " ", t) #remove noise from parsing
    
    # 3. Convert words to lower case and split them
    words = t.lower().split()
    #
    # 4. Optionally remove stop words (True by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        
        words = [w for w in words if not w in stops]
        words = [w for w in words if not w in myStops]
    stem_word=[]
    stemmer = WordNetLemmatizer() #PorterStemmer()
    
    for word in words:
      
       
        stem_word.append(stemmer.lemmatize(word))
        
    # 5. Return a list of words
    return stem_word


def KeySentence_with_weight(data, article_number, key_word_list,prob_list,n_words,top_n_sentence,myStops):
    article=data[article_number]
    i=0
    j=0
    sentenceList = sent_tokenize(article)
    newList=[]
    resultList=[]
    LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
    #print(article)
    #print(sentenceList)
    all_clean_sentence=[]
    for s in sentenceList:           
  
        #Pre-processing
        processed_sentence = review_to_wordlist(s,myStops)
        
        
        # add tokens to list
        if len(processed_sentence)>= 20:
            all_clean_sentence.append(LabeledSentence1(processed_sentence,[j]))
            j=j+1
            newList.append(sentenceList[i])
        i=i+1
    if j>0:
        sentence_scores=np.zeros(j)
        d2v_model = Doc2Vec(all_clean_sentence, vector_size = 1000, min_count = 1, workers=7, dm = 1, 
                        alpha=0.025, min_alpha=0.001)
        d2v_model.train(all_clean_sentence, total_examples=d2v_model.corpus_count, epochs=100, start_alpha=0.002, end_alpha=-0.016)
        for n in range(0,n_words):
            tokens = key_word_list[1][n].split()
            prob=prob_list[1][n]
            new_vector = d2v_model.infer_vector(tokens)
            sims = d2v_model.docvecs.most_similar([new_vector],topn=j)

            for result in sims:
                if(result[0]<=j):
                    sentence_scores[result[0]]=sentence_scores[result[0]]+prob*result[1]
            #print(all_clean_sentence)
            #print(sentenceList)


        index=np.argsort(sentence_scores)
        if j<top_n_sentence:
            top_n_sentence=j
        for i in range(0,top_n_sentence):
            resultList.append(newList[index[j-i-1]])
    return resultList


def getLDAPerfomance(corpus_tfidf, dictionary,all_content, n_topics,iterations,passes):
    LDA_model = ldamodel.LdaModel(corpus_tfidf, 
                              id2word=dictionary, 
                              num_topics=n_topics,
                              iterations=iterations,
                              random_state=np.random.seed(1),
                            
                              passes=passes,
                              alpha='auto',
                              eta='auto'
                              )
    # a measure of how good the model is. lower the better.
    perplexity=LDA_model.log_perplexity(corpus_tfidf)
    # a measure of how good the model is. higher the better.
    coherence_model_lda = CoherenceModel(model=LDA_model, texts=all_content, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    
    return perplexity,coherence_score


def getNameFromFile(round):
    StartTime=time.time()
    
    myList=[]
    if round==4:
        file = open('Round4name.txt', 'r') 
     
        for line in file: 
            nameList=line.split(',')
            for name in nameList:
                name=name.strip()
                myList.append(name[1:-1])
    elif round ==3:
        file = open('Round3name.txt', 'r') 
        for line in file: 
            nameList=line.split(',')
            for name in nameList:
                name=name.strip()
                myList.append(name[1:-1])
    elif round ==5:
        file = open('Round3name.txt', 'r') 
        for line in file: 
            nameList=line.split(',')
            for name in nameList:
                name=name.strip()
                myList.append(name[1:-1]) 
    EndTime=time.time()
    print("Running Time:",np.round(EndTime-StartTime,2),'s')
    return myList

def getNameFromClassifier(data):
    # Number of corpus
    l=len(data)
    # Create Stop words
    StartTime=time.time()
    myStops=[]
    st = StanfordNERTagger('/Users/shengyuan/desktop/study/Capstone/RoyalCommission/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
                               '/Users/shengyuan/desktop/study/Capstone/RoyalCommission/stanford-ner-2018-02-27/stanford-ner.jar',
                                encoding='utf-8')
    count=0
    Organization_Name=[]
    for text in data:  
        StopWordList=getMyStopWordList(text,st,'PERSON')
        OrganizationList=getMyStopWordList(text,st,'ORGANIZATION')
        for item in StopWordList:
            myStops.append(item)
        for item in OrganizationList:
            Organization_Name.append(item)  
        count=count+1
        print(count)

    Organization_Name=set(Organization_Name)
    for item in Organization_Name:
        if item in myStops:
            myStops.remove(item)

    EndTime=time.time()
    Manual_StopWords=['royal','commission','ms','mr','per','cent','commissions','c','m','orr','hearing','hear','show','said','u']
    for item in Manual_StopWords:
        myStops.append(item)
    myStops=set(myStops)   
    print("Running Time:",np.round(EndTime-StartTime,2),'s')
    return myStops

def LDA_Analysis(data,myStops):

    # Number of corpus
    l=len(data)
    all_content = []
    #Pre-processing
    for text in data:             
        processed_content = review_to_wordlist(text,myStops)
        # add tokens to list
        all_content.append(processed_content)

    # Prepare LDA 
    dictionary = Dictionary(all_content)
    corpus = [dictionary.doc2bow(text) for text in all_content]

    # TF-IDF
    tfidf_model = TfidfModel(corpus)  # fit model
    corpus_tfidf = tfidf_model[corpus]
    
    # Analysis of LDA model
    # Number of Passes
    print("Passes Analysis")
    StartTime=time.time()
    perplexity_list=[]
    coherence_score_list=[]
    iteration=80
    n_topic=5
    passes_range=np.arange(1,6)
    for passes in passes_range:
        p,c=getLDAPerfomance(corpus_tfidf, dictionary, all_content,n_topic,iteration,passes)
        perplexity_list.append(p)
        coherence_score_list.append(c)

    plt.plot(passes_range, coherence_score_list)
    plt.xlabel("Num Passes")
    plt.ylabel("Coherence score")
    plt.title('Passes VS Coherence score')
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    plt.plot(passes_range, perplexity_list)
    plt.xlabel("Passes")
    plt.ylabel("perplexity")
    plt.title('Passes VS perplexity')
    plt.legend(("perplexity"), loc='best')
    plt.show()
    EndTime=time.time()
    print("Running Time:",np.round(EndTime-StartTime,2),'s','\n')
    
    # Number of Topic
    print("Number of Topic Analysis")
    StartTime=time.time()
    perplexity_list=[]
    coherence_score_list=[]
    iteration=80
    passes=3
    topic_num_range=np.arange(2,11)
    for n in topic_num_range:
        p,c=getLDAPerfomance(corpus_tfidf, dictionary, all_content,n,iteration,passes)
        perplexity_list.append(p)
        coherence_score_list.append(c)

    plt.plot(topic_num_range, coherence_score_list)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.title('Num Topics VS Coherence score')
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    plt.plot(topic_num_range, perplexity_list)
    plt.xlabel("Num Topics")
    plt.ylabel("perplexity")
    plt.title('Num Topics VS perplexity')
    plt.legend(("perplexity"), loc='best')
    plt.show()
    EndTime=time.time()
    print("Running Time:",np.round(EndTime-StartTime,2),'s','\n')
    
    # Number of iteration
    print("Number of Topic Analysis")
    StartTime=time.time()
    perplexity_list=[]
    coherence_score_list=[]
    iteration_range=np.arange(10,200,10)
    passes=3
    n_topic=5
    for iteration in iteration_range:
        p,c=getLDAPerfomance(corpus_tfidf, dictionary, all_content,n_topic,iteration,passes)
        perplexity_list.append(p)
        coherence_score_list.append(c)

    plt.plot(iteration_range, coherence_score_list)
    plt.xlabel("Iteration")
    plt.ylabel("Coherence score")
    plt.title('Iteration VS Coherence score')
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    plt.plot(iteration_range, perplexity_list)
    plt.xlabel("Iteration")
    plt.ylabel("perplexity")
    plt.title('Num Topics VS perplexity')
    plt.legend(("perplexity"), loc='best')
    plt.show()
    EndTime=time.time()
    print("Running Time:",np.round(EndTime-StartTime,2),'s','\n')



def myLDA(data,myStops,n_topics,n_words,passes,iteration, top_n_sentence,top_n_article,graph):
    StartTime=time.time()
    # Number of corpus
    l=len(data)
    all_content = []
    #Pre-processing
    for text in data:             
        processed_content = review_to_wordlist(text,myStops)
        # add tokens to list
        all_content.append(processed_content)
    # Prepare LDA 
    dictionary = Dictionary(all_content)
    corpus = [dictionary.doc2bow(text) for text in all_content]

    # TF-IDF
    tfidf_model = TfidfModel(corpus)  # fit model
    corpus_tfidf = tfidf_model[corpus]
    LDA_model = ldamodel.LdaModel(corpus_tfidf, 
                              id2word=dictionary, 
                              num_topics=n_topics,
                              iterations=iteration,
                              random_state=np.random.seed(1),# setting random seed to get the same results each time.        
                              passes=passes,
                              alpha='auto',
                              eta='auto'
                             )
    print("Topic word probability")
    print(LDA_model.show_topics())
    print("*******************************************************************************************************\n")
    
    # Finding the topic word list and their coreponding probablity
    x=LDA_model.show_topics(num_topics=n_topics, num_words=n_words,formatted=False)
    topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
    topics_words_probability=[(tp[0], [wd[1] for wd in tp[1]]) for tp in x]

    all_topics = LDA_model.get_document_topics(corpus_tfidf, per_word_topics=True)
    count=1
    content_list=[]
    content_num=[]
    topic_list=[]
    prob_list=[]
    tp_words=[]
    for doc_topics, word_topics, phi_values in all_topics:
        #print('Document ',count,'\n')
        #print('Document topics:', doc_topics)

        for doc_prob in doc_topics:
            topic_list.append(doc_prob[0])
            prob_list.append(doc_prob[1])
            words=" ".join(topics_words[doc_prob[0]][1])
            tp_words.append(words)

            content_list.append(data[count-1])
            content_num.append(count-1);
        #print ('Word topics:', word_topics)
        #print ('Phi values:', phi_values)
        #print(" ")
        #print('-------------- \n')
        count=count+1
    
    # Create the table to display the sorted result
    df2 = pd.DataFrame(topic_list,columns=['Topic'])
    df3 = pd.DataFrame(content_list,columns=['Content'])
    df4 =pd.DataFrame(tp_words,columns=['Topic Words'])
    df5 =pd.DataFrame(content_num,columns=['Article Number'])
    df1=pd.DataFrame(prob_list,columns=['Probability'])

    df1 = df1.assign(df2=df2.values)
    df1 = df1.assign(df3=df3.values)
    df1=df1.assign(df4=df4.values)
    myTable = df1.assign(df5=df5.values)
    myTable.columns=['Probability','Topic','Cotent','Topic Words','Article Number']
    sortedTable=myTable.sort_values(by=['Topic', 'Probability'],ascending=[True,False])
    print(sortedTable.head())
    print("*******************************************************************************************************\n")
    
    
    # Retrieve key sentences
    #Parameter

    
    Hearing_node=Node("Hearings",Round=round)
    graph.create(Hearing_node)

    MyResult={}
    MyResult['Round'+str(round)]={}
    Topic_size=[]
    Topic_label=[]
    Topic_color= ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(n_topics)]
    for i in range(0,n_topics):
        df=sortedTable.loc[sortedTable['Topic'] == i]
        df.reset_index()

        
        
        Topic_index="Topic"+str(i+1)
        
        Topic_size.append(df.shape[0])
        Topic_label.append(Topic_index)
        
        key_words=df.iloc[0,-2]
        Topic_node=Node("Topic",Round=round,Topic_ID=int(i+1),Key_Words=key_words)

        Relation_Shows=Relationship(Hearing_node,'Shows',Topic_node)
        graph.create(Topic_node)
        graph.create(Relation_Shows)

        for j in range(0,top_n_article):
            article_number=df.iloc[j,-1]
            key_words=df.iloc[j,-2]
            key_word_list=topics_words[i] 
            prob_list=topics_words_probability[i] 
            a2=KeySentence_with_weight(data,article_number, key_word_list,prob_list,n_words,top_n_sentence,myStops)

            Article_node=Node("Article",Article_ID= int(article_number),Round=round,Topic_ID=int(i+1))


            Relation_Retrieves=Relationship(Topic_node,'Retrieves',Article_node)
            graph.create(Article_node)
            graph.create(Relation_Retrieves)

            for item in a2:
                Key_Sentence_node=Node("Key_Sentences",Key_Sentence=item)
                Relation_Provides=Relationship(Article_node,'Provides',Key_Sentence_node)
                graph.create(Key_Sentence_node)
                graph.create(Relation_Provides)

            MyResult['Round'+str(round)][Topic_index]={}   
            MyResult['Round'+str(round)][Topic_index]['Key Words']=key_words
            MyResult['Round'+str(round)][Topic_index]['Article'+str(article_number)]={}
            MyResult['Round'+str(round)][Topic_index]['Article'+str(article_number)]['Key Setences']=a2

    pprint(MyResult)
    # Plot
    print("Topic distribution")
    explode = np.zeros(n_topics)
    explode[0]=0.1
    plt.pie(Topic_size, explode=explode,labels=Topic_label, colors=Topic_color,
        autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('Topic distribution')
    plt.figure(figsize=(40, 40))
    plt.show()
    EndTime=time.time()
    print("Running Time:",np.round(EndTime-StartTime,2),'s')
    return MyResult,LDA_model, corpus_tfidf, dictionary


if __name__=='__main__':
    # Loading
    # Loading Parameter
    dir='/Users/shengyuan/Desktop/Study/CAPSTONE/RoyalCommission' #Data fold path
    args = 'content'                                              #Data type
    round=4                                                       #Round number
    print("Start to load data")
    data = load_data(dir,round,args)
    # Number of corpus
    l=len(data)
    print('Total corpus: ',l)
    print("End")
    print("-------------------------------------------------------------------------------------------------------\n")

    # Get stop-words
    print("Start to get stop-words")
    #myStops=getNameFromClassifier(data)
    myStops=getNameFromFile(round)
    print("End")
    print("-------------------------------------------------------------------------------------------------------\n")

    # LDA performance
    print("Start to analyze performance of LDA")
    LDA_Analysis(data,myStops)
    print("End")
    print("-------------------------------------------------------------------------------------------------------\n")

    # Optimal Parameter
    n_topics=7
    n_words=8
    passes=3
    iteration=50
    top_n_sentence=2
    top_n_article=2
    graph = Graph(password="123")
    print("Start to get the theme")
    myResult,LDA_model, corpus_tfidf, dictionary=myLDA(data,myStops,n_topics,n_words,passes,iteration, top_n_sentence,top_n_article,graph)
    print("End")
    print("-------------------------------------------------------------------------------------------------------\n")

    # Visulization
    lda_vis = pyLDAvis.gensim.prepare(LDA_model, corpus_tfidf, dictionary)
    pyLDAvis.display(lda_vis)
