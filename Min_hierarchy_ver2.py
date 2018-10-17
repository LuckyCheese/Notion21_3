#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:19:59 2018

@author: minoh
"""
#%%
import json
import numpy as np
import pandas as pd
import nltk
import re
from sklearn import feature_extraction
import matplotlib.pyplot as plt
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn 
#%%
data_r3 = []
data_r4 = []
data_r5 = []
for i in range(3, 6):
    for j in range(1, 6):
        with open('/Users/minoh/Documents/Usyd2018-2/COMP5703_Capstone/data/r%d/%d.json'%(i, j),'r') as d:  #modify path
            tem_data = json.load(d)
        if i == 3:
            data_r3.extend(tem_data['result']['docs'])
        elif i == 4:
            data_r4.extend(tem_data['result']['docs'])
        else:
            data_r5.extend(tem_data['result']['docs'])
    
#%%
print(len(data_r3))
print(len(data_r4))
print(len(data_r5))
#%%
# Delete duplicate json objects

def deleteDuplicate(data):
    all_data = [ each['content'] for each in data ] 
    unique = [ data[ all_data.index(id) ] for id in set(all_data) ]
    return unique

#%%
unique_r3 = deleteDuplicate(data_r3)
unique_r4 = deleteDuplicate(data_r4)
unique_r5 = deleteDuplicate(data_r5)

print(len(unique_r3))
print(len(unique_r4))
print(len(unique_r5))
#%%
# Extract content and metatdata from json 

title = []
content = []
publishedDate = []
summary = []
domain = []
#adultLanguage = [] no 'true' in our dataset


for i in range(len(unique_r3)):
    if 'title' in unique_r3[i]:
        title.append(unique_r3[i]['title'])
    else:
        title.append('')
    if 'content' in unique_r3[i]:
        content.append(unique_r3[i]['content'])
    else: 
        content.append('')
    if 'publishedDate' in unique_r3[i]:
        publishedDate.append(unique_r3[i]['publishedDate'])
    else:
        publishedDate.append('')
    if 'summary' in unique_r3[i]:        
        summary.append(unique_r3[i]['summary'])
    else:
        summary.append('')
        
    for j in range(len(unique_r3[i]['indexTerms'])):
        domain.append(unique_r3[i]['indexTerms'][j]['name'])
    
    #if 'adultLanguage' in unique_r3[i]:
    #    adultLanguage.append(unique_r3[i]['adultLanguage'])
   # else:
    #    adultLanguage.append('')
    
    
    
#%%
print(len(content), len(title), title[5])
print(content[0])

#%% Clean contents
def clean_text(text_list):
    """
    Take a text_list as an argument and 
    lower case, replace numbers
    return a cleaned text list.
    """
    clean_text = []
    clean = []
    for text in text_list:
        
        t = re.sub("\n", " ", text) #remove newline character
        t = re.sub("\xa0", " ", t) #remove noise from parsing
        t = re.sub("\ufeff", " ", t) #remove noise from parsing
        t = re.sub("[0-9]+", "NUM", t.lower()) #replace numbers
        clean.append(t)
        
    clean_text.extend(clean)
    return clean_text

#%%
clean_content = clean_text(content)
print(clean_content[0])
#%%
# load nltk's English stopwords as variable called 'stopwords'
#nltk.download('stopwords')
#stopwords = nltk.corpus.stopwords.words('english')

#%%
#print(stopwords)

#%%
## NLTK stopwords and sklear stopwords have different sets
## We first remove NLTK stop words from the tokenizer, then remove sklearn stopwords from tf-idf vectorizer

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer

# Define a tokenizer

def tokenize_stopwords_stem(text):
    # tokenize by sentence level, then word level
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    filtered_tokens = []
    
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    
    stopWords = set(nltk.corpus.stopwords.words('english'))
    stopwordsFiltered = []
    
    for w in filtered_tokens:
        if w not in stopWords:
            stopwordsFiltered.append(w)
            
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(t) for t in stopwordsFiltered]
    
    return(stems)
#%% test tokenizer
    
tokens = tokenize_stopwords_stem(clean_content[0])

#%%============================================================================
#Build a dictionary of stemmed words to locate the actual word in the documents later

# Tokenizer withouth stemming
def tokenize_stopwords(text):
    # tokenize by sentence level, then word level
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    filtered_tokens = []
    
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    
    stopWords = set(nltk.corpus.stopwords.words('english'))
    stopwordsFiltered = []
    
    for w in filtered_tokens:
        if w not in stopWords:
            stopwordsFiltered.append(w)
    
    return(stopwordsFiltered)
    
#%%
    '''
vocab_stemmed = []
vocab_not_stemmed = []
for i in clean_content:
    allwords_stemmed = tokenize_stopwords_stem(i)
    vocab_stemmed.extend(allwords_stemmed) 
    
    allwords_tokenized = tokenize_stopwords(i)
    vocab_not_stemmed.extend(allwords_tokenized)
    
#%%
# create a pandas DataFrame with the stemmed vocabulary as the index and the tokenized words as the column. 
# an efficient way to look up a stem and return a full token

vocab_dictionary = pd.DataFrame({'words': vocab_not_stemmed}, index = vocab_stemmed)
print('there are ' + str(vocab_dictionary.shape[0]) + ' items in vocab_frame')
'''
#==============================================================================
#%%
#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_stopwords, max_df=0.70,
                                 min_df=2, stop_words='english', lowercase=False,
                                 use_idf=True, ngram_range=(1,2))
        
vect = tfidf_vectorizer.fit(clean_content) #fit the vectorizer to content
feat_num = len(vect.get_feature_names())

#bag of words representation
X_train_vectorised = vect.transform(clean_content)
print(X_train_vectorised.shape)

#%%
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(X_train_vectorised)

#%%
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
'''
fig, ax = plt.subplots(figsize=(100, 200)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=title);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
'''
#uncomment below to save figure
#plt.savefig('ward_clusters.png', dpi=300) #save figure as ward_clusters
#%%
plt.title('Hierarchical Clustering Dendrogram (truncated)')
dendrogram(
    linkage_matrix, 
    orientation="right",
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

# Numbers in () is the number of documents in each cluster
#%%
# Retrieve the clusters

from scipy.cluster.hierarchy import fcluster
k=12
clusters = fcluster(linkage_matrix, k, criterion='maxclust')
print(clusters)
print(len(clusters))
#%% Create a list of document index for each cluster


#%%
# Logistic regression
model = LogisticRegression()
model.fit(X_train_vectorised, clusters)
#%%
feature_names = vect.get_feature_names()
class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for i, class_label in enumerate(class_labels):
    top50 = np.argsort(model.coef_[i])[-50:] # Most informative: positive
    print("%s: %s" % (class_label, ", ".join(feature_names[j] for j in top50)))
    #the last word is the most informative one
    top50 = np.argsort((-1) * model.coef_[i])[-50:] # Most informative: negative
    print("%s: %s" % (class_label, ", ".join(feature_names[j] for j in top50)))
    #the last word is the most informative one
    bottom50 = np.argsort(abs(model.coef_[i]))[:50] # Most irrelevant. Bottom 50 of the absolute values of coefficients 
    print("%s: %s" % (class_label, ", ".join(feature_names[j] for j in bottom50)))
    #the last word is the least informative one
    
    
    
#%% Extract sentences from the documents (create a dict)
    
    
    
    
    

#%%  
#===========================================================================
# Cut dendrogram at different levels to see the most informative feature at each level

# 2 clusters 
sorted_coef_index = model.coef_[0].argsort()
print("1: %s" % ", ".join(feature_names[i] for i in sorted_coef_index[:10]))
print("2: %s" % ", ".join(feature_names[i] for i in sorted_coef_index[:-11:-1]))

# 4 clusters

# and so on..


                 