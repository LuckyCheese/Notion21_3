# -Notion21_3

# 1. Hierarchical clustering model
## Required files:
 - Hierarchy_Min_Final.ipynb
 - stop.txt

## Required data:
 - Royal Commissions Round3, Round4, Round5 hearings data
 The data for each hearing round should be stored in separate folders. Each round contains 5 json files. Folder and file names should be created as shown below:

parent(specify the path to the parent directory)<br>
|--r3 (folder)<br>
|--|--1.json<br>
|--|--2.json<br>
|--|--3.json<br>
|--|--4.json<br>
|--|--5.json<br>
|--r4 (folder)<br>
|--|--1.json<br>
|--|--2.json<br>
|--|--3.json<br>
|--|--4.json<br>
|--|--5.json<br>
|--r5 (folder)<br>
|--|--1.json<br>
|--|--2.json<br>
|--|--3.json<br>
|--|--4.json<br>
|--|--5.json<br>

## Instructions:

To run this file, stop.txt should be in the working directory.

Run the code from top to bottom to see the same results.

 - Section 1: Import libraries.

 - Section 2: Import data and extract content.

 - Sections 3-7: Define functions.

 - Sections 8: Show results (Top features, top sentences, features network)
 
# 2. LDA model
## Required files and environment:
 - Recommend complier: Jupyter notebook Version 4.4.0
 - Jupyter notebook file: LDA_Method.ipynb
 - Python file: LDA_Method.py
 - Data file: Json file (Royal commission)
=================================================================
Installation
* Library:
* nltk
* pandas
* gensim
* pyLDAvis
* py2neo
* matplotlib
* pprint
* re

=================================================================

Loading data
1. Download the 3 hearing data file 
2. Rename the folder name to r3, r4 and r5 respectively
3. Rename the json file in each folder to 1.json,...5.json


Inside the LDA_Method.ipynb or LDA_Method.py
1. Change the value of the variable called 'dir' to the path of data file
e.g. dir='/Users/shengyuan/Desktop/Study/CAPSTONE/RoyalCommission'

2. Change the value of the variable called 'round' to the round number you want to analyze
e.g round=3                                                   
=================================================================

Get the stop words
There are two ways to get the customized stop-words
1. myStops=getNameFromClassifier(data,path1,path2)

- Requirement
* Download the StandfordTagger Library from https://nlp.stanford.edu/software/tagger.shtml
* Inside the LDA_Method.ipynb or LDA_Method.py
* Change the value of the variable called 'path1' and 'path2' to the path of StandfordTagger
e.g. path1='/Users/shengyuan/desktop/study/Capstone/RoyalCommission/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz'
path2='/Users/shengyuan/desktop/study/Capstone/RoyalCommission/stanford-ner-2018-02-27/stanford-ner.jar'

2. myStops=getNameFromFile(round)
* Download the file called Round3name.txt, Round4name.txt and Round5name.txt 
* Place those files in the same folder of the code file

Attension: You just neec to execute one of above function
=================================================================
Getting the LDA performance
- Execution code
LDA_Analysis(data,myStops)

=================================================================

Getting the result by using optimal parameter
- Execution code
myResult,LDA_model, corpus_tfidf, dictionary=myLDA(data,myStops,n_topics,n_words,passes,iteration, top_n_sentence,top_n_article,graph)

Parameter
* Number of topic
* Number of words provided by a topic
* Passes
* Iteration
* Number of relevant article
* Number of key sentences for one article
e.g.
n_topics=8
n_words=10;
passes=1
iteration=50
top_n_sentence=2
top_n_article=4

Visualization

- Requirement
* Download the Neo4j from https://neo4j.com/
* Create a graph and set a password
* Give the password to the variable called 'graph'
e.g.graph = Graph(password="123")

- Execution code
lda_vis = pyLDAvis.gensim.prepare(LDA_model, corpus_tfidf, dictionary)
pyLDAvis.display(lda_vis)

* In the browser of Neo4j, type "Match(n) return n" for viewing the whole result
type "Match (n) detach delete n" for deleting the whole result
=================================================================
