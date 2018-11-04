mport h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import fcluster
import nltk
import pandas as pd
import networkx as nx


def get_tokens(data):
    items = ['NN', 'JJ', 'VB', 'RB']
    token_list=[]
    word_list = []
    token = nltk.word_tokenize(data)
    x = nltk.pos_tag(token)       
    token_list.append(x)
    grammar = lambda pos: pos[:2] in items
    words = [t for t,pos in x if grammar(pos)]
    word_list.append(words)

    return token_list, word_list
    

def get_phrases(token_list):
    noun1 = "NP1: {<JJ.*>+<NN.*>}"
    noun2 = "NP2: {<NN.*>+<NN.*>}"
    verb1 = "VP1: {<VB.*>+<NN.*>}"
    verb2 = "VP2: {<VB.*>+<DT>+<NN.*>}"
    verb3 = "VP3: {<VB.*>+<TO>+<DT><NN.*>}"
    verb4 = "VP4: {<VB.*>+<TO>+<NN.*>}"
    verb5 = "VP5: {<VB.*>+<IN>+<DT>+<NN.*>}"
    adj1 = "ADJP1: {<JJ.*>+<JJ.*>}"
    adj2 = "ADJP2: {<RB.*>+<JJ.*>}"
    adv = "ADVP: {<RB.*>+<VB.*>}"
    #prep1 = "PREP1: {<IN>+<DT>+<NN.*>}"
    #prep2 = "PREP2: {<IN>+<NN.*>}"
    grammar = [noun1, noun2, verb1, verb2, verb3, verb4, adj1, adj2, adv] #prep1, prep2]
    phrase_list = []
    for tokens in token_list:
        phrase = []
        for g in grammar:
            cp = nltk.RegexpParser(g)
            result=cp.parse(tokens)
            phrase.append(result)
            #print(type(result))
            #print([w for w,pos in result.pos() if (pos == 'PREP')])

        grammar1 = ['NP1', 'NP2', 'VP1', 'VP2', 'VP3', 'VP4', 'VP5', 'ADJP1', 'ADJP2', 'ADVP'] #, 'PREP1', 'PREP2']
        ps = []
        Ps = [(p.subtrees(filter=lambda x: x.label() in grammar1)) for p in phrase]#got subtrees

        for i in range(0,len(Ps)):
            ph = [p for p in Ps[i]]
            ps.extend([w for w in ph])

        p = []
        p.extend([ps[i] for i in range(0,len(ps))])
        phrases = []
        for i in range(0,len(p)):
            phrases.append(' '.join([w for w, t in p[i].leaves()]))

        #phrase_list.append(phrases)
    #print(len(phrases))
    #print(phrase_list[660])
    return phrases


def create_clusters(num_of_clusters):
#    plt.title('Hierarchical Clustering Dendrogram (truncated)')
#    dendrogram(
#            linkage_matrix, 
#            orientation="right",
#            truncate_mode='lastp',  # show only the last p merged clusters
#            p=num_of_clusters,  # show only the last p merged clusters
#            leaf_font_size=12.,
#            show_contracted=True,  # to get a distribution impression in truncated branches
#    )
#    plt.show()
    
    clusters = fcluster(linkage_matrix, num_of_clusters, criterion='maxclust')
    #print(clusters)
    print(len(clusters))
    
    return clusters


def get_top_features(feature_names, model):
    class_labels = model.classes_
    
    top_features_list = []
    top_coef_list = []
    for i, class_label in enumerate(class_labels):
        top_features = []
        top_coef = []
        top5 = np.argsort(model.coef_[i])[-5:] # Most informative: positive
        for j in top5:
          top_features.append(feature_names[j])
          top_coef.append(model.coef_[i][j])
        
        top_features_list.append(top_features)
        top_coef_list.append(top_coef)
          
        print("%s: %s" % (class_label, ", ".join(feature_names[j] for j in top5)))
        #the last word is the most informative one
        
        #top50 = np.argsort((-1) * model.coef_[i])[-50:] # Most informative: negative
        #print("%s: %s" % (class_label, ", ".join(feature_names[j] for j in top50)))
        #the last word is the most informative one
        #bottom50 = np.argsort(abs(model.coef_[i]))[:50] # Most irrelevant. Bottom 50 of the absolute values of coefficients 
        #print("%s: %s" % (class_label, ", ".join(feature_names[j] for j in bottom50)))
        #the last word is the least informative one
  
    return top_features_list, top_coef_list


def get_top_features_binary(feature_names, model):
    class_labels = model.classes_
    top_features_list = []
    top_features_list = []
    top_coef_list = []
    
    sorted_coef_index = model.coef_[0].argsort()
    print("1: %s" % ", ".join(feature_names[i] for i in sorted_coef_index[:5]))
    print("2: %s" % ", ".join(feature_names[i] for i in sorted_coef_index[:-6:-1]))

    top_features_1 = []
    top_coef_1 = []
    for i in sorted_coef_index[:50]:
        top_features_1.append(feature_names[i])
        top_coef_1.append(model.coef_[0][i])
    
    top_features_2 = []
    top_coef_2 = []
    for i in sorted_coef_index[:-6:-1]:
        top_features_2.append(feature_names[i])
        top_coef_2.append(model.coef_[0][i])
     
    top_features_list.append(top_features_1)
    top_features_list.append(top_features_2)
    top_coef_list.append(top_coef_1)
    top_coef_list.append(top_coef_2)
    
    return top_features_list, top_coef_list


def get_doc_list(clusters):
    cluster_dict = {}
    for i , cluster in enumerate(clusters):
        cluster_dict.setdefault(cluster, [])
        cluster_dict[cluster].append(i)
    return cluster_dict

    
def get_concordance(word, textlist):
    """
    Print out the concordance of a word in a list of text
    """
    for text in textlist:
        
        ph, tokens = get_tokens(text)
        phrases = get_phrases(ph)
        ci = nltk.ConcordanceIndex(phrases)
        if ci.offsets(word):
            ci.print_concordance(word)


def get_model(clusters):
    
    model = LogisticRegression()
    return model.fit(X_train_vectorised, clusters)

    
    


hf = h5py.File('input_data3.h5')
X_train_vectorised = hf.get('input_data3').value

features = []

# open file and read the content in a list
with open('feature3.txt', 'r') as file:  
    for line in file:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        features.append(currentPlace)



dist = 1 - cosine_similarity(X_train_vectorised)


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

c = list(range(2, 14))
clusters = [create_clusters(cl) for cl in c]
# Logistic regression
models = [get_model(cluster) for cluster in clusters]


feature_names = features
model_2_topfeatures, model_2_topcoef = get_top_features_binary(feature_names, models[0])
model_3_topfeatures, model_3_topcoef = get_top_features(feature_names, models[1])
model_4_topfeatures, model_4_topcoef = get_top_features(feature_names, models[2])
model_5_topfeatures, model_5_topcoef = get_top_features(feature_names, models[3])
model_6_topfeatures, model_6_topcoef = get_top_features(feature_names, models[4])
model_7_topfeatures, model_7_topcoef = get_top_features(feature_names, models[5])
model_8_topfeatures, model_8_topcoef = get_top_features(feature_names, models[6])
model_9_topfeatures, model_9_topcoef = get_top_features(feature_names, models[7])
model_10_topfeatures, model_10_topcoef = get_top_features(feature_names, models[8])
model_11_topfeatures, model_11_topcoef = get_top_features(feature_names, models[9])
model_12_topfeatures, model_12_topcoef = get_top_features(feature_names, models[10])
model_13_topfeatures, model_13_topcoef = get_top_features(feature_names, models[11])

doc_list_13 = get_doc_list(clusters[11])


with open('clean_text.txt', 'r') as file:
    text = []
    for t in file:
        t = re.sub("\n","",t)
        text.append(t)
                
clean_content = text

       
for i in range(13):
    #print("Cluster " + str(i+1) + "\n")
    textlist_per_cluster = []
    for index in doc_list_13[i+1]:
        textlist_per_cluster.append(clean_content[index])
    #print(textlist_per_cluster[0])
        
    for j in range(5):
        feature = model_13_topfeatures[i][j]
        #print("\n Getting concordance for: " + feature + "\n")
        #get_concordance(feature, textlist_per_cluster)
    #print("\n\n")



graph_list = []
root = 'R3'

node1 = ''
for i in range(5):
    temp = model_2_topfeatures[0][i] + '\n'
    node1 = node1 + temp
    
print(node1)

graph_list.append((root, node1))

node2 = ''
for i in range(5):
    temp = model_2_topfeatures[1][i] + '\n'
    node2 = node2 + temp
    
print(node2)

graph_list.append((node1, node2))

node3 = ''
for i in range(5):
    temp = model_3_topfeatures[1][i] + '\n'
    node3 = node3 + temp
    
print(node3)

graph_list.append((node1, node3))

node4 = ''
for i in range(5):
    temp = model_3_topfeatures[2][i] + '\n'
    node4 = node4 + temp
    
print(node4)

graph_list.append((node3, node4))

node5 = ''
for i in range(5):
    temp = model_4_topfeatures[2][i] + '\n'
    node5 = node5 + temp
    
print(node5)

graph_list.append((node3, node5))

node6 = ''
for i in range(5):
    temp = model_8_topfeatures[2][i] + '\n'
    node6 = node6 + temp
    
print(node6)

graph_list.append((node5, node6))

node7 = ''
for i in range(5):
    temp = model_8_topfeatures[3][i] + '\n'
    node7 = node7 + temp
    
print(node7)

graph_list.append((node5, node7))


node8 = ''
for i in range(5):
    temp = model_9_topfeatures[3][i] + '\n'
    node8 = node8 + temp
    
print(node8)

graph_list.append((root, node8))

node9 = ''
for i in range(5):
    temp = model_9_topfeatures[4][i] + '\n'
    node9 = node9 + temp
    
print(node9)

graph_list.append((node8, node9))

node10 = ''
for i in range(5):
    temp = model_4_topfeatures[3][i] + '\n'
    node10 = node10 + temp
    
print(node10)

graph_list.append((node8, node10))


node11 = ''
for i in range(5):
    temp = model_5_topfeatures[3][i] + '\n'
    node11 = node11 + temp
    
print(node11)

graph_list.append((node10, node11))


node12 = ''
for i in range(5):
    temp = model_5_topfeatures[4][i] + '\n'
    node12 = node12 + temp
    
print(node12)

graph_list.append((node11, node12))


node13 = ''
for i in range(5):
    temp = model_6_topfeatures[4][i] + '\n'
    node13 = node13 + temp
    
print(node13)

graph_list.append((node12, node13))

node14 = ''
for i in range(5):
    temp = model_6_topfeatures[5][i] + '\n'
    node14 = node14 + temp
    
print(node14)

graph_list.append((node12, node14))

node15 = ''
for i in range(5):
    temp = model_7_topfeatures[5][i] + '\n'
    node15 = node15 + temp
    
print(node15)

graph_list.append((node11, node15))

node16 = ''
for i in range(5):
    temp = model_7_topfeatures[6][i] + '\n'
    node16 = node16 + temp
    
print(node16)

graph_list.append((node15, node16))

node17 = ''
for i in range(5):
    temp = model_10_topfeatures[8][i] + '\n'
    node17 = node17 + temp
    
print(node17)

graph_list.append((node15, node17))

node18 = ''
for i in range(5):
    temp = model_10_topfeatures[9][i] + '\n'
    node18 = node18 + temp
    
print(node18)

graph_list.append((node10, node18))

node19 = ''
for i in range(5):
    temp = model_11_topfeatures[9][i] + '\n'
    node19 = node19 + temp
    
print(node19)

graph_list.append((node18, node19))


node20 = ''
for i in range(5):
    temp = model_11_topfeatures[10][i] + '\n'
    node20 = node20 + temp
    
print(node20)

graph_list.append((node18, node20))

node21 = ''
for i in range(5):
    temp = model_12_topfeatures[10][i] + '\n'
    node21 = node21 + temp
    
print(node21)

graph_list.append((node20, node21))

node22 = ''
for i in range(5):
    temp = model_12_topfeatures[11][i] + '\n'
    node22 = node22 + temp
    
print(node22)

graph_list.append((node20, node22))

node23 = ''
for i in range(5):
    temp = model_13_topfeatures[12][i] + '\n'
    node23 = node23 + temp
    
print(node23)

graph_list.append((node22, node23))

node24 = ''
for i in range(5):
    temp = model_13_topfeatures[11][i] + '\n'
    node24 = node24 + temp
    
print(node24)

graph_list.append((node22, node24))


####create node edge list based on this model's dendogram
df = pd.DataFrame({ 'from':[root, root, node2, node2, node4, node4, node5, node5, node7, node7, node10, node10, node12,
                            node12, node14, node14, node16, node16, node18, node18, node20, node20, node22, node22], 
                   'to':[node1, node2, node3, node4, node5, node10, node6, node7, node8, node9, node11, node12, node13,
                         node14, node15, node16, node17, node18, node19, node20, node21, node22, node23, node24]}) 

# And a data frame with characteristics for your nodes
carac = pd.DataFrame({ 'ID':[root, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12,
                             node13, node14, node15, node16, node17, node18, node19, node20, node21, node22, node23, node24], 
                      'myvalue':['group1','group1','group2','group2','group2', 'group2', 'group2', 'group2', 'group1', 'group3', 'group4', 'group4','group4','group4',
                                'group4','group4','group4','group4','group4','group4','group4','group4','group4','group4','group4'] })


# Build your graph
G=nx.from_pandas_edgelist(df, 'from', 'to')

print(len(G.nodes()))
print(G.nodes())
print(len(G.edges()))


#carac= carac.set_index('ID')
#carac=carac.reindex(G.nodes())
#carac['myvalue']=pd.Categorical(carac['myvalue'])
#carac1['myvalue'].cat.codes

# Plot it
# Graph with Custom nodes:
plt.figure(1, figsize = (20, 20))
nx.draw(G, with_labels=True, node_size=300, font_size=7, node_color = 'skyblue', font_weight="bold", node_shape="s", alpha=0.4, linewidths=40, pos=nx.kamada_kawai_layout(G)) #, node_color=carac['myvalue'].cat.codes
plt.show()


    
