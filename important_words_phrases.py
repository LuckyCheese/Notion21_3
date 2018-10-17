import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression


hf = h5py.File('input_data4.h5')
X_train_vectorised = hf.get('input_data4').value

features = []

# open file and read the content in a list
with open('feature4.txt', 'r') as file:  
    for line in file:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        features.append(currentPlace)


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
feature_names = features
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


    
