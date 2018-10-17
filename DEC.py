
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ae import autoencoder

class ClusteringLayer(Layer):

    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha


    def build(self, input_shape): #weight shape = (n_clusters, input_dim) = (20,10)
        assert len(input_shape) == 2          #input_dim = 10 (encoder o/p dim set in autoencoder)
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        self.built = True       

    def call(self, inputs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DEC(object):

    def __init__(self, input_data, n_clusters=50, alpha=1):

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(input_data, 'relu', 'relu') #info about the Model i/p & o/p

        clustering_layer = ClusteringLayer(self.n_clusters, name = 'clustering')(self.encoder.output)  #Clustering Layer(Kernel) x (encoder output)
        self.model = Model(inputs = self.encoder.input, outputs = clustering_layer)


    def feature_extraction(self, input_data, optimizer, epochs, batch_size):

        self.autoencoder.compile(optimizer=optimizer, loss = 'mse')
        self.autoencoder.fit(input_data, input_data, epochs=epochs, batch_size = batch_size) #fitting autoencoder
        return self.encoder.predict(input_data)  #predict using autoencoder

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T


    def predict(self, input_data):
        q = self.model.predict(input_data, verbose=0)
        return q.argmax(1)

        
    def compile(self, optimizer = 'sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)


    def fit(self, input_data, reduced_data, max_iter=1000, batch_size=25, tol = 1e-3, update_interval=100):

        #initialise cluster centres using kmeans
        kmeans = KMeans(n_clusters = self.n_clusters, n_init = 15)
        y_pred = kmeans.fit_predict(reduced_data)
        y_pred_last = np.copy(y_pred)
        print('The shape of kmeans cluster centres is: ',kmeans.cluster_centers_.shape)
        self.model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])
        
        index = 0
        index_array = np.arange(input_data.shape[0])
        for i in range(max_iter):
            if i % update_interval == 0:

                q = self.model.predict(input_data, verbose = 0)
                p = self.target_distribution(q)

                y_pred = q.argmax(1)
                
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if i > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            x = input_data
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            return y_pred

        
def pred_dict(y_pred):
    from collections import Counter
    pred_count = Counter(y_pred)
    keylist = sorted(pred_count.keys())
    pred_dict = {}
    for key in keylist:
        pred_dict[key] = pred_count[key]
        #print "%s: %s" % (key, pred_count[key])
    print(pred_dict)

    
if __name__ == "__main__":

    #load vectorised data 
    from loaddata_raw import load_data, pre_processing
    dir='/Users/preronamajumder/Documents/COMP5703/data/' #data fold path
    args = 'content'  #data type
    max_words = 2000  #dimensons 
    round = 3   #select which hearing

    data = load_data(dir,round,args)
    data_matrix = pre_processing(data,max_words)   #return a numpy array
    print('The shape of the feature space is:',data_matrix.shape)
    #input_data = data_matrix
    input_data = StandardScaler().fit_transform(data_matrix)

    #DEC Model
    
    dec = DEC((input_data))
    reduced_data = dec.feature_extraction(input_data, optimizer='adam', epochs = 500, batch_size=50)
    print('The shape of the reduced feature space is:',reduced_data.shape)
    dec.compile(optimizer = SGD(0.01, 0.7), loss = 'kld')
    
    y_pred = dec.fit(input_data, reduced_data)

    print('The shape of prediction is: ',y_pred.shape)

    pred_dict(y_pred)
