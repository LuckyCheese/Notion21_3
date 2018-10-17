
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras import optimizers

    
def autoencoder(input_data, act1, act2):    #different activation for output layer

    dim = input_data.shape[1]
    x = Input(shape=(dim,))
    print(x)
    h = x

    #encoder
    #input layer
    h = Dense(500, activation=act1)(h)
    #hidden layers
    h = Dense(500, activation=act1)(h)
    h = Dense(2000, activation=act1)(h)
    #output layer
    h = Dense(10, activation=act2)(h)        #encoder output. reduced dimensions.

    y = h            

    #decoder
    #input layer
    y = Dense(2000, activation=act1)(y)
    #hidden layers
    y = Dense(500, activation=act1)(y)
    y = Dense(500, activation=act1)(y)
    #output layer
    y = Dense(dim, activation=act2)(y)      #decoder output. reconstructed input.         
    
    return Model(inputs = x, outputs = y), Model(inputs = x, outputs = h)


#Loading reuters data
def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y

x, y = load_reuters()
input_data = x[0:1000,]
print(input_data.shape)

#sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
auto_encoder, encoder = autoencoder(input_data,'relu','relu')
auto_encoder.compile(optimizer='adam', loss='mse')
auto_encoder.fit(input_data, input_data, epochs = 400, batch_size = 25)


encoded_data = encoder.predict(x)
decoded_data = auto_encoder.predict(x)


