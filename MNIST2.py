from sklearn.datasets import fetch_mldata
import numpy as np

mnist = fetch_mldata('MNIST original')

data   = mnist.data / 255.0
labels = mnist.target.astype('int')


train_rank = 15000
test_rank = 100
#------- MNIST subset --------------------------
train_subset = np.random.choice(data.shape[0], train_rank)
test_subset = np.random.choice(data.shape[0], test_rank)

# train dataset
train_data = data[train_subset]
train_labels = labels[train_subset]

# test dataset
test_data = data[test_subset]
test_labels = labels[test_subset]

def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal

test = [3, 5, 9]
print to_categorical(test, 10)

# train and test to categorical
train_out = to_categorical(train_labels, 10)
test_out = to_categorical(test_labels, 10)

from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

# prepare model
model = Sequential()
model.add(Dense(70, input_dim=784))
model.add(Activation('relu'))
#model.add(Dense(50))
#model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('relu'))

# compile model with optimizer
sgd = SGD(lr=0.01, decay=0.000001, momentum=0.9,nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

# training
training = model.fit(train_data, train_out, nb_epoch=200, batch_size=400, verbose=1)
print training.history['loss'][-1]

# evaluate on test data
scores = model.evaluate(test_data, test_out, verbose=1)
print 'test', scores

# evaluate on train data
scores = model.evaluate(train_data, train_out, verbose=1)
print 'train', scores

import matplotlib.pyplot as plt  # za prikaz slika, grafika, itd.
#%matplotlib inline

imgN = 2
img = test_data[imgN]
print img.shape
img = img.reshape(28,28)

plt.imshow(img, cmap="Greys")
plt.show()

t = model.predict(test_data, verbose=1)
print t[imgN]

#%matplotlib inline
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.xticks(x)
width = 1/1.5
plt.bar(x, t[imgN], color="blue")

rez_t = t.argmax(axis=1)
print rez_t[imgN]

print sum((test_labels.T == rez_t.T)*1)

model.save("MINSTmodel.h5")