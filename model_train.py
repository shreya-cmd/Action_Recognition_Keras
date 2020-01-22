import csv
from keras.utils import to_categorical
from processor import process_image
from extractorVgg16 import Extractor
import numpy as np
import glob
import os.path
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

with open('data_file.csv', 'r') as fin:
    reader = csv.reader(fin)
    data = list(reader)

classes = []
for item in data:
    if item[0] not in classes:
        classes.append(item[0])
        
classes = sorted(classes)
def get_class_one_hot(class_str):
    """return the class label in one hot encoded form."""
    # Encode it first.
    label_encoded = classes.index(class_str)

    # Now one-hot it.
    label_hot = to_categorical(label_encoded, len(classes))

    assert len(label_hot) == len(classes)

    return label_hot 

def build_image_sequence(frames):
    """Given a set of frames build the sequence."""
    return [process_image(x, (256,293,3)) for x in frames]

def get_frames_for_sample(sample):
    """Given a sample row from the data_file.csv, get all the corresponding image 
    framenames."""
    a = sample[1].split("/")
    path = os.path.join("data_frames", sample[0], a[2].replace('.avi',''))
    images = sorted(glob.glob(os.path.join(path) + '/*jpg'))
    return images

def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 20 and we have a list of size 40, return a new
    list of size 20 which is every 2nd element of the original list."""
    assert len(input_list) >= size
    
    # Get the number to skip between iterations.
    skip = len(input_list) // size
    
    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    return output[:size]

model = Extractor()

# Generate the dataset   
def get_all_sequences_in_memory(dataset):
    dataX, dataY = [] , []
    for row in dataset:
        frames = get_frames_for_sample(row)
        frames = rescale_list(frames, 20)
        sequence = []
        for img in frames:
            features = model.extract(img)
            sequence.append(features)
        dataX.append(sequence)
        dataY.append(get_class_one_hot(row[0]))
    return np.array(dataX), np.array(dataY)

dataX, dataY = get_all_sequences_in_memory(data)

model = Sequential()
model.add(Bidirectional(LSTM(4096, return_sequences=True,
               input_shape=(20,4096),
               dropout=0.5)))
model.add(Bidirectional(LSTM(4096)))
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.6))
model.add(Dense(4, activation='softmax'))


optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(optimizer = optimizer,
               loss = 'categorical_crossentropy', 
               metrics = ['accuracy'])

checkpointer = ModelCheckpoint(filepath="weights.hdf5", 
                       monitor = 'val_accuracy',
                       verbose=1, 
                       save_best_only=True)
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=2)
history = model.fit(X_train, y_train, batch_size=16, epochs = 50, verbose=2, callbacks=[checkpointer], 
                    validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=2)
print("test loss, test acc: : ", score)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('model_acc1.png', dpi=100)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
fig2 = plt.gcf()
plt.show()
plt.draw()
fig2.savefig('model_loss1.png', dpi=100)

# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")

"""We will convert the features X_train, X_test, y_train, y_test,
dataX, dataY in h5 file so that we dont have to extract them again 
for future use."""

import h5py

#for saving h5 files
h5f = h5py.File('dataX.h5', 'w')
h5f.create_dataset('dataX', data=dataX)
h5f.close()

h5f = h5py.File('dataY.h5', 'w')
h5f.create_dataset('dataY', data=dataY)
h5f.close()

h5f = h5py.File('X_train.h5', 'w')
h5f.create_dataset('X_train', data=X_train)
h5f.close()

h5f = h5py.File('X_test.h5', 'w')
h5f.create_dataset('X_test', data=X_test)
h5f.close()

h5f = h5py.File('y_train.h5', 'w')
h5f.create_dataset('y_train', data=y_train)
h5f.close()

h5f = h5py.File('y_test.h5', 'w')
h5f.create_dataset('y_test', data=y_test)
h5f.close()


############Steps for future use###############

#for loading h5 files
h5f = h5py.File('dataX.h5', 'r')
dataX = h5f['dataX'][:]
h5f.close()

h5f = h5py.File('dataY.h5', 'r')
dataY = h5f['dataY'][:]
h5f.close()

h5f = h5py.File('dataY.h5', 'r')
dataY = h5f['dataY'][:]
h5f.close()

h5f = h5py.File('X_train.h5', 'r')
X_train = h5f['X_train'][:]
h5f.close()

h5f = h5py.File('X_test.h5', 'r')
X_test = h5f['X_test'][:]
h5f.close()

h5f = h5py.File('y_train.h5', 'r')
y_train = h5f['y_train'][:]
h5f.close()

h5f = h5py.File('y_test.h5', 'r')
y_test = h5f['y_test'][:]
h5f.close()

from keras.models import load_model

#loading model
model = load_model('model.h5')

"""X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, 
                                test_size=0.5, random_state=2)"""

scores = model.evaluate(X_test, y_test)
print("test loss, test acc: : ", scores)