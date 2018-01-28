## LSTM model in keras to learn on training data.

from keras.layers import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers import Embedding

from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import os
import matplotlib.pyplot as plt

maxlen = 0
word_freq = collections.Counter()
num_recs = 0
path = os.path.join(os.getcwd(), "data/training.txt")
ftrain = open(path)

#Vocabulary creation. to be used in word2index and Embedding
for line in ftrain:
    #print(line.strip().split("\t"))
    label, sentence = line.strip().split("\t")
    words = nltk.word_tokenize(sentence.lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freq[word] += 1
    num_recs += 1   
ftrain.close()

MAX_FEATURES = 2000
MAX_SEQ_LEN = 40

#0 for padding and 1 for words not present in vocabulary.
vocab_size = min(MAX_FEATURES, len(word_freq)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freq.most_common(MAX_FEATURES))}
word2index['PAD'] = 0
word2index['UNK'] = 1
index2word = {value:key for key, value in word2index.items()}

#Creating data for train test
X = np.empty((num_recs, ), dtype=list) 
y = np.zeros((num_recs,))
i = 0
ftrain = open(path)
for line in ftrain:
    label, sentence = line.strip().split("\t")
    words = nltk.word_tokenize(sentence.lower())
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index['UNK'])
    X[i] = seqs
    y[i] = np.float(label)
    i+=1
ftrain.close()
X = sequence.pad_sequences(X, maxlen=MAX_SEQ_LEN)

# Creating LSTM model.
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=40))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

#Train
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1)
model.evaluate(X_test, y_test)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("semtiment_LSTM.h5")
print("Saved model to disk")
