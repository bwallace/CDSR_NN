import pdb
import operator

import cPickle 

import numpy as np
import pandas as pd 

#import sklearn 
#from sklearn.feature_extraction.text import CountVectorizer

import nltk 

try:
    import cochranenlp
    from cochranenlp.readers.biviewer import PDFBiViewer
except:
    print("cochrannlp not found!")

import keras
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge, Activation, RepeatVector, TimeDistributedDense
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.layers.recurrent import LSTM, GRU

import gensim 
from gensim.models import Word2Vec

START_STR = "startstartstart"
STOP_STR  = "stopstopstop"

'''
def _get_init_vectors(vectorizer, wv, unknown_words_to_vecs):
    init_vectors = []
    for token_idx, t in enumerate(vectorizer.vocabulary):
        try:
            init_vectors.append(wv[t])
        except:
            init_vectors.append(unknown_words_to_vecs[t])
    init_vectors = np.vstack(init_vectors)
    return init_vectors
'''

def load_trained_w2v_model(path="/Users/byron/dev/Deep-PICO/PubMed-w2v.bin"):
    m = Word2Vec.load_word2vec_format(path, binary=True)
    return m 


def get_docs_and_intervention_summaries(pico_elem_str="CHAR_INTERVENTIONS"):
    pairs = []
    p = PDFBiViewer()
    for study in p: 
        cdsr_entry = study.cochrane 
        text = study.studypdf['text']
        intervention_text = cdsr_entry["CHARACTERISTICS"][pico_elem_str]
        if intervention_text is not None:
            #pairs.append((nltk.word_tokenize(text), 
            #              nltk.word_tokenize(intervention_text)))
            pairs.append((text_to_word_sequence(text), 
                          text_to_word_sequence(intervention_text)))

    return pairs 

class ISummarizer:

    # 100000
    def __init__(self, pairs, nb_words=10000, hidden_size=64, max_input_size=5000, max_output_size=50):
        self.pairs = pairs 
        self.nb_words = nb_words + 2 # number of words; +2 for start and stop tokens!
        self.max_input_size = max_input_size
        self.max_output_size = max_output_size + 2 # again +2 for start/stop

        self.hidden_size = hidden_size
        print("loading pre-trained word vectors...")
        self.wv = load_trained_w2v_model()
        # here you want to add start and stop 
        print("OK!")
        self.word_embedding_size = self.wv.vector_size 

        # call to sequences
        # call init_word_vectors
        print("building sequences...")
        self.build_sequences()

        print("initializing word vectors...")
        self.init_word_vectors()

        print("ok!")

    def build_sequences(self):
        self.tokenizer = Tokenizer(nb_words=self.nb_words)

        self.raw_input_texts  = [START_STR + " " + " ".join(pair[0]) + " " + STOP_STR for pair in self.pairs]
        self.raw_output_texts = [START_STR + " " + " ".join(pair[1]) + " " + STOP_STR for pair in self.pairs]

        def _get_max(seqs):
            return max([len(seq) for seq in seqs])

        self.tokenizer.fit_on_texts(self.raw_input_texts+self.raw_output_texts)
        self.input_sequences  = list(self.tokenizer.texts_to_sequences_generator(self.raw_input_texts))
        #self.max_input_len    = _get_max(self.input_sequences)
        #X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        #X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        self.input_sequences = list(pad_sequences(self.input_sequences, maxlen=self.max_input_size))

        self.output_sequences = list(self.tokenizer.texts_to_sequences_generator(self.raw_output_texts))
        self.output_sequences = list(pad_sequences(self.output_sequences, maxlen=self.max_output_size))
        
    def init_word_vectors(self):
        self.init_vectors = []
        unknown_words_to_vecs = {}
        for t, token_idx in self.tokenizer.word_index.items():
            if token_idx <= self.nb_words:
                try:
                    self.init_vectors.append(self.wv[t])
                except:
                    if t not in unknown_words_to_vecs:
                        # randomly initialize
                        unknown_words_to_vecs[t] = np.random.random(
                                                self.word_embedding_size)*-2 + 1

                    self.init_vectors.append(unknown_words_to_vecs[t])

        self.init_vectors = np.vstack(self.init_vectors)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.nb_words, self.word_embedding_size, weights=[self.init_vectors]))
        ### 
        # run embeddings through a Gated Recurrent Unit
        self.model.add(GRU(self.hidden_size))
        self.model.add(Dense(self.hidden_size))
        self.model.add(Activation('relu'))
        self.model.add(RepeatVector(self.max_output_size))
        self.model.add(GRU(self.hidden_size, return_sequences=True))
        self.model.add(TimeDistributedDense(self.nb_words, activation="softmax"))
        # does cross entropy make sense here?
        self.model.compile(loss="categorical_crossentropy", optimizer='adam')
        return self.model 

    def X_y(self):
        self.X = np.array(self.input_sequences) # np.zeros((n, self.max_input_size, self.nb_words), dtype=np.bool)
        self.Y = np.zeros((len(self.output_sequences), self.max_output_size, self.nb_words), dtype=np.bool)
        for i in range(self.X.shape[0]):
            #for j, token_idx in enumerate(self.input_sequences[i]):
            #    self.X[i, j, token_idx] = 1

            for j, token_idx in enumerate(self.output_sequences[i]):
                self.Y[i, j, token_idx] = 1

        print "X shape: %s; Y shape: %s" % (self.X.shape, self.Y.shape)

    def train(self):
        # @TODO revisit; batchsize, etc
        print "fitting model..."
        self.model.fit(self.X, self.Y)

def all_systems_go():
    from keras.callbacks import ModelCheckpoint
    pairs = cPickle.load(open("pairs.pickle"))
    
    IS = ISummarizer(pairs)
    model = IS.build_model()
    IS.X_y()

    n_train = 12751

    X_train, Y_train = IS.X[:n_train], IS.Y[:n_train]
    X_test, Y_test   = IS.X[n_train:], IS.Y[n_train:]

    print "ok... fitting ..."
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=25, verbose=0, 
                show_accuracy=True,
                validation_data=(X_test, Y_test), callbacks=[checkpointer])





'''
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#saves the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
'''

def toy():
    nb_w = 10
    embedding_size = 5

    #embeddings = [np.random.random(5) for _ in range(10)]

    X = [[9, 8, 3], [4, 5, 1], [3, 6, 3], [6,2,1]]
    y = [[1, 2], [3,1], [1,6], [5,2]]


    X_arr  = np.array(X) #np.zeros((len(X), len(X[0]), 10)) # n sample; time steps; vocab size
    y_tens = np.zeros((len(y), len(y[0]), 10)) # n sample; time steps; vocab size

    for i in range(len(X)):
        #for j, t_idx in enumerate(X[i]):
        #    X_tens[i, j, t_idx] = 1

        for j, t_idx in enumerate(y[i]):
            y_tens[i, j, t_idx] = 1
        

    model = Sequential()
    model.add(Embedding(10, embedding_size))

    max_output_size = 2
    hidden_size = 8

    ### 
    # I think we only want the output dim here!
    #model.add(GRU(self.word_embedding_size, self.hidden_size))
    model.add(GRU(hidden_size))
    model.add(Dense(hidden_size))

    model.add(Activation('relu'))

    model.add(RepeatVector(max_output_size))
    model.add(GRU(hidden_size, return_sequences=True))
    model.add(TimeDistributedDense(nb_w, activation="softmax"))

    model.compile(loss='mse', optimizer='adam')
    
    model.fit(X_arr, y_tens)


if __name__ == "__main__":
    all_systems_go()

'''
module load cuda
module load python
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
'''

'''
batch_size = 16
embedding_size = 32
hidden_size = 512

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_size))
model.add(GRU(embedding_size, hidden_size))
model.add(Dense(hidden_size, hidden_size))
model.add(Activation('relu'))
model.add(RepeatVector(maxlen))
model.add(GRU(hidden_size, hidden_size, return_sequences=True))
model.add(TimeDistributedDense(hidden_size, max_features, activation="softmax"))
'''
