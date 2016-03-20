import pdb
import operator
import cPickle 
from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import pandas as pd 

import pdb 

import nltk 

try:
    import cochranenlp
    from cochranenlp.readers.biviewer import PDFBiViewer
except:
    print("cochrannlp not found!")

import keras
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import accuracy
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint


import gensim 
from gensim.models import Word2Vec

def load_trained_w2v_model(path="/Users/byron/dev/Deep-PICO/PubMed-w2v.bin"):
    m = Word2Vec.load_word2vec_format(path, binary=True)
    return m 


class PICOizer: 

    def __init__(self, filters=None, n_filters=50, dropout=0.0):
        ### some params for the CNN bit
        if filters is None:
            self.ngram_filters = [2, 3]
        else:
            self.ngram_filters = filters 
        self.nb_filter = n_filters 
        self.dropout = dropout

        self.df = pd.read_csv("ctg_pico_cuis.csv")
        self.build_interventions_set()
        print("loading embeddings...")
        wvs = load_trained_w2v_model()
        print("done.")
        # 350 is for abstract length; seems OK
        self.preprocessor = Preprocessor(10000, 350, wvs=wvs)
        self.preprocessor.preprocess(self.df["abstract"])
        print("constructing X,Y...")
        self.X_Y()
        print("done.")
        print("building model...")
        self.build_model()
        print("ok.")

    def train(self, batch_size=32, nb_epoch=5):
        #self.build_model()
        self.model.compile(loss={'output': 'binary_crossentropy'}, 
                                optimizer='adam')
        self.model.fit({'input': self.X, 'output': self.Y},
                batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)

                #verbose=2, callbacks=[checkpointer])

    def build_interventions_set(self, top_k=1000):
        #interventions = self.df["interventions_concept_names"]
        interventions = self.df["interventions_cuis"]
        interventions_set = []
        intervention_counts = defaultdict(int) # init to zero counts
        self.interventions_lists = [] # keep list for each abstract
        for intervention_list in interventions.values:
            if not pd.isnull(intervention_list):
                intervention_list_split = intervention_list.split("|")
                self.interventions_lists.append(intervention_list_split)
                for intervention in intervention_list_split:
                    interventions_set.append(intervention)
                    intervention_counts[intervention] += 1
            else: 
                self.interventions_lists.append([])

        # sort by count
        sorted_interventions = sorted(intervention_counts.items(), 
                                        key=operator.itemgetter(1), 
                                        reverse=True)
        top_k_interventions  = [t[0] for t in sorted_interventions[:top_k]]

        self.interventions_set = [intervention for intervention in list(set(interventions_set))
                                    if intervention in top_k_interventions]
        self.num_interventions = len(self.interventions_set)

        # map from names to indices, for convienence
        self.interventions_to_indices = {}
        for i, intervention in enumerate(self.interventions_set):
            self.interventions_to_indices[intervention] = i 


    def X_Y(self):
        print("saving last 50 for testing!") ### TMP TMP TMP just for playing around
        self.X = self.preprocessor.build_sequences(self.df.abstract)#[:-50]
        self.Y = np.zeros((self.X.shape[0], self.num_interventions), dtype=np.bool)
        for i in range(self.X.shape[0]):
            for intervention in self.interventions_lists[i]:
                try:
                    intervention_idx = self.interventions_to_indices[intervention]
                    self.Y[i, intervention_idx] = 1
                except:
                    # this intervention not in our set
                    pass 

        print "X shape: %s; Y shape: %s" % (self.X.shape, self.Y.shape)


    def build_model(self):
        self.model = Graph()
        self.model.add_input(name='input', input_shape=(self.preprocessor.maxlen,), dtype=int)

        self.model.add_node(Embedding(self.preprocessor.max_features, self.preprocessor.embedding_dims, 
                                input_length=self.preprocessor.maxlen, weights=self.preprocessor.init_vectors), 
                                name='embedding', input='input')
        self.model.add_node(Dropout(self.dropout), name='dropout_embedding', input='embedding')
        for n_gram in self.ngram_filters:
            self.model.add_node(Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=n_gram,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=self.preprocessor.embedding_dims,
                                         input_length=self.preprocessor.maxlen),
                           name='conv_' + str(n_gram),
                           input='dropout_embedding')
            self.model.add_node(MaxPooling1D(pool_length=self.preprocessor.maxlen - n_gram + 1),
                           name='maxpool_' + str(n_gram),
                           input='conv_' + str(n_gram))
            self.model.add_node(Flatten(),
                           name='flat_' + str(n_gram),
                           input='maxpool_' + str(n_gram))
        self.model.add_node(Dropout(self.dropout), name='dropout', inputs=['flat_' + str(n) for n in self.ngram_filters])
        self.model.add_node(Dense(self.num_interventions, 
                                  input_dim=self.nb_filter * len(self.ngram_filters)), 
                                  name='dense', input='dropout')
        self.model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
        self.model.add_output(name='output', input='sigmoid')
        print("model built")
        print(self.model.summary())
        self.model.compile(loss={'output': 'binary_crossentropy'}, 
                            optimizer='adam')

class Preprocessor:
    def __init__(self, max_features, maxlen, embedding_dims=200, wvs=None):
        '''
        max_features: the upper bound to be placed on the vocabulary size.
        maxlen: the maximum length (in terms of tokens) of the instances/texts.
        embedding_dims: size of the token embeddings; over-ridden if pre-trained
                          vectors is provided (if wvs is not None).
        wvs: pre-trained embeddings (for embeddings initialization)
        '''

        self.max_features = max_features  
        self.tokenizer = Tokenizer(nb_words=self.max_features)
        self.maxlen = maxlen  

        self.use_pretrained_embeddings = False 
        self.init_vectors = None 
        if wvs is None:
            self.embedding_dims = embedding_dims
        else:
            # note that these are only for initialization;
            # they will be tuned!
            self.use_pretrained_embeddings = True
            self.embedding_dims = wvs.vector_size
            self.word_embeddings = wvs


    def preprocess(self, all_texts):
        ''' 
        This fits tokenizer and builds up input vectors (X) from the list 
        of texts in all_texts. Needs to be called before train!
        '''
        self.raw_texts = all_texts
        self.fit_tokenizer()
        if self.use_pretrained_embeddings:
            self.init_word_vectors()

    def fit_tokenizer(self):
        ''' Fits tokenizer to all raw texts; remembers indices->words mappings. '''
        self.tokenizer.fit_on_texts(self.raw_texts)
        self.word_indices_to_words = {}
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token

    def build_sequences(self, texts):
        X = list(self.tokenizer.texts_to_sequences_generator(texts))
        X = np.array(pad_sequences(X, maxlen=self.maxlen))
        return X

    def init_word_vectors(self):
        ''' 
        Initialize word vectors.
        '''
        self.init_vectors = []
        unknown_words_to_vecs = {}
        for t, token_idx in self.tokenizer.word_index.items():
            if token_idx <= self.max_features:
                try:
                    self.init_vectors.append(self.word_embeddings[t])
                except:
                    if t not in unknown_words_to_vecs:
                        # randomly initialize
                        unknown_words_to_vecs[t] = np.random.random(
                                                self.embedding_dims)*-2 + 1

                    self.init_vectors.append(unknown_words_to_vecs[t])

        # note that we make this a singleton list because that's
        # what Keras wants. 
        self.init_vectors = [np.vstack(self.init_vectors)]



'''
X_test = p.X[-100:]
preds = m.predict({'input': X_test})['output']

''' 

def multi_label_precision_and_recall(y, y_hat):
    # make sure y is numerical (0/1) rather than bool
    y = y.astype("int")
    # and round predictions to 0/1
    y_hat = np.round(y_hat)


    tp_v = y * y_hat

    precisions = np.zeros(y.shape[1])
    recalls = np.zeros(y.shape[1])
    for j in range(y.shape[1]):
        total_pos_preds = np.sum(y_hat[:,j])
        true_pos_preds  = np.sum(tp_v[:,j])
        if total_pos_preds == 0:
            precisions[j] = np.nan
        else:
            precisions[j] = float(true_pos_preds) / float(total_pos_preds)

        total_true_pos  = np.sum(y[:,j])
        if total_true_pos == 0:
            recalls[j] = np.nan
        else:
            recalls[j] = float(true_pos_preds) / float(total_true_pos)

    #averaged = 0
    #for prec, rec in zip(precisions, recalls):
    #    averaged = 
    return precisions, recalls 





if __name__ == '__main__':
    picoizer = PICOizer(dropout=.2)
    
    checkpointer = ModelCheckpoint(filepath="PICO_CUIs_weights.hdf5", verbose=2)

    json_string = picoizer.model.to_json()
    open('PICO_CUIs_model_architecture.json', 'w').write(json_string)
    print("dumped model!")


    self.model.fit({'input': picoizer.X[:-100], 'output': picoizer.Y[:-100]},
            batch_size=32, nb_epoch=20, verbose=2, callbacks=[checkpointer])

    '''
    picoizer.model.fit(picoizer.X, picoizer.Y, 
        batch_size=32, nb_epoch=40, 
        verbose=2, callbacks=[checkpointer])
    '''
    #while epochs_so_far < total_epochs:
    #cnn.train(train_X, y_train, nb_epochs=epochs_per_iter)#, X_val=test_X, y_val=y_test)
        

        
    #epochs_so_far += epochs_per_iter
    
    #yhat = cnn.predict(test_X, binarize=True)
    
    #print("acc @ epoch %s: %s" % (epochs_so_far, accuracy_score(y_test, yhat)))

