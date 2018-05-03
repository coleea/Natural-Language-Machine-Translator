# -*- coding: utf-8 -*-
__original_author = "Haizou Qu"
__modifier__ = "Lee Guk Beom, Lee Jae Sang, Jang Jae Kwang (alphabetical Order)"

import readFile
import numpy as np
import theano
import theano.tensor as T
import theano.printing as TP
from theano.compile.debugmode import DebugMode

import nltk
from nltk.tokenize import sent_tokenize

import gzip, cPickle
import pickle
import itertools

import sys
import os
import codecs
import time
from six.moves import zip
from datetime import datetime
from pprint import pprint

theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.compute_test_value = 'warn'

epsilon = 1e-6
dtype = theano.config.floatX
MINIBATCH_UNIT = 0
VOCA_DIM = 0
N_TIME_STEP_INPUT = 0
N_TIME_STEP_TARGET = 0
WORD2IDX_INPUT = dict()
WORD2IDX_TARGET = dict()
IDX2WORD_TARGET = dict()
SAVE_W_WHEN = 3
HIDDEN_DIM = 100

#########################################################################################################################

def shared(value, name=None):
    return theano.shared(value.astype(dtype), name=name)
#    return theano.shared(value.astype(dtype), name=name, borrow=True)

def shared_zeros(shape, name=None):
    return shared(value=np.zeros(shape).astype(dtype), name=name)
#    return theano.shared(value=np.zeros(shape), name=name, borrow=True)

def shared_zeros_like(x, name=None):
    return shared_zeros(shape=x.shape, name=name)
#    return theano.shared_zeros(shape=x.shape, name=name, dtype=theano.config.floatX, borrow=True)

def init_weights(shape, name=None):
    bound = np.sqrt(1.0/shape[1])
    w = np.random.uniform(-bound, bound, shape)
    return theano.shared(value=w.astype(dtype), name=name)
#    return theano.shared(value=w, name=name, borrow=True)

#########################################################################################################################

def adadelta(params, cost, lr=1.0, rho=0.95):
    # from https://github.com/fchollet/keras/blob/master/keras/optimizers.py
    grads = T.grad(cost, params)
    accus = [shared_zeros_like(p.get_value()) for p in params]
    delta_accus = [shared_zeros_like(p.get_value()) for p in params]
    updates = []

    for p, g, a, d_a in zip(params, grads, accus, delta_accus):
        new_a = rho * a + (1.0 - rho) * T.square(g)
        updates.append((a, new_a))
        update = g * T.sqrt(d_a + epsilon) / T.sqrt(new_a + epsilon)
        new_p = p - lr * update
        updates.append((p, new_p))
        new_d_a = rho * d_a + (1.0 - rho) * T.square(update)
        updates.append((d_a, new_d_a))

    return updates

#########################################################################################################################

def categorical_crossentropy(y_true, y_pred):
    # from https://github.com/fchollet/keras/blob/master/keras/objectives.py
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-2, keepdims=True)

    y_pred = y_pred.reshape( (-1,MINIBATCH_UNIT, VOCA_DIM) )
#    TP.Print("y_pred",attrs=('shape',))(y_pred)
    y_true = y_true.reshape( (-1,MINIBATCH_UNIT, VOCA_DIM) )
#    y_true = TP.Print("y_true", attrs=('shape',))(y_true)

    cce, updates = theano.scan(
    fn=T.nnet.categorical_crossentropy,   
        sequences=[y_pred,y_true]
    )   
#    cce = TP.Print("cce :",attrs=('shape',))(cce)
    return T.mean( cce )

#########################################################################################################################

class LSTM(object):
    def set_state(self, h, c):
        self.enc_h.set_value(h.get_value())
        self.c_tm1.set_value(c.get_value())

    def reset_state(self):
        self.enc_h = shared_zeros( (MINIBATCH_UNIT, VOCA_DIM), "enc_h")
        self.c_tm1 = shared_zeros( (MINIBATCH_UNIT, VOCA_DIM), "c_tm1")

#########################################################################################################################

class LSTMEncoder(LSTM):
    def __init__(self, hidden_node_dim, word_dim, enc_w = None):
        if enc_w != None :
            self.Ui, self.Wi, \
            self.Uf, self.Wf, \
            self.Uo, self.Wo, \
            self.Ug, self.Wg = itertools.chain( enc_w )

            self.params = []
            self.params = itertools.chain( enc_w )          

        elif enc_w == None :
            shape_b = (MINIBATCH_UNIT, hidden_node_dim )
            shape_U = (word_dim, hidden_node_dim )
            shape_W = (hidden_node_dim, hidden_node_dim )
            self.enc_h = shared_zeros(shape_b, "enc_h")
            self.c_tm1 = shared_zeros(shape_b, "c_tm1")

            self.Ui = init_weights(shape_U, "Ui")
            self.Wi = init_weights(shape_W, "Wi")

            self.Uf = init_weights(shape_U, "Uf")
            self.Wf = init_weights(shape_W, "Wf")

            self.Uo = init_weights(shape_U, "Uo")
            self.Wo = init_weights(shape_W, "Wo")

            self.Ug = init_weights(shape_U, "Ug")
            self.Wg = init_weights(shape_W, "Wg")

            self.params = [
                self.Ui, self.Wi,
                self.Uf, self.Wf,
                self.Uo, self.Wo,
                self.Ug, self.Wg,
            ]
		
    @staticmethod
    def step(x_t, enc_h, c_tm1,
        Ui, Wi,
        Uf, Wf,
        Uo, Wo,
        Ug, Wg
        ):

        i_t = T.nnet.sigmoid(T.dot(x_t, Ui) + T.dot(enc_h, Wi))
        f_t = T.nnet.sigmoid(T.dot(x_t, Uf) + T.dot(enc_h, Wf))
        o_t = T.nnet.sigmoid(T.dot(x_t, Uo) + T.dot(enc_h, Wo))
        g_t = T.tanh(T.dot(x_t, Ug) + T.dot(enc_h, Wg))

        c_t = c_tm1 * f_t + g_t * i_t
        h_t = T.tanh(c_t) * o_t

        return h_t, c_t

    def forward(self, X):
        states, updates = theano.scan(
            fn=self.step,
            sequences=[X],
            outputs_info=[self.enc_h, self.c_tm1],
            non_sequences=[
                self.Ui, self.Wi,
                self.Uf, self.Wf,
                self.Uo, self.Wo,
                self.Ug, self.Wg,
            ]        
        )
        return states[0][-1], states[1][-1]

    def encode(self, X, enc_w = None):
        return self.forward(X)

    def write_weights(self,encoder_w):
        self.Ui, self.Wi,\
        self.Uf, self.Wf,\
        self.Uo, self.Wo,\
        self.Ug, self.Wg = itertools.chain(encoder_w)
        return

    def returnW(self):
        self.w = [
            self.Ui, self.Wi,
            self.Uf, self.Wf,
            self.Uo, self.Wo,
            self.Ug, self.Wg,
        ]
        return self.w 

#########################################################################################################################

class LSTMDecoder(LSTM):
    def __init__(self, hidden_node_dim , word_dim, dec_w = None):

        if dec_w != None :
            self.Ui, self.Wi,\
            self.Uf, self.Wf,\
            self.Uo, self.Wo,\
            self.Ug, self.Wg,\
            self.V,\
            self.c_h, self.c_y,\
            self.y_t1 = itertools.chain( dec_w )

            self.params = []
            self.params = itertools.chain( dec_w )          

        elif dec_w == None :

            shape_b = (MINIBATCH_UNIT, hidden_node_dim )
            shape_U = (word_dim, hidden_node_dim )
            shape_W = (hidden_node_dim , hidden_node_dim )

            self.enc_h = shared_zeros(shape_b, "enc_h")
            self.c_tm1 = shared_zeros(shape_b, "c_tm1")

            self.Ui = init_weights(shape_U, "Ui")
            self.Wi = init_weights(shape_W, "Wi")

            self.Uf = init_weights(shape_U, "Uf")
            self.Wf = init_weights(shape_W, "Wf")

            self.Uo = init_weights(shape_U, "Uo")
            self.Wo = init_weights(shape_W, "Wo")

            self.Ug = init_weights(shape_U , "Ug")
            self.Wg = init_weights(shape_W, "Wg")

            # weights pertaining to output neuron
            self.V = init_weights( (hidden_node_dim, word_dim) , "V")

            # to weight 'context' from encoder
            self.c_h = init_weights((hidden_node_dim , hidden_node_dim ) , "c_h")
            self.c_y = init_weights( (HIDDEN_DIM, word_dim) , "c_y")

            # to weight 'y_t-1' for decoder's 'y'
            self.y_t = shared_zeros( ( MINIBATCH_UNIT, word_dim) , "y_t")
            self.y_t1 = init_weights( (word_dim, word_dim), "y_t1")

            self.params = [
                self.Ui, self.Wi,
                self.Uf, self.Wf,
                self.Uo, self.Wo,
                self.Ug, self.Wg,
                self.V,
                self.c_h, self.c_y,
                self.y_t1
            ]

    def write_weights(self,decoder_w):
        self.Ui, self.Wi,\
        self.Uf, self.Wf,\
        self.Uo, self.Wo,\
        self.Ug, self.Wg,\
        self.V,\
        self.c_h, self.c_y,\
        self.y_t1 = itertools.chain(decoder_w)              
        return

    def returnW(self):
        w = [
            self.Ui, self.Wi,
            self.Uf, self.Wf,
            self.Uo, self.Wo,
            self.Ug, self.Wg,
            self.V,
            self.c_h, self.c_y,
            self.y_t1
        ]
        return w

    ################################################################

    def decode_step(self, y_t, enc_h, c_tm1,
            Ui, Wi,
            Uf, Wf,
            Uo, Wo,
            Ug, Wg,
            V,
            c_h, c_y,
            y_t1
        ):
        x_i = T.dot(self.y_t, self.Ui)
        x_f = T.dot(self.y_t, self.Uf)
        x_o = T.dot(self.y_t, self.Uo) + T.dot(self.enc_h, self.c_h)
        x_c = T.dot(self.y_t, self.Ug)

        # minibatch_size * word_dim
        i_t = T.nnet.sigmoid(x_i + T.dot(enc_h, Wi))
        f_t = T.nnet.sigmoid(x_f + T.dot(enc_h, Wf))
        o_t = T.nnet.sigmoid(x_o + T.dot(enc_h, Wo))
        
        g_t = T.tanh(x_c + T.dot(enc_h, Wg))		
        c_t = (c_tm1 * f_t) + (g_t * i_t)
        h_t = T.tanh(c_t) * o_t
        y_t = T.dot(h_t, self.V) + T.dot(self.enc_h, self.c_y) + T.dot(self.y_t, self.y_t1) 

        return y_t, h_t, c_t

    ################################################################

    def decode(self, enc_h, c_tm1, timesteps):

        outputs, updates = theano.scan(
            fn=self.decode_step,
            outputs_info=[self.y_t, enc_h, c_tm1],
            non_sequences=[
                self.Ui, self.Wi,
                self.Uf, self.Wf,
                self.Uo, self.Wo,
                self.Ug, self.Wg,
                self.V,
                self.c_h, self.c_y,
                self.y_t1
            ],  
            n_steps=timesteps
        )

        return outputs[0]

    ################################################################

    @staticmethod
    def argmax(seq):
        seq = T.argmax(seq, axis=2)
        return seq

#########################################################################################################################

class Seq2Seq(object):
    def __init__(self, hidden_node_dim , word_dim, w = None):

        if w != None :
            enc_w, dec_w = itertools.chain(w)
            self.encoder = LSTMEncoder(hidden_node_dim , word_dim, enc_w)
            self.decoder = LSTMEncoder(hidden_node_dim , word_dim, dec_w)

        elif w == None :
            self.encoder = LSTMEncoder(hidden_node_dim , word_dim)
            self.decoder = LSTMDecoder(hidden_node_dim , word_dim)

        self.params = []
        self.params += self.encoder.params
        self.params += self.decoder.params

        self._predict = None
        self._train = None

    def compile(self, loss_func, optimizer):
        seq_input = T.tensor3()
        seq_target = T.tensor3()
        decode_timesteps = T.iscalar()

        enc_h, c_tm1 = self.encoder.encode(seq_input)
        seq_predict_flex = self.decoder.decode(enc_h, c_tm1, decode_timesteps)
        seq_argmax = self.decoder.argmax(seq_predict_flex)
        seq_predict = self.decoder.decode(enc_h, c_tm1, T.shape(seq_target)[0])

        loss = loss_func(seq_target,seq_predict)
        self._predict = theano.function([seq_input, decode_timesteps], seq_argmax)

        updates = []
        updates += optimizer(self.params, loss)
        self._train = theano.function([seq_input, seq_target], loss, updates = updates)

    def returnW(self):
        return self.encoder.returnW(), self.decoder.returnW()

    def write_weights(self,encoder_w, decoder_w):
        self.encoder.write_weights(encoder_w)
        self.decoder.write_weights(decoder_w)
        return

    def predict(self, seq_input, decode_timesteps):
        self.encoder.reset_state()
        self.decoder.reset_state()
        return self._predict(seq_input, decode_timesteps)

    def train(self, seq_input, seq_target):
        self.encoder.reset_state()
        self.decoder.reset_state()
        return self._train(seq_input, seq_target)

#########################################################################################################################

def apply_to_m1(lst, dtype=np.int64):
    inner_max_len = max(map(len, lst))
    result = np.zeros(  [len(lst), inner_max_len], dtype  )
    result[:] = -1

    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result

#########################################################################################################################

def sort_by_timestep(sentence_group):
    same_len_seq = np.asarray(sentence_group)

    same_len_seq = apply_to_m1(same_len_seq)
    sorted_seq = same_len_seq.transpose()

    return sorted_seq

#########################################################################################################################

def gen_1hot(seqs, minibatch_unit) :
    max_seq_len = find_maxlen(seqs)            
    row = max_seq_len * minibatch_unit
    one_hot = np.zeros( (row, VOCA_DIM) )
    time_step_seq = sort_by_timestep(seqs)

    j = 0        
    for word_idx in np.nditer( time_step_seq ) :
        if word_idx != -1:
            one_hot[j][word_idx] = 1
        j+=1

    one_hot = np.reshape(one_hot, (max_seq_len, -1 , VOCA_DIM)  )
    one_hot = one_hot.astype(dtype)
    return one_hot

#########################################################################################################################

def find_maxlen(sentence_group):
    max_seq_len = 0
    for seq in sentence_group :
        if len(seq) > max_seq_len :
            max_seq_len = len(seq)
    return max_seq_len

#########################################################################################################################

def train_go(seq_input, seq_target, minibatch_unit, ended_i):

    print("FUNC START : train_go ")

    for i, (group_i, group_t) in enumerate( zip(seq_input, seq_target )):
        i+=1
        print("train_go : ", i, "loop ")
        if i < ended_i : 
            continue

        if i % SAVE_W_WHEN == 0:
            print("train_go : save weight when ", i, "th loop")
            weights = []
            encoder_w, decoder_w = seq2seq.returnW()
            weights.append(encoder_w)
            weights.append(decoder_w)
            weights.append(i-1)
            f = file('data_seq2seq_weight.pkl', 'wb')
            cPickle.dump(weights , f , protocol=cPickle.HIGHEST_PROTOCOL )
            f.close()
            print("train_go : save weight -> completed")

        if group_i and len(group_i[0]) != 0 :
            si = gen_1hot(group_i, minibatch_unit)
            st = gen_1hot(group_t, minibatch_unit)
            print(i, "th train")
            
            print("result of train function(loss or update) :", seq2seq.train(si, st) )
    return 

#########################################################################################################################

def gen_processed_seq(input_sentence):
    # tokenized seq
    tokenized_seq = nltk.word_tokenize( input_sentence )
    print("tokenized_seq :", tokenized_seq)

    # insert tokenized seq into minibatch
    input_sentences = [ None for _ in range(MINIBATCH_UNIT) ]
    input_sentences[0] = tokenized_seq

    # ok until this
    print("input_sentences[0] :", input_sentences[0])

    # word_to_idx
    seq_input = readFile.word_to_idx(tokenized_seq, WORD2IDX_INPUT )
    print ("seq_input",seq_input)

    # indexed seq into minibatch
    sorted_seq_input = [ None for _ in range(MINIBATCH_UNIT) ]
    sorted_seq_input[0] = seq_input[0]
    print("sorted_seq_input[0] : ", sorted_seq_input[0])

    # ok until this
    input_len = len(seq_input[0])

    # make '-1'
    for i in range(MINIBATCH_UNIT-1):
        for j in range(input_len):
            sorted_seq_input[i+1] = [-1]

    input_finally = [] 
    input_finally.append(sorted_seq_input)
    return input_finally

#########################################################################################################################

def make_1hot(max_len, sorted_sentences, type, minibatch_unit, num_of_seq):
    print("[make_1hot] sorted_sentences : ", sorted_sentences)
    one_hot = [None for _ in range( len(sorted_sentences) )]
     
    for i, sentence_group in enumerate(sorted_sentences):

        if sentence_group and len(sentence_group[0]) != 0 :

            max_seq_len = find_maxlen(sentence_group)            
            row = max_seq_len * minibatch_unit
            one_hot[i] = np.zeros( (row, VOCA_DIM) )
            time_step_seq = sort_by_timestep(sentence_group)

            j = 0

            for word_idx in np.nditer( time_step_seq ) :
                if word_idx != -1:
                    one_hot[i][j][word_idx] = 1
                j+=1

            one_hot[i] = np.reshape(one_hot[i], ( max_seq_len, -1, VOCA_DIM)  )
    return one_hot


def gen_one_hot(input_len, input_seq):
    one_hot = make_1hot(N_TIME_STEP_INPUT, input_seq, "predict", MINIBATCH_UNIT, 1)
    one_hot[0] = one_hot[0].astype(dtype)
    print("one_hot : ", one_hot)
    return one_hot

#########################################################################################################################

def get_idx(argmax, num_of_word):
    idx_list = argmax[ : num_of_word, 0]
    return idx_list
    
#########################################################################################################################

def predict():
    input_sentence = raw_input("enter English Sentence : ")
    print("input_sentence : " ,input_sentence)

    input_seq = gen_processed_seq(input_sentence)

    print("input_seq : ",input_seq)
    print("input_seq[0][0] : ",input_seq[0][0])
    num_of_word = len(input_seq[0][0])

    one_hot = gen_one_hot(N_TIME_STEP_INPUT, input_seq)
    print("one_hot[0].shape : ", one_hot[0].shape)

    argmax = seq2seq.predict(one_hot[0] ,N_TIME_STEP_INPUT )
    print("argmax_fin shape : ", argmax.shape)
    print("argmax_fin : ", argmax)

    idx_list_np = get_idx(argmax, num_of_word)
    idx_list_py = idx_list_np.tolist()
    print("IDX2WORD_TARGET : ",IDX2WORD_TARGET)
#    print("IDX2WORD_TARGET[6] :", IDX2WORD_TARGET[6])
    result = readFile.idx_to_word(idx_list_py, IDX2WORD_TARGET)
    translated = ""

    for elem in result :
        translated += elem
        translated += " "

    print("translated : " , translated)
    print("Translation End")

#########################################################################################################################

if __name__ == "__main__":

    global WORD2IDX_INPUT
    global WORD2IDX_TARGET
    global VOCA_DIM
    global MINIBATCH_UNIT
    global N_TIME_STEP_INPUT
    global N_TIME_STEP_TARGET
    global IDX2WORD_TARGET

    while(True):
        print("select a menu")

        print("0. pre-processing")
        print("1. Training")
        print("2. Translate English into Spanish.")
        val = input("selection : ") 
        
        if val == 0:      
            seq_by_mini_batch_input, seq_by_mini_batch_target, minibatch_unit, voca_word_dim, word_to_index_input, word_to_index_targrt, index_to_word_target, maxlen_input, maxlen_target = readFile.preprocessing()
            dataset = [seq_by_mini_batch_input, seq_by_mini_batch_target, minibatch_unit, voca_word_dim, maxlen_input, maxlen_target]
            f = file('./data_for_train.pkl', 'wb')
            cPickle.dump(dataset , f , protocol=cPickle.HIGHEST_PROTOCOL )
            f.close()

            dataset = [minibatch_unit, voca_word_dim, word_to_index_input, index_to_word_target, maxlen_input, maxlen_target]
            f = file('./data_for_predict.pkl', 'wb')
            cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        elif val == 1:
            print("load data for train")
            f = file('data_for_train.pkl', 'rb')
            data = pickle.load(f)
            seq_by_mini_batch_input, seq_by_mini_batch_target, minibatch_unit, voca_word_dim, maxlen_input, maxlen_target = data

            VOCA_DIM = voca_word_dim +2 
            MINIBATCH_UNIT = minibatch_unit                
            N_TIME_STEP_INPUT = maxlen_input
            N_TIME_STEP_TARGET = maxlen_target

            seq2seq = Seq2Seq(HIDDEN_DIM, VOCA_DIM )
            seq2seq.compile(loss_func=categorical_crossentropy, optimizer=adadelta)
            
            ended_i = 0
			
            try:
                f = file('data_seq2seq_weight.pkl', 'rb')
            except IOError:
                print("file not found")
            else:
                print("load trained weight")
                weights = pickle.load(f)
                f.close()
                encoder_w, decoder_w, ended_i = weights
                print("seq2seq.write_weights(encoder_w, decoder_w)")
                seq2seq.write_weights(encoder_w, decoder_w)
             
            print("seq_to_1hot()")
            train_go(seq_by_mini_batch_input, seq_by_mini_batch_target, MINIBATCH_UNIT, ended_i)

            print("save weight")
            weights = []
            weights += seq2seq.returnW()
            weights.append(0)
            f = file('./data_seq2seq_weight.pkl', 'wb')
            cPickle.dump(weights , f , protocol=cPickle.HIGHEST_PROTOCOL )
            f.close()

        elif val == 2:
            print("load data for predict")
            f = file('data_for_predict.pkl', 'rb')
            data = pickle.load(f)
            print("pickle.load complete")
            minibatch_unit, voca_word_dim, word_to_index_input, index_to_word_target, maxlen_input, maxlen_target = data
            print("data extract done")
            VOCA_DIM = voca_word_dim +2 
            MINIBATCH_UNIT = minibatch_unit                
            N_TIME_STEP_INPUT = maxlen_input
            N_TIME_STEP_TARGET = maxlen_target
            WORD2IDX_INPUT = word_to_index_input
            IDX2WORD_TARGET = index_to_word_target

            print("gen seqseq")
            seq2seq = Seq2Seq(HIDDEN_DIM, VOCA_DIM )
            print("seq2seq.compile")
            seq2seq.compile(loss_func=categorical_crossentropy, optimizer=adadelta)

            ended_i = 0
            print("try")            
            try:
                f = file('data_seq2seq_weight.pkl', 'rb')
            except IOError:
                print("file not found")
            else:
                print("file found : data_seq2seq_weight.pkl")
                weights = pickle.load(f)
                print("pickle.load : data_seq2seq_weight.pkl")

                f.close()
                print("encoder_w, decoder_w, ended_i = weights")
                encoder_w, decoder_w, ended_i = weights
                print("seq2seq.write_weights(encoder_w, decoder_w)")
                seq2seq.write_weights(encoder_w, decoder_w)
			
            while(True):
                predict()