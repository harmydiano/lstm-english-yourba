from __future__ import print_function

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop,SGD,Adadelta
from keras import models
from keras.utils.data_utils import get_file
import random
import sys

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

language = {}

"""we write a function to load the languages test data
    All the datas are read and converted into lower case letters
"""

def load_text(file):
    text = open(file).read().lower()
    return text

"""This function takes two parameters file & type
    file : This is the dataset corups
    type: The type specifies the type of data to be processed
"""
def process_text(file, type):
    text = load_text(file)
    print('%s corpus length:' % type, len(text))
    chars_lang = sorted(list(set(text)))
    if type == 'yor':
        language['type'] = chars_lang
    else:
        language['type'] = chars_lang
    print('total chars %s :' % type, len(chars_lang))
    char_indices_lang = dict((c, i) for i, c in enumerate(chars_lang))
    indices_char_lang = dict((i, c) for i, c in enumerate(chars_lang))
    return language['type'], text


"""This function Combining both the character sets"""
def combine_lang_chars():
    chars_eng = process_text('data/eng.txt', 'eng')
    print(chars_eng[0])

    chars_yor = process_text('data/yoruba.txt', 'yor')
    print(chars_yor)
    chars = set(chars_eng[0])
    chars = sorted(list(chars.union(set(chars_yor[0]))))
    print('total chars :', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, chars, indices_char

"""This function splits the data into train set
    The datas are splited into 80% each to train the model
"""
#split train data into %80
def split_train_data():
    text_eng = load_text('data/eng.txt')
    text_yor = load_text('data/yoruba.txt')
    text_eng_train = text_eng[:int(0.8*len(text_eng))]
    text_yor_train = text_yor[:int(0.8*len(text_yor))]

    return text_eng_train, text_yor_train

"""This function splits the data into test set
    The datas are splited into 20% each to test the model
"""

#split train data into %20
def split_test_data():
    text_eng = load_text('data/eng.txt')
    text_yor = load_text('data/yoruba.txt')
    text_eng_test = text_eng[int(0.8*len(text_eng)):]
    text_yor_test = text_yor[int(0.8*len(text_yor)):]

    return text_eng_test, text_yor_test

"""Function to split the train corpus into sentences
    The next_chars retains the next character after a particular word
    it takes 3 params (text, maximum length and the diffference between values of each range

"""
def get_sentences(text, maxlen, step):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    #print('nb sequences:', len(sentences_eng))
    return sentences, next_chars

"""Function to split the test corpus into sentences
    The next_chars retains the next character after a particular word
"""

def get_sentences_test(text, maxlen, step):
    print(text)
    sentences = []
    next_string = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_string.append(text[i + maxlen:i + maxlen + 5])
    return sentences, next_string

"""Converting sentences to vectors form"""
def get_vectors(sentences, chars, char_indices, next_chars):
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X,y

"""calling the lstm model library from scikit"""
def build_model(chars, maxlen, X, y, name):
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=20)
    model.save(name)
    return model

"""Get vector for test set"""

def get_vector(sentence, chars , char_indices ):
    X = np.zeros((1, maxlen, len(chars)), dtype=np.bool)
    for t, char in enumerate(sentence):
        X[0, t, char_indices[char]] = 1
    return X

"""predicting each of the 5 characters from the model
    This function takes the model , the mixture of 5 sentences each from yoruba and english,
    the next 5 chars , the mixture of all characters and the characters with their indexes
    it returns the probability of the sentence
"""

def get_predictions(model, X_test, Y_test, chars, indices_char):
    y_true_list = []
    y_pred_list = []
    prob_list = []

    for i in range(len(X_test)):
        term = X_test[i]
        termy = Y_test[i]
        x = np.zeros((5, maxlen, len(chars)))

        for i in range(5):
            x[i] = get_vector(term[i:] + termy[:i], chars, char_indices)

        term_probs = []

        for i in range(5):
            #5 columns and 59 rows
            preds = model.predict(x[i].reshape(1, 5, 59), verbose=0)[0]
            term_probs.append(preds[char_indices[termy[i]]])

        prob_list.append(term_probs)


    return prob_list

def get_single_predictions(model, X_test, Y_test, chars, indices_char):
    y_true_list = []
    y_pred_list = []
    prob_list = []


    term = X_test
    termy = Y_test
    x = np.zeros((5, maxlen, len(chars)))

    for i in range(5):
        print(term[i:])
        print(termy[:i])
        print(chars)
        print(char_indices)
        x[i] = get_vector(term[i:] + termy[:i], chars, char_indices)

    term_probs = []

    for i in range(5):
        #5 columns and 59 rows
        preds = model.predict(x[i].reshape(1, 5, 59), verbose=0)[0]
        term_probs.append(preds[char_indices[termy[i]]])

    prob_list.append(term_probs)


    return prob_list

"""Predicting if it is english or yoruba"""
def predict_language(prob_eng, prob_yor):
    y = [1] * len(prob_eng)
    probs_eng = []
    probs_yor = []
    for i in range(len(prob_eng)):
        sum_eng = 0
        sum_yor = 0
        for j in range(len(prob_eng[0])):
            #take the log of each and sum
            sum_eng += np.log(prob_eng[i][j])
            sum_yor += np.log(prob_yor[i][j])
        #print (sum_eng,sum_frn)
        #find the exp of the summation of the values
        sum_eng = np.exp(sum_eng)
        sum_yor = np.exp(sum_yor)
        if sum_eng < sum_yor:
            y[i] = 0
        probs_eng.append(sum_eng)
        probs_yor.append(sum_yor)
    return y, probs_eng, probs_yor


def predict_single_language(prob_eng, prob_yor):
    y = [1] * len(prob_eng)
    print(len(prob_eng))
    print(len(prob_eng[0]))
    probs_eng = []
    probs_yor = []
    sum_eng = 0
    sum_yor = 0
    i = 0
    for j in range(len(prob_eng[0])):
        sum_eng += np.log(prob_eng[i][j])
        sum_yor += np.log(prob_yor[i][j])
    print (sum_eng,sum_yor)
    sum_eng = np.exp(sum_eng)
    sum_yor = np.exp(sum_yor)
    if sum_eng < sum_yor:
        y[i] = 0
    probs_eng.append(sum_eng)
    probs_yor.append(sum_yor)
    return y, probs_eng, probs_yor


char_indices , chars, indices_char = combine_lang_chars()

#fecth training sets sentences
maxlen = 5 #initialze the maximum length of words for trained dataset
step = 1 #initialize difference between each range outputs for trained dataset
"""call the function to split the training sets into sentences"""
text_eng_train, text_yor_train = split_train_data()
"""we passed maximum length of the sentences as 5 and step as 1"""
sentences_eng, next_chars_eng = get_sentences(text_eng_train, maxlen, step)
sentences_yor, next_chars_yor = get_sentences(text_yor_train, maxlen, step)
print(sentences_eng)
print(next_chars_eng)

#fetch test sets sentences
text_eng_test, text_yor_test = split_test_data()
maxlen = 5 #initialze the maximum length of words for trained dataset
step = 20 #initialize difference between each range outputs for trained dataset
"""call the function to split the training sets into sentences"""
"""we passed maximum length of the sentences as 5 and step as 20"""
sentences_eng_test, next_string_eng_test = get_sentences_test(text_eng_test, maxlen, step)
sentences_yor_test, next_string_yor_test = get_sentences_test(text_yor_test, maxlen, step)
print(sentences_eng_test)
print(next_string_yor_test)
"""select the first 100"""
sentences_eng_test = sentences_eng_test[:100]
next_string_eng_test = next_string_eng_test[:100]
sentences_yor_test = sentences_yor_test[:100]
next_string_yor_test = next_string_yor_test[:100]

X_eng, y_eng = get_vectors(sentences_eng, chars, char_indices, next_chars_eng)
X_yor, y_yor = get_vectors(sentences_yor, chars, char_indices, next_chars_yor)

print ("Shape of X Eng", X_eng.shape)
print ("Shape of Y Eng", y_eng.shape)
print ("Shape of X Frn", X_yor.shape)
print ("Shape of Y Frn", y_yor.shape)

X_test = sentences_eng_test + sentences_yor_test
Y_test = next_string_eng_test + next_string_yor_test
y_true = ([1] * 100) + ([0] * 100) #1 for English, 0 for Yoruba

print("Shape of X_test :", len(X_test),len(X_test[0]))
print("Shape of Y_test :", len(Y_test),len(Y_test[0]))
print("Shape of Y_true :", len(y_true),1)

"""This builds and save the model"""
#model_eng = build_model(chars, maxlen, X_eng, y_eng, 'eng.h5')
#model_yor = build_model(chars, maxlen, X_yor, y_yor, 'yor.h5')

"""load the saved models"""
model_eng = models.load_model('eng.h5')
model_frn = models.load_model('yor.h5')

#print(chars)
#print(indices_char)
print(X_test)
print(Y_test)
print(chars)
print(indices_char)

"""This is use to run predictions for all the test dataset"""
#y_prob_eng = get_predictions(model_eng, X_test,Y_test ,chars, indices_char)
#y_prob_yor = get_predictions(model_frn, X_test,Y_test ,chars, indices_char)
#print(y_prob_eng, y_prob_yor)

#y_pred, probs_eng, probs_frn = predict_language(y_prob_eng, y_prob_yor)
#y_hat = [ prob_eng - prob_frn for prob_eng, prob_frn in zip(probs_eng, probs_frn)]

#print(y_pred, probs_eng, probs_frn )
#print(y_pred, y_true)

"""This is use to run predictions for a sample in the test dataset"""
y_prob_eng = get_single_predictions(model_eng, X_test[50],Y_test[50] ,chars, indices_char)
y_prob_yor = get_single_predictions(model_frn, X_test[50],Y_test[50] ,chars, indices_char)
print(y_prob_eng, y_prob_yor)

y_pred, probs_eng, probs_frn = predict_single_language(y_prob_eng, y_prob_yor)
if y_pred == [1]:
    print('Text is English')
else:
    print('Text is Yoruba')
#y_hat = [ prob_eng - prob_frn for prob_eng, prob_frn in zip(probs_eng, probs_frn)]

print(y_pred, probs_eng, probs_frn )
#print ("Accuracy : ",accuracy_score(y_true, y_pred))