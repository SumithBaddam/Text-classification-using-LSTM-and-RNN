######SR Symptom classification######
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys
from ast import literal_eval

def unique_words(sentence, number):
    return [w for w in set(sentence.translate(None, punctuation).lower().split()) if len(w) >= number]

def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line

def remove_punctuation(text):
    return text.translate(tbl)


tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

#df = pd.read_csv("All_SRNotes_Extracted.csv", encoding='utf-8')
df = pd.read_excel('CRS_Data_Labeled.xls', encoding = 'utf-8')
df['Keywords_SR'] = df['Keywords_SR'].fillna('')
#Clean the keywords data...before training...


#If groupby not done...use below code
#df = df.groupby('sr_number').agg({'sr_number':'first', 'Keywords': ', '.join})
df = df.groupby('SR_NUMBER').agg({'SR_NUMBER':'first', 'Keywords_SR': ', '.join, 'FA Label':'first', 'SR Symptom':'first','SR_cluster_id(Symptom)':'first', 'SYMPTOM_CODE':'first'})
old_df = df.copy()
######Mark any label less than certain value as OTHERS######
b = df['SR Symptom'].value_counts()[0]/15
a = df['SR Symptom'].value_counts()
m = df['SR Symptom'].isin(a.index[a < b])
df.loc[m, 'SR Symptom'] = 'OTHER'


sr_label = df['SR Symptom'].tolist()
keywords = df['Keywords_SR'].values.tolist()
for i in range(len(keywords)):
    keywords[i] = keywords[i].replace('[], ', '')
    keywords[i] = keywords[i].replace('], [', ',')

docs = []
for i in range(len(keywords)):
    s = ''
    if(len(keywords[i].replace(',', '').replace(' ', '')) > 2):
        for keyword in literal_eval(keywords[i])[:3]:
            #print(keyword)
            keyword = keyword.split(':')[0]
            #s = s + (' '.join(set(keyword.split()))) + ','
            s = s + keyword + ','
            #print(s)
        docs.append(s.rstrip(','))
    else:
        docs.append(' ')

d = pd.DataFrame()
d['keywords'] = docs
d['Label'] = sr_label

d = d.groupby('Label').agg({'Label':'first', 'keywords': ', '.join})
unique_labels = d['Label'].tolist()

data = {}
for i in range(d.shape[0]):
    data[d.iloc[i]['Label']] = d.iloc[i]['keywords'].split(',')


#data = training_data

# get a list of all categories to train for
categories = list(data.keys())
words = []
# a list of tuples with words in the sentence and category name
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        print(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        print ("tokenized words: ", w)
        words.extend(w)
        docs.append((w, each_category))


# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

print (words)
print (docs)


stemmer = LancasterStemmer()
# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)
 
for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)
    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1
    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)
 
# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:,0])
train_y = list(training[:,1])


# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=100, batch_size=8, show_metric=True)
model.save('sr_symptom_model.tflearn')


######TESTING######
sent_1 = data['Errors'][1]
sent_2 = data['Fan low rpm'][1]
sent_3 = data['Linecard Reset ( MBE Error)'][0]
sent_4 = data['Low Power Register Bits'][3]

def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1
    return(np.array(bow))


print(categories[np.argmax(model.predict([get_tf_record(sent_1)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent_2)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent_3)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent_4)]))])


######Testing on ASR9K######
df = pd.read_excel('CRS_Data_Labeled.xls', encoding = 'utf-8')
df['Keywords_SR'] = df['Keywords_SR'].fillna('')
#Clean the keywords data...before training...


#If groupby not done...use below code
#df = df.groupby('sr_number').agg({'sr_number':'first', 'Keywords': ', '.join})
df = df.groupby('SR_NUMBER').agg({'SR_NUMBER':'first', 'Keywords_SR': ', '.join, 'FA Label':'first', 'SR Symptom':'first','SR_cluster_id(Symptom)':'first', 'SYMPTOM_CODE':'first'})
old_df = df.copy()
######Mark any label less than certain value as OTHERS######
b = df['SR Symptom'].value_counts()[0]/15
a = df['SR Symptom'].value_counts()
m = df['SR Symptom'].isin(a.index[a < b])
df.loc[m, 'SR Symptom'] = 'OTHER'


sr_label = df['SR Symptom'].tolist()
keywords = df['Keywords_SR'].values.tolist()
for i in range(len(keywords)):
    keywords[i] = keywords[i].replace('[], ', '')
    keywords[i] = keywords[i].replace('], [', ',')

docs = []
for i in range(len(keywords)):
    s = ''
    if(len(keywords[i].replace(',', '').replace(' ', '')) > 2):
        for keyword in literal_eval(keywords[i])[:3]:
            #print(keyword)
            keyword = keyword.split(':')[0]
            #s = s + (' '.join(set(keyword.split()))) + ','
            s = s + keyword + ','
            #print(s)
        docs.append(s.rstrip(','))
    else:
        docs.append(' ')

predictions = []
for doc in docs:
    p = []
    for sent in doc.split(','):
        a = categories[np.argmax(model.predict([get_tf_record(sent)]))]
        if(a not in p):
            p.append(a)
    predictions.append(p)

df['Predictions'] = predictions
df.to_csv('CRS_Predictions.csv', encoding = 'utf-8', index = False)
df[df['SR Symptom']=='Fan low rpm'][['Predictions']]

###Loading FA model
a = 1434 #len(train_x[0])
b = 10 #len(train_y[0])
net1 = tflearn.input_data(shape=[None, a])
net1 = tflearn.fully_connected(net1, 8)
net1 = tflearn.fully_connected(net1, 8)
net1 = tflearn.fully_connected(net1, b, activation='softmax')
net1 = tflearn.regression(net1)

# Define model and setup tensorboard
model1 = tflearn.DNN(net1, tensorboard_dir='tflearn_logs')
model1.load('fa_model.tflearn')

fa_predictions = []
for doc in docs:
    p = []
    for sent in doc.split(','):
        a = categories[np.argmax(model1.predict([get_tf_record(sent)]))]
        if(a not in p):
            p.append(a)
    fa_predictions.append(p)


df['FA_Predictions'] = fa_predictions
df.to_csv('CRS_FA_SR_Predictions.csv', encoding = 'utf-8', index = False)



######Testing on new data set######
df1 = pd.read_excel('FARC_Data_Extracted_ASR9000_Final.xls', encoding = 'utf-8')
df1['Keywords'] = df1['Keywords'].fillna('')
df1 = df1.groupby('SR_NUMBER').agg({'SR_NUMBER':'first', 'Keywords': ', '.join, 'FA Label':'first', 'SR Symtom':'first', 'SYMPTOM_CODE':'first'})

df1 = df1[~df1['Keywords'].isnull()]

###Check if this is correct way???
keywords = df1['Keywords'].values.tolist()
for i in range(len(keywords)):
    keywords[i] = keywords[i].replace('[], ', '')
    keywords[i] = keywords[i].replace('], [', ',')

docs = []
for i in range(len(keywords)):
    s = ''
    for word in keywords[i].split(',')[:3]:
        s = s + word.split(':')[0]
    docs.append(s)

df1['Keywords_SR'] = docs
predictions = []
for sent in docs:
    l = []
    for idx in np.argsort(model.predict([get_tf_record(sent)])[0])[-3:]:
        l.append(categories[idx])
    predictions.append(l)

df1['Predictions'] = predictions
df1.to_csv('Predictions_ASR9000.csv', encoding = 'utf-8', index = False)
df1[df1['SR Symtom']=='boot'][['Predictions', 'Keywords_SR']]

#############Plot different clusters and symptoms to validate clustering method############
s = df[df['cluster_id'] == 7]['SR Symptom'].value_counts()
import matplotlib.pyplot as plt
s.plot.bar()
plt.show()
