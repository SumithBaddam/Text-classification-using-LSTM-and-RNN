with open("amazon_cells_labelled.txt") as f1:
    lines = f1.readlines()

with open("imdb_labelled.txt") as f1:
    temp = f1.readlines()
    lines=lines+temp

with open("yelp_labelled.txt") as f1:
    temp = f1.readlines()
    lines=lines+temp

x = []
y = []
for value in lines:
    temp = value.split('\t')
    x.append(temp[0])
    temp[1].replace('\n','')
    y.append(int(temp[1]))

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=2500,split=' ')
tokenizer.fit_on_texts(x)

from keras.preprocessing.sequence import pad_sequences

X = tokenizer.texts_to_sequences(x)
X = pad_sequences(X)

import numpy as np 
from sklearn.model_selection import train_test_split

Y = []
for val in y:
    if(val == 0):
        Y.append([1,0])
    else:
        Y.append([0,1])

Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.8)


import keras 
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential

model = Sequential()
model.add(Embedding(2500,128,input_length=X.shape[1],dropout=0.2))
model.add(LSTM(300, dropout_U=0.2,dropout_W=0.2))
model.add(Dense(2,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,verbose=2,batch_size=32)

print(model.evaluate(x_test,y_test)[1])

