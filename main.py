import numpy
import numpy as np
import binascii
import pandas as pd
from keras.models import load_model
import operator
from sklearn.utils import shuffle
from keras import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

def format_string_to_list(list):
    max = 100
    x = []
    for i in range(len(list)):
        x.append(int(list[i]))
    l = len(x)
    while l < max:
        x.append(0)
        l = len(x)
    x = x[:max]
    return x
data = pd.read_csv('vgsales.csv', sep=',', header=None, names=["Rank","Name","Platform","Year",
                                                             "Genre","Publisher","NA_Sales","EU_Sales","JP_Sales",
                                                             "Other_Sales","Global_Sales"])
data = shuffle(data)
Name = data['Name'].values.tolist()
Publisher = data['Publisher'].values.tolist()
n = 0
nintendo = 0
activision = 0
dataX = []
dataY = []
real_name = []
for i in range(len(Name)):
    sname = ''.join(format(ord(x), 'b') for x in Name[i])
    append_data = []
    append_data.extend(format_string_to_list(sname))
    if Publisher[i] == "Nintendo":
        dataY.append([1])
        dataX.append(append_data)
        n+=1
        nintendo += 1
        real_name.append(Name[i])
    elif Publisher[i] == "Activision":
        dataY.append([0])
        dataX.append(append_data)
        n += 1
        activision += 1
        real_name.append(Name[i])
print(nintendo, activision, n)
dataX = np.array(dataX)
dataY = np.array(dataY)


persent = 0.01
len_train = int(len(dataX) * persent)
len_test = int(len(dataX) - len_train)
trainX = dataX[:len_train]
trainY = dataY[:len_train]
testX = dataX[-len_test:]
testY = dataY[-len_test:]
real_name = real_name[-len_test:]
# model = Sequential()
# model.add(Dense(len(dataX[0]), input_dim=len(dataX[0]), activation='relu'))
# model.add(Dense(len(dataX[0]), activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
# model.fit(trainX, trainY, batch_size=1, epochs=10, verbose=1)
# model.save('model_Nintendo.h5')
model = load_model('model_Nintendo.h5')

print("Обучаем")
# Делаем предсказания класса на основе testX
prediction = model.predict(testX)
# Вычисляем ошибку
eval = model.evaluate(testX, testY, verbose=1)
print(eval)
graph_test = []
for i in range(len(testY)):
    print(str(testY[i][0])+"\t"+str(prediction[i][0])+"\t"+str(real_name[i]))

