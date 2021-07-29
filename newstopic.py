import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

dataset = pd.read_csv('../_data/train_data.csv',sep=',')
x_test = pd.read_csv('../_data/test_data.csv',sep=',')
y_predic = pd.read_csv('../_data/topic_dict.csv',sep=',') 
sub =  pd.read_csv('../_data/sample_submission.csv',sep=',') 
print(type(dataset))
print(type(x_test))
print(type(y_predic))
print(sub)
'''
<class 'pandas.core.frame.DataFrame'>
<class 'pandas.core.frame.DataFrame'>
<class 'pandas.core.frame.DataFrame'>
'''
x_train = np.array([i for i in dataset['title']])
x_test = np.array([i for i in x_test['title']])
y_predic = np.array([i for i in dataset['topic_idx']])

print(type(x_train))
print(type(x_test))
print(type(y_predic))

'''
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
'''
token = Tokenizer(45654) # 반환 사이즈 지정

token.fit_on_texts(x_train)
print(token.word_index)
'''

'''
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)


print(len(x_train))
print(len(x_test))
print("최대길이 : ", max(len(i) for i in x_train)) 
print("평균길이 : ", sum(map(len, x_train))/ len(x_train)) 
print("최대길이 : ", max(len(i) for i in x_test)) 
print("평균길이 : ", sum(map(len, x_test))/ len(x_test))
'''
45654 x_train
9131 x_test
train
최대길이 :  13
평균길이 :  6.623954089455469
test
최대길이 :  11
평균길이 :  5.127696856861242
'''
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
x_train = pad_sequences(x_train, maxlen=14,padding='pre')
x_test = pad_sequences(x_test, maxlen=12,padding='pre')
print('pading')

y_predic = to_categorical(y_predic)

print(y_predic)
'''
[[0. 0. 0. ... 1. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]
 ...
 [0. 1. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]]
(45654, 7)
'''
print(y_predic.shape)# 
'''
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
scaler.fit(x_test) 
x_test = scaler.transform(x_test)
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM,Conv1D,Dropout
model = Sequential()     
model.add(Embedding(input_dim=45654, output_dim=55,input_length=None))
model.add(Conv1D(128,2,activation='relu'))
model.add(Conv1D(64,2,activation='relu'))
model.add(LSTM(32, return_sequences=True)) # bidirectional(양방향 LSTM)
model.add(Dropout(0.2))
model.add(Conv1D(128,2,activation='relu'))
model.add(Conv1D(64,2,activation='relu'))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(Conv1D(128,2,activation='relu'))
model.add(Conv1D(64,2,activation='relu'))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation= 'softmax'))
'''

'''
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', patience=30, mode='max', verbose=3)
model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics=['acc'])
model.fit(x_train,y_predic,epochs=1000,batch_size=64,validation_split=0.3,callbacks=[es])

loss = model.evaluate(x_train,y_predic)
y_pred = model.predict(x_test) 

print('Loss :' , loss[0])
print('Acc :' , loss[1])
print('predict',y_pred)
'''
결과는 테스트를 비교하기 때문에 최종적으로 나오는 값은 트레인을 fit했을때랑 다르다
->트레인에대한 loss와acc는 신뢰할수없다 과적합되어있기때문이다 validation이 더 정확함

Loss : [0.683982789516449, 0.7577649354934692]
Loss : 0.6781153678894043
Acc : 0.7602619528770447

단순연산수 증가
Loss : 0.8178616762161255
Acc : 0.8747535943984985

Conv1D 레이어 추가
Loss : 0.8904010653495789
Acc : 0.8832084536552429

returnsequences 추가
Loss : 0.6705284118652344
Acc : 0.8800762295722961

단순 모델링증가
Loss : 0.6345469951629639
Acc : 0.8622683882713318

데이터 반환사이즈 증가 -> 2000 20000
Loss : 0.6649211645126343
Acc : 0.8940728306770325

데이터 반환사이즈 증가 -> 20000 45654
Loss : 0.7939437031745911
Acc : 0.8999430537223816

dropout 0.4
Loss : 0.7781057953834534
Acc : 0.8954527378082275
'''
'''
topic = []
for i in  range(len(y_pred)):
     topic.append(np.argmax(y_pred[i]))
sub['topic_idx'] = topic
sub.to_csv('./save/sample_submission.csv',index=False)

'''




