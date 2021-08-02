import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
sub =  pd.read_csv('../_data/sample_submission.csv',sep=',') 

x_train = np.load('../_npy/news_train_data.npy',allow_pickle=True)
x_test = np.load('../_npy/news_test_data.npy',allow_pickle=True)
y_predic = np.load('../_npy/news_topic_dict.npy',allow_pickle=True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM,Conv1D,Dropout,Bidirectional,Flatten,MaxPooling1D,GlobalAveragePooling1D
model = Sequential()     
model.add(Embedding(input_dim=45655, output_dim=256,input_length=None))
model.add(Conv1D(128,2,activation='relu'))

model.add(Conv1D(64,2,activation='relu'))

model.add(LSTM(128, return_sequences=True)) # bidirectional(양방향 LSTM)
model.add(Dropout(0.4))
model.add(Conv1D(128,2,activation='relu'))

model.add(Conv1D(64,2,activation='relu'))

model.add(MaxPooling1D(2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.4))
model.add(Conv1D(128,2,activation='relu'))
model.add(Conv1D(64,2,activation='relu'))

model.add(LSTM(128, return_sequences=True))

model.add(MaxPooling1D(2))
model.add(LSTM(128))
model.add(Dense(7, activation= 'softmax'))

'''

'''
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', patience=50, mode='max', verbose=3)
model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics=['acc'])
model.fit(x_train,y_predic,epochs=1000,batch_size=64,validation_split=0.4,callbacks=[es])

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

npy 구분 후
Loss : 0.49315738677978516
Acc : 0.8840846419334412

Bidirectional 적용 후
Loss : 0.8296099305152893
Acc : 0.8960660696029663

valdationdata
Loss : 0.5655934810638428
Acc : 0.8905462622642517
'''

topic = []
for i in  range(len(y_pred)):
     topic.append(np.argmax(y_pred[i]))
sub['topic_idx'] = topic
sub.to_csv('./save/sample_submission.csv',index=False)






