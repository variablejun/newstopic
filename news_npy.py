import numpy as np
import pandas as pd
import re
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
def clean(sent):
     sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]","",sent)
     return sent_clean

dataset["title"] = dataset["title"].apply(lambda x : clean(x))
x_test["title"]  = x_test["title"].apply(lambda x : clean(x))


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

np.save('../_npy/news_train_data.npy',arr=x_train)
np.save('../_npy/news_test_data.npy',arr=x_test)
np.save('../_npy/news_topic_dict.npy',arr=y_predic)
