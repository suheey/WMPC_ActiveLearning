import numpy as np
import os, time
import pandas as pd
import pickle as pkl

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score, recall_score, precision_score
from skimage.transform import resize

from scipy.spatial import distance
import tqdm, csv


import tensorflow as tf
import keras
from keras import layers, Input, models, Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, Activation, concatenate, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import matplotlib.pyplot as plt



# Data Load
print(':: load data')
with open('WM.pkl','rb') as f:
    [fea_all, fea_all_tst, X_rs, X_tst, y, Y_tst] = pkl.load(f)

    
print(fea_all.shape, fea_all_tst.shape, len(X_rs), len(X_tst), y.shape, Y_tst.shape)




# Number of each class
unique, counts = np.unique(np.where(y==1)[1], return_counts=True)
num_trn= dict(zip(unique, counts))
print("Number of Train Class", num_trn)

unique, counts = np.unique(np.where(Y_tst==1)[1], return_counts=True)
num_tst= dict(zip(unique, counts))
print("Number of Test Class", num_tst)



def _permutation(set):
    permid = np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i] = set[i][permid]

    return set


# # CNN Model

# read training data
X_U = X_rs
X_test = X_tst
y_U = y
y_test = Y_tst

# assemble initial data
n_initial = 100
initial_idx = np.random.choice(range(len(X_U)), size=n_initial, replace=False)
X_train = X_U[initial_idx]
y_train = y_U[initial_idx]

# generate the pool
# remove the initial data from the training dataset
X_U = np.delete(X_U, initial_idx, axis=0)
y_U = np.delete(y_U, initial_idx, axis=0)

print(X_U.shape, X_train.shape, X_test.shape)


def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)

def create_model(mc=False):
    dim = 64
    input_wbm_tensor = Input((dim, dim, 1))
    conv_1 = Conv2D(16, (3,3), activation='relu', padding='same')(input_wbm_tensor)
    pool_1 = MaxPool2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_1)
    conv_2 = Conv2D(32, (3,3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_2)
    conv_3 = Conv2D(64, (3,3), activation='relu', padding='same')(pool_2)
    pool_3 = MaxPool2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_3)
    GAP = GlobalAveragePooling2D()(pool_3)
    dense_1 = Dense(128, activation='tanh')(GAP)
    dense_2 = Dense(128, activation='tanh')(dense_1)
    dp=get_dropout(dense_2, p=0.5, mc=mc)
    prediction = Dense(9, activation='softmax')(dp)

    model = models.Model(input_wbm_tensor, prediction)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

model = create_model(mc=False)
mc_model=create_model(mc=True)


unique, counts = np.unique(np.where(y_train==1)[1], return_counts=True)
num_trn= dict(zip(unique, counts))
print("Number of Train Class", num_trn)


# active learning loop
n_queries = 50
epoch=10
batch_size = 20

n_query=100
f1_mac=[]
f1_mic=[]
f1_p=[]
f1_r=[]

for idx in range(n_queries):
    if idx==0:
        print('Query no. %d' % (idx + 1))
        h = model.fit(X_train, y_train,epochs=epoch,batch_size=batch_size)
        
    else:
        print('Query no. %d' % (idx + 1))
        
        #Uncertainty Estimation
        h_mc=mc_model.fit(X_train, y_train,epochs=10,batch_size=batch_size)
        mc_pd=[]
        for i in tqdm.tqdm(range(10)):
            y_p=mc_model.predict(X_U, batch_size=20)
            mc_pd.append(y_p)
        
        mc_conf=np.mean(np.std(mc_pd,0),1)
        query_idx=np.argsort(-mc_conf)[:n_query]

        unique, counts = np.unique(np.where(y_U[query_idx]==1)[1], return_counts=True)
        num_trn= dict(zip(unique, counts))
        print("Number of Train Class", num_trn)
        
        # remove queried instance from pool
        X_query=X_U[query_idx]
        y_query=y_U[query_idx]
        
        X_U= np.delete(X_U, query_idx, axis=0)
        y_U =np.delete(y_U, query_idx, axis=0)
        
        X_train=np.concatenate([X_train, X_query], 0)
        y_train=np.concatenate([y_train, y_query], 0)

        history = model.fit(X_train, y_train,epochs=epoch,batch_size=batch_size)
    
    y_hat=np.argmax(model.predict(X_tst), axis=1)
    y_true = np.argmax(Y_tst,axis=1)

    # performance metric
    print('Macro average F1 score:', f1_score(y_true, y_hat, average='macro'))
    print('Micro average F1 score:', f1_score(y_true, y_hat, average='micro'))

    f1_mac.append(f1_score(y_true, y_hat, average='macro'))
    f1_mic.append(f1_score(y_true, y_hat, average='macro'))
    f1_p.append(recall_score(y_true, y_hat, average='macro'))
    f1_r.append(precision_score(y_true, y_hat, average='macro'))


with open( 'f1_mac', 'w') as output:
    out = csv.writer(output)
    out.writerows(map(lambda x: [x], f1_mac))
    
with open( 'f1_mic', 'w') as output:
    out = csv.writer(output)
    out.writerows(map(lambda x: [x], f1_mac))
    
with open( 'f1_p', 'w') as output:
    out = csv.writer(output)
    out.writerows(map(lambda x: [x], f1_mac))

with open( 'f1_r', 'w') as output:
    out = csv.writer(output)
    out.writerows(map(lambda x: [x], f1_mac))





