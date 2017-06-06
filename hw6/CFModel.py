import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dense, Lambda, Dropout
from keras.layers.merge import Dot
from keras.models import Sequential
from keras import backend as K

class CFModel(Sequential):

    def __init__(self, n_users, m_items, emb_dim, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, emb_dim, input_length=1))
        P.add(Reshape((emb_dim,)))
        Q = Sequential()
        Q.add(Embedding(m_items, emb_dim, input_length=1))
        Q.add(Reshape((emb_dim,)))
        super(CFModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='dot', dot_axes=1))
class DeepModel(Sequential):
    def __init__(self, n_users, m_items, emb_dim, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, emb_dim, input_length=1))
        P.add(Reshape((emb_dim, )))
        Q = Sequential()
        Q.add(Embedding(m_items, emb_dim, input_length=1))
        Q.add(Reshape((emb_dim, )))
        super(DeepModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='concat'))
        self.add(Dense(150, activation='relu'))
        self.add(Dropout(0.15))
        self.add(Dense(1, activation='relu'))
