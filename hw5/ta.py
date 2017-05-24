import numpy as np
import string
import sys
import argparse
import keras.backend as K 
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense,Dropout, Flatten
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--out_file', type=str, default='./output.csv')
parser.add_argument('--train_file', type=str, default='./train_data.csv')
parser.add_argument('--test_file', type=str, default='./test_data.csv')
args = parser.parse_args()

##train_path = sys.argv[1]
##test_path = sys.argv[2]
##output_path = sys.argv[3]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 1000
batch_size = 128


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
        if training :
            all_tag = np.array(tags_list)
            print('all_tag:',all_tag)
            np.save('all_tag.npy',all_tag)
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.3
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))

#########################
###   Main function   ###
#########################
def main():
    ### read training and testing data
    #(Y_data,X_data,tag_list) = read_data(args.train_file,True)
    (_, X_test,_) = read_data(args.test_file,False)
    #all_corpus = X_data + X_test
    #print ('Find %d articles.' %(len(all_corpus)))
            
    ### tokenizer for all data
    '''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    word_index = tokenizer.word_index
    '''
    ### convert word sequences to index sequence
    '''
    print ('Convert to index sequences.')
    train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    '''
    ### padding to equal length
    '''
    print ('Padding sequences.')
    train_sequences = pad_sequences(train_sequences)
    max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    np.save('pad_seq.npy',test_sequences)
    '''     
    ###
    '''
    train_tag = to_multi_categorical(Y_data,tag_list) 
            
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
    '''        
    ## get mebedding matrix from glove
    '''
    print ('Get embedding dict from glove.')
    embedding_dict = get_embedding_dict('glove.6B.%dd.txt'%embedding_dim)
    print ('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    print('num_words:',num_words)
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)
    '''
    if args.train:
        print ('Building model.')
        model = Sequential()
        model.add(Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_article_length,
                            trainable=False))
        model.add(GRU(128,recurrent_dropout=0.25,dropout=0.25))
        #model.add(LSTM(64,activation='tanh',dropout=0.2))
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(38,activation='sigmoid'))
        #model.summary()

        adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
        model.compile(loss='binary_crossentropy',
                          optimizer=adam,
                          metrics=[f1_score])
           
        earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(filepath='best.hdf5',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         monitor='val_f1_score',
                                         mode='max')
        hist = model.fit(X_train, Y_train, 
                            validation_data=(X_val, Y_val),
                            epochs=nb_epoch, 
                            batch_size=batch_size,
                            callbacks=[earlystopping,checkpoint])
    else:
        all_tag = np.load('all_tag.npy')
        print(all_tag)
        data = np.load('pad_seq.npy')
        print('data',data.shape)
        model = Sequential()
        model.add(Embedding(51867,
                            embedding_dim,
                            #weights=[embedding_matrix],
                            #input_length=max_article_length,
                            trainable=False))
        model.add(GRU(128,recurrent_dropout=0.25,dropout=0.25))
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(38,activation='sigmoid'))
        model.load_weights("15_best.hdf5")
        Y_pred = model.predict(data)
        '''
        print('Y_pred shape:' ,Y_pred.shape)
        print('Y_pred:', Y_pred[3])
        '''
        thresh = 0.3
        print('###############################3')
        with open(args.out_file,'w') as output:
            print ('\"id\",\"tags\"',file=output)
            Y_pred_thresh = (Y_pred > thresh).astype('int')
            for index,labels in enumerate(Y_pred_thresh):
                labels = [all_tag[i] for i,value in enumerate(labels) if value==1 ]
                labels_original = ' '.join(labels)
                print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
