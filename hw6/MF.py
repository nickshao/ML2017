import pandas as pd
import numpy as np
import csv
from CFModel import CFModel
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
#parser.add_argument('--train_file', type=str, default='./train.csv')
parser.add_argument('--test_file', type=str, default='./test.csv')
parser.add_argument('--out_file', type=str, default='./output.csv')
args = parser.parse_args()

##set some arguments
batch_size = 150
nb_epoch = 100
RNG_SEED = 1446557
emb_dim = 150
def read_test(test_path):
    test_data = pd.read_csv(test_path, usecols = ['TestDataID','UserID','MovieID'])
    test_users = test_data['UserID'].values
    print('test users shape: ',test_users.shape)
    test_movies = test_data['MovieID'].values
    print('test movies shape: ',test_movies.shape)
    return test_users, test_movies
if __name__ == '__main__':
    #train_data, max_userid, max_movieid = read_train(args.train_file)
    #Users, Movies, Ratings = shuffle_data(train_data)
    test_users, test_movies = read_test(args.test_file+'test.csv')
    ###training start!
    print('Training start!')
    model = CFModel(6041, 3953, emb_dim)
    model.compile(loss='mse', optimizer='adagrad')
    if args.train:
        earlystopping = EarlyStopping(monitor='val_loss', patience = 3)
        checkpoint = ModelCheckpoint(filepath = 'best.hdf5', save_best_only=True, save_weights_only=True)
        model.fit([Users, Movies], Ratings, epochs = nb_epoch,validation_split=0.1, batch_size = batch_size, verbose = 1, callbacks=[earlystopping, checkpoint])
    else:
        print('Test!')
        model.load_weights('MF.hdf5')
        predict_ans = model.predict([test_users, test_movies], batch_size= batch_size)
        predict_ans = predict_ans.reshape(-1, )
        print('predict_ans: ', predict_ans.shape)
        f = open(args.out_file, 'w')
        ANS = [['TestDataID','Rating']]
        for _id, label in enumerate(predict_ans, 1):
            ans = [_id, label]
            ANS.append(ans)
        w = csv.writer(f)
        w.writerows(ANS)
        f.close()
