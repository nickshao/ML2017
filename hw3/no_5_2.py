import numpy as np
import csv
import sys
import h5py
from PIL import Image
import keras.backend as K
import matplotlib.pyplot as plt
from termcolor import colored,cprint
from keras.models import load_model
from keras.utils import np_utils
pixel_row = 48
pixel_col = 48
def read_train_feature(filename):
    X_train,Y_train = [],[]
    file = open(filename,'r')
    count = -1
    csvCursor = csv.reader(file)
    for row in csvCursor:
        print('count',count)
        if count == -1:
            count += 1
            continue
        image = [ int(pixel) for pixel in row[1].split(' ')]
        cls = int(row[0])
        X_train.append(image)
        Y_train.append(cls)
        count += 1
    X_train = np.array(X_train).reshape(len(X_train),pixel_row,pixel_col)
    #X_train = np.array(X_train).reshape(len(X_train),pixel_row,pixel_col,1)
    X_train = X_train.astype('float32')
    X_train /= 255
    Y_train = np.array(Y_train) 
    Y_train = np_utils.to_categorical(Y_train,7)
    return X_train, Y_train
def model_exec():
    X_train,Y_train = read_train_feature(sys.argv[1]) 
    train_x = X_train[0:20709]
    val_x = X_train[20709:28709]
    train_y = Y_train[0:20709]
    val_y = Y_train[20709:28709]
    #val_x *= 255
    _in = val_x[4].reshape(1,48,48,1)
    emotion_classifier = load_model('weight_best.hdf5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[:])

    input_img = emotion_classifier.input
    name_ls = ["conv2d_1"]
    collect_layers = [K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls]
    
    ##deal with photo
    for cnt, fn in enumerate(collect_layers):
        im = fn([_in, 0])
        fig = plt.figure(figsize=(14, 8))
        nb_filter = 64
        print('shape: ',im[0][0,:,:,0].shape)
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16,16,i+1)
            ax.imshow(im[0][0, :, :, i],cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer 0')
        fig.savefig('out_layer.png')
if __name__ == '__main__':
    model_exec()
