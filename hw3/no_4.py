import numpy as np
import csv
import sys
import h5py
from PIL import Image
import keras.backend as K
import matplotlib.pyplot as plt
from termcolor import colored,cprint
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import optimizers
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
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
    val_x *= 255
    _in = val_x[2].reshape(1,48,48,1)
    emotion_classfier = load_model('weight_best.hdf5')
    input_img = emotion_classfier.input

    val_proba = emotion_classfier.predict(_in)
    pred = val_proba.argmax(axis=-1)
    target = K.mean(emotion_classfier.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads)))+1e-5)
    fn = K.function([input_img, K.learning_phase()],[grads])
    
    heatmap = np.array(fn([_in, 0])).reshape(pixel_col, pixel_row)
    
    heatmap = np.absolute(heatmap)
    _min = np.amin(heatmap)
    _max = np.amax(heatmap)
    heatmap -= _min
    heatmap /= (_max - _min)
    print('heatmap shape:',heatmap)
    
    thres = 0.1
    see = val_x[2].reshape(48,48)
    see[np.where(heatmap <= thres)] = np.mean(see)

    plt.figure()
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('heatmap.png',dpi=100)
    
    plt.figure()
    plt.imshow(see, cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('see.png',dpi=100)
    '''
    print('val_x:',val_x[2])
    im = Image.fromarray(val_x[2])
    im = im.convert('RGB')
    im.save('origin.jpg')
    '''
    #############datagen########################
    '''
    datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=0,
            width_shift_range=0.1,
            zca_whitening=False,
            height_shift_range=0.1,
            horizontal_flip=True)
    datagen.fit(X_train)
    batches = 0
    for X_batch, Y_batch in datagen.flow(train_x,train_y, batch_size=200):
        #print('batches:',batches)
        train_x = np.concatenate((train_x,X_batch), axis = 0)
        train_y = np.concatenate((train_y,Y_batch), axis = 0)
        batches += 1
        if batches >= 1000:
            break
    #print('X_train shape:',X_train.shape)
    #X_test = read_test_feature('test.csv')
    '''
    '''
    model = Sequential()
    model.add(Conv2D(64,(3,3), activation='relu',input_shape = (pixel_row, pixel_col, 1)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dropout(0.5))
    model.add(Dense(7,activation='softmax'))

    #model.load_weights("weight_best.hdf5")
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    #for checkpoint
    
    filepath="weight_test.hdf5"
    checkpoint=ModelCheckpoint(filepath, monitor='val_acc', verbose = 1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(X_train,Y_train,validation_split=0.1, batch_size=200, epochs=15, callbacks=callbacks_list, verbose=0)
    '''
if __name__ == '__main__':
    model_exec()
