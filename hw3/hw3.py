import numpy as np
import csv
import sys
import h5py
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
    X_train = np.array(X_train).reshape(len(X_train),pixel_row,pixel_col,1)
    X_train = X_train.astype('float32')
    X_train /= 255
    Y_train = np.array(Y_train) 
    Y_train = np_utils.to_categorical(Y_train,7)
    return X_train, Y_train
def model_exec():
    X_train,Y_train = read_train_feature(sys.argv[1]) 

    #############datagen########################
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
    for X_batch, Y_batch in datagen.flow(X_train,Y_train, batch_size=200):
        #print('batches:',batches)
        X_train = np.concatenate((X_train,X_batch), axis = 0)
        Y_train = np.concatenate((Y_train,Y_batch), axis = 0)
        batches += 1
        if batches >= 1000:
            break
    #print('X_train shape:',X_train.shape)
    #X_test = read_test_feature('test.csv')
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
    
if __name__ == '__main__':
    model_exec()
