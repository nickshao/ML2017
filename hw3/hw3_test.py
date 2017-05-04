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
def read_test_feature(filename):
    X_test = []
    file = open(filename,'r')
    count = -1
    csvCursor = csv.reader(file)
    for row in csvCursor:
        if count == -1:
            count += 1
            continue
        image = [int(pixel) for pixel in row[1].split(' ')]
        X_test.append(image)
    X_test = np.array(X_test).reshape(len(X_test),pixel_row,pixel_col,1)
    X_test = X_test.astype('float32')
    X_test /= 255
    return X_test
def model_exec():
    #print('argv 1:',sys.argv[1])
    X_test = read_test_feature(sys.argv[1])
    model = Sequential()
    model.add(Conv2D(64,(3,3), activation='relu',input_shape = (pixel_row, pixel_col, 1)))
    model.add(BatchNormalization())
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
    model.load_weights("weight_best.hdf5")
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    predict_ans = model.predict_classes(X_test, batch_size=200)

    f = open(sys.argv[2],'w')
    ANS = [['id','label']]
    for _id,label in enumerate(predict_ans):
        ans = [_id,label]
        ANS.append(ans)
    w = csv.writer(f)
    w.writerows(ANS)
    f.close()
if __name__ == '__main__':
    model_exec()
