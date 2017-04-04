import numpy as np
import csv
import math
import random
import sys
train_size = 32561
test_size = 16281
w_size = 113
def read_train_feature(filename):
    file = open(filename,'r')
    csvCursor = csv.reader(file)
    count = -1
    data = np.empty([32561,w_size], dtype=float)
    for row in csvCursor:
        if count == -1:
            count += 1
            continue
        data[count][0:106] = row
        #data[count][1:105] = row[2:106]
        
        data[count][106] = float(row[3])**2
        data[count][107] = float(row[0])**2
        data[count][108] = float(row[2])**2
        data[count][109] = float(row[4])**2
        data[count][110] = float(row[5])**2
        data[count][111] = float(row[3])**4

        data[count][112] = float(row[2])*float(row[5])*float(row[0])
        #data[count][109] = float(row[25])*5
        #data[count][111] = float(row[63])*2
        
        count += 1
    #print('count', count) ##32561
    return data
def read_test_feature(filename):
    file = open(filename,'r')
    csvCursor = csv.reader(file)
    count = -1
    data = np.empty([16281,w_size], dtype=float)
    for row in csvCursor:
        if count == -1:
            count += 1
            continue
        data[count][0:106] = row
        #data[count][1:105] = row[2:106]
        
        data[count][106] = float(row[3])**2
        data[count][107] = float(row[0])**2
        data[count][108] = float(row[2])**2
        data[count][109] = float(row[4])**2
        data[count][110] = float(row[5])**2
        data[count][111] = float(row[3])**3
        data[count][112] = float(row[3])*float(row[5])*float(row[0])
        #data[count][109] = float(row[25])*5
        #data[count][111] = float(row[63])*2
        
        count += 1
    return data
def read_y(filename):
    file = open(filename,'r')
    csvCursor = csv.reader(file)
    count = 0
    data = np.empty([32561,1], dtype=float)
    for row in csvCursor:
        data[count] = row
        count += 1
    return data
def feature_scaling(train,test):
    train_data = read_train_feature(train)
    test_data = read_test_feature(test)
    data = np.concatenate((train_data,test_data),axis=0)
    data = (data-np.mean(data,axis=0))/np.std(data,axis=0)
    return data
def train():
    y_data = read_y(sys.argv[2])
    data = feature_scaling(sys.argv[1],sys.argv[3])
    w = np.zeros([w_size,1],dtype=float)
    w_lr = np.zeros([w_size,], dtype=float)
    b = -1.5
    b_lr = 0.0
    scalar = np.empty([32561,1], dtype=float)
    cross_z = np.empty([32561,1], dtype=float)
    iteration = 1000 #3000
    lr = 0.5

    for i in range(iteration):
        #print('i:', i)
        b_grad = 0
        w_grad = np.zeros([w_size,], dtype=float)
        scalar = 1/(1+np.exp(np.multiply(np.add(np.matmul(data[0:train_size],w),b),-1)))
        scalar = np.subtract(y_data,scalar)
        cross_z = 1/(1+np.exp(np.multiply(np.add(np.matmul(data[0:train_size],w),b),-1)))
        #print(scalar.shape)
        w_grad = np.multiply(np.sum(np.multiply(scalar,data[0:train_size]),axis=0),-1)
        b_grad = np.sum(np.multiply(scalar,-1))
        w_lr = w_lr + w_grad**2
        b_lr = b_lr + b_grad**2
        w = w - (lr/np.sqrt(w_lr)*w_grad).reshape(-1,1)
        b = b - (lr/np.sqrt(b_lr)) * b_grad
        Loss = np.sum(np.add(np.multiply(y_data,np.log(cross_z+1e-15)),np.multiply(1-y_data,np.log(1-cross_z+1e-15))),axis=0)
        Loss = Loss*(-1)
        #print('Loss:', Loss/train_size)
    #print('weight: ',w)
    #print('bias: ',b)
    return w,b
def test():
    data = feature_scaling(sys.argv[1],sys.argv[3])
    ANS = [['id','label']]
    w,b = train()
    final_ans = np.add(np.matmul(data[train_size:train_size+test_size],w),b)
    f = open(sys.argv[4],'w')
    for item in range(16281):
        if(final_ans[item] >= 0):
            ans = [item+1,1]
        else:
            ans = [item+1,0]
        ANS.append(ans)
    w = csv.writer(f)
    w.writerows(ANS)
    f.close()
if __name__ == '__main__':
    #read_train_feature('X_train')
    test()
