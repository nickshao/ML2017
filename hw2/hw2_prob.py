import numpy as np
import csv
import math
import random
import sys
train_size = 32561
test_size = 16281
w_size = 106
def read_train_feature(filename):
    file = open(filename, 'r')
    csvCursor = csv.reader(file)
    count = -1
    data = np.empty([32561,w_size], dtype=float)
    for row in csvCursor:
        if count == -1:
            count += 1
            continue
        data[count] = row
        count += 1
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
        data[count] = row
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
def divide(train,y):
    x_count = 0 #24720
    y_count = 0 #7841
    cls1_data = np.empty([7841,w_size], dtype=float)
    cls2_data = np.empty([24720,w_size], dtype=float)
    data = feature_scaling(sys.argv[1],sys.argv[3])
    y_data = read_y(y)
    for i in range(train_size):
        if y_data[i] == 1:         #class 1
            cls1_data[x_count] = data[i]
            x_count += 1
        elif y_data[i] == 0:       #class 2
            cls2_data[y_count] = data[i]
            y_count += 1
    cls1_data = cls1_data.transpose()
    cls2_data = cls2_data.transpose()
    mu_1 = np.mean(cls1_data,axis=1).reshape(-1,1)
    mu_2 = np.mean(cls2_data,axis=1).reshape(-1,1)
    co_1 = np.matmul((cls1_data-mu_1),(cls1_data-mu_1).transpose())
    co_2 = np.matmul((cls2_data-mu_2),(cls2_data-mu_2).transpose())
    co_1 /= 7841
    co_2 /= 24720
    co = np.add(np.multiply(co_1,7841/32561),np.multiply(co_2,24720/32561))
    weight = np.matmul((mu_1-mu_2).transpose(),np.linalg.inv(co))
    bias = -0.5*np.matmul(np.matmul(mu_1.transpose(),np.linalg.inv(co)),mu_1)+0.5*np.matmul(np.matmul(mu_2.transpose(),np.linalg.inv(co)),mu_2)+np.log(7841/24720)
    return weight,bias
def test():
    data = feature_scaling(sys.argv[1],sys.argv[3])
    ANS = [['id','label']]
    w,b = divide(sys.argv[1],sys.argv[2])
    final_ans = np.add(np.matmul(data[train_size:train_size+test_size],w.transpose()),b)
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
    test()

