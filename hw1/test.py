import numpy as np
import csv
import math
import random
import statistics        
def read_train(filename):
    file = open(filename,'r',encoding="big5")
    csvCursor = csv.reader(file)
    count = -1
    data = np.empty([19,5760], dtype=float)
    for row in csvCursor:
        if count == -1:
            count += 1
            continue
        for item in range(3,27):
            if row[item] == 'NR':
                row[item] = 0.0
            rem = count % 18
            div = count//18
            if rem == 9:
                data[18][(item-3)+(div*24)] = float(row[item])**2
            data[rem][(item-3)+(div*24)] = row[item]
        count += 1
    return data
def read_test(filename):
    file = open(filename,'r',encoding="big5")
    csvCursor = csv.reader(file)
    data = np.empty([19,2160])
    count = 0
    for row in csvCursor:
        for item in range(2,11):
            if row[item] == 'NR':
                row[item] = 0.0
            rem = count % 18
            div = count//18
            if rem == 9:
                data[18][(item-2)+(div*9)] = float(row[item])**2
            data[rem][(item-2)+(div*9)] = row[item]
        count += 1
    return data
def feature_scaling():
    train_data = read_train('train.csv')
    test_data = read_test('test_X.csv')
    all_data = np.empty([19,7920])
    all_data = np.concatenate((train_data,test_data),axis = 1)
    mean = np.empty([19,1], dtype=float)
    stdev = np.empty([19,1], dtype=float)
    for i in range(19):
        mean[i] = statistics.mean(all_data[i])
        stdev[i] = statistics.stdev(all_data[i])
        all_data[i] = (all_data[i] - mean[i])/stdev[i]
    return all_data
def del_train():
    data = feature_scaling()
    ori_data = read_train('train.csv')
    final_data = np.empty([5652,72])
    y_data = np.empty([5652,1])
    for mon in range(12):
        for sample in range(471):
            for fea in range(4):
                final_data[mon*471+sample][fea*8:fea*8+8] = data[fea+7][mon*480+sample+1:mon*480+sample+9]
            for fea in range(4,8):
                final_data[mon*471+sample][fea*8:fea*8+8] = data[fea+10][mon*480+sample+1:mon*480+sample+9]
            final_data[mon*471+sample][64:72] = data[18][mon*480+sample+1:mon*480+sample+9]
            y_data[mon*471+sample] = ori_data[9][mon*480+sample+9]
    return final_data, y_data
def del_test():
    ori_data = read_test('test_X.csv')
    data = feature_scaling()
    test_data = np.empty([240,72])
    fak_test = np.empty([240,72])
    y = np.empty([240, 1])
    for sample in range(240):
        for fea in range(4):
            test_data[sample][fea*8:fea*8+8] = data[fea+7][9*sample+5761:9*sample+5769]
            fak_test[sample][fea*8:fea*8+8] = data[fea+7][9*sample+5760:9*sample+5768]        
        for fea in range(4,8):
            test_data[sample][fea*8:fea*8+8] = data[fea+10][9*sample+5761:9*sample+5769]
            fak_test[sample][fea*8:fea*8+8] = data[fea+10][9*sample+5760:9*sample+5768]
        fak_test[sample][64:72] = data[18][9*sample+5760:+9*sample+5768]
        test_data[sample][64:72] = data[18][9*sample+5761:9*sample+5769]
        y[sample] = ori_data[9][9*sample+8]
    return test_data, fak_test, y
def train():
    final_data,y_data = del_train()
    test_data, fak_test ,y= del_test()
    fin_y = np.empty([5892, 1], dtype=float)
    fin = np.empty([5892,72], dtype=float)
    fin_y = np.concatenate((y_data,y), axis = 0)
    fin = np.concatenate((final_data,fak_test), axis = 0)
    w = np.random.random_sample((72,1))*0.8
    scalar = np.empty([5892,1], dtype=float)
    print('w',w)
    b = random.uniform(0,1)
    lr = 2.5
    sumQ = np.empty([5892,1], dtype=float)
    iteration = 10000
    w_lr = np.zeros([72,], dtype=float)
    b_lr = 0.0
    lamb = 100
    for i in range(iteration):
        b_grad = 0
        w_grad = np.zeros([72,], dtype=float)
        print('iter:',i)
        scalar = 2.0*np.subtract(fin_y[0:5892],np.add(np.matmul(fin[0:5892],w),b))
        w_grad = np.multiply(np.sum(np.multiply(scalar,fin[0:5892]),axis=0),-1)
        b_grad = np.sum(np.multiply(scalar,-1))
        w_grad = w_grad + 2.0*lamb*np.sum(w)
        w_lr = w_lr + w_grad**2
        b_lr = b_lr + b_grad**2
        w = w - (lr/np.sqrt(w_lr)*w_grad).reshape(-1,1)
        b = b - lr/np.sqrt(b_lr) * b_grad
        sumQ = np.sum(np.subtract(fin_y[0:5892],np.add(np.matmul(fin[0:5892],w),b))**2)
        sumQ = sumQ/5892
        print('RMSE:',math.sqrt(sumQ))
    return b,w

def test():
    data, fak_test, y = del_test()
    ANS = [['id','value']]
    bias = 0.0
    weight = np.zeros([72,1], dtype=float)
    final_ans = np.zeros([240,1], dtype=float)
    bias, weight = train()
    f = open('predict.csv','w')
    final_ans = np.add(np.matmul(data,weight),bias).reshape(-1,)
    for item in range(240):
        str_val = 'id_' + str(item)
        ans = [str_val,final_ans[item]]
        ANS.append(ans)
    w = csv.writer(f)
    w.writerows(ANS)
    f.close()
if __name__ == '__main__':
    test()
    # feature_scaling_train()
