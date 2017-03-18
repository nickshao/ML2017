import numpy as np
import csv
import math
import sys
def read_train(filename):
    file = open(filename,'r',encoding="big5")
    csvCursor = csv.reader(file)
    count = -1
    final_data = np.empty([5652,8])
    y_data = np.empty([5652,1])
    data = np.empty([18,5760], dtype=float)
    for row in csvCursor:
        if count == -1:
            count += 1
            continue
        for item in range(3,27):
            if row[item] == 'NR':
                row[item] = 0.0
            rem = count % 18
            div = count//18
            data[rem][(item-3)+(div*24)] = row[item]
        count += 1
    for mon in range(12):
        for sample in range(471):
            #for fea in range(18):
            final_data[mon*471+sample][0:8] = data[9][mon*480+sample+1:mon*480+sample+9]
            y_data[mon*471+sample] = data[9][mon*480+sample+9]
    return final_data, y_data
def cal_related():
    data = read_train(sys.argv[1])
    for i in range(18):
        div = np.sqrt(np.sum(data[i]**2)*5760-(np.sum(data[i]))**2)*np.sqrt(np.sum(data[9]**2)*5760-(np.sum(data[9]))**2)
        son = 5760*np.sum(data[i]*data[9]) - np.sum(data[i])*np.sum(data[9])
        print('cal_related i: ', i,son/div)
def read_test(filename):
    file = open(filename,'r',encoding="big5")
    csvCursor = csv.reader(file)
    data = np.empty([18,2160])
    y = np.empty([240,1])
    test_data = np.empty([240,8], dtype=float)
    fak_test = np.empty([240,8], dtype=float)
    count = 0
    for row in csvCursor:
        for item in range(2,11):
            if row[item] == 'NR':
                row[item] = 0.0
            rem = count % 18
            div = count//18
            data[rem][(item-2)+(div*9)] = row[item]
        count += 1
    for sample in range(240):
        test_data[sample][0:8] = data[9][9*sample+1:9*sample+9]
        fak_test[sample][0:8] = data[9][9*sample:9*sample+8]
        y[sample] = data[9][9*sample+8]
    return test_data, fak_test,y

def train():
    final_data,y_data = read_train(sys.argv[1])
    test_data, fak_test, y = read_test(sys.argv[2])
    w = np.random.random_sample((8,1))
    fin = np.empty([5891,8], dtype=float)
    fin_y = np.empty([5891,1], dtype=float)
    fin = np.concatenate((final_data,fak_test), axis = 0)
    fin_y = np.concatenate((y_data,y), axis = 0)
    scalar = np.empty([5891,1], dtype=float)
    w = [[0.37773179],[0.66850027],[0.57437727],[0.94694371],[0.97021427],[0.10314471],[0.54085996],[0.61041322]]
    b = 0
    lr = 0.8
    sumQ = np.empty([942,1], dtype=float)
    iteration = 1000
    w_lr = np.zeros([8,], dtype=float)
    b_lr = 0.0
    lamb = 10
    for i in range(iteration):
        b_grad = 0
        w_grad = np.zeros([8,], dtype=float)
        #print('iter:',i)
        scalar = 2.0*np.subtract(fin_y[0:5891],np.add(np.matmul(fin[0:5891],w),b))
        w_grad = np.multiply(np.sum(np.multiply(scalar,fin[0:5891]),axis=0),-1)
        b_grad = np.sum(np.multiply(scalar,-1))
        w_lr = w_lr + w_grad**2
        b_lr = b_lr + b_grad**2
        w = w - (lr/np.sqrt(w_lr)*w_grad).reshape(-1,1)
        b = b - lr/np.sqrt(b_lr) * b_grad
        sumQ = np.sum(np.subtract(fin_y[4709:5651],np.add(np.matmul(fin[4709:5651],w),b))**2)
        sumQ = sumQ/942
    return b,w

def test():
    data,fak_test, y = read_test(sys.argv[2])
    ANS = [['id','value']]
    bias = 0.0
    weight = np.zeros([8,1], dtype=float)
    final_ans = np.zeros([240,1],dtype=float)
    bias, weight = train()
    final_ans = np.add(np.matmul(data,weight),bias).reshape(-1,)
    f = open(sys.argv[3],'w')
    for item in range(240):
        str_val = 'id_' + str(item)
        ans = [str_val,final_ans[item]]
        ANS.append(ans)
    w = csv.writer(f)
    w.writerows(ANS)
    f.close()
if __name__ == '__main__':
    test()
    #cal_related()
