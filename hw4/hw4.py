import numpy as np
import csv
import sys
import pickle
from math import log
from sklearn import linear_model
from argparse import ArgumentParser
def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--output',type=str, default=None)
    args = parser.parse_args()

    if args.train:
        #print('during training:\n')
        train_x = []
        train_y = []
        for i in range(3000):
            dim = np.random.randint(1,60)
            #print('i:', i)
            N = 6000
            layer_dims = [np.random.randint(60, 80), 100]
            data = gen_data(dim, layer_dims, N)
            U, s, V = np.linalg.svd(data)
            train_x.append(s)
            train_y.append(s[dim])
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        regr = linear_model.LinearRegression()
        regr.fit(train_x, train_y)
        #save file
        filename = "linear_model.sav"
        pickle.dump(regr, open(filename, 'wb'))
    else:
        #print(args.test_data)
        test_data = np.load(args.test_data)
        test_x = []
        result = []
        loaded_model = pickle.load(open('linear_model.sav', 'rb'))
        print('load success!')
        for i in range(200):
            #print('i:',i)
            x = test_data[str(i)]
            data = x[0:6000,:]
            #print(data.shape)
            U, s, V = np.linalg.svd(data)
            #print('s shape:',s.shape)
            threshold = loaded_model.predict(s)
            #print('threshold shape',threshold.shape)
            for j in range(100):
                if s[j] <= threshold:
                    print('dim:',j)
                    if j >= 60:
                        j = 60
                    result.append(log(j))
                    break
        f = open(args.output,'w') 
        ANS = [['SetId','LogDim']]
        for _id, pre_dim in enumerate(result):
            ans = [_id,pre_dim]
            ANS.append(ans)
        w = csv.writer(f)
        w.writerows(ANS)
        f.close()
