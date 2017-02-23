import numpy as np
import sys
x = np.loadtxt(sys.argv[1],dtype='int',delimiter=',', ndmin=2)
y = np.loadtxt(sys.argv[2],dtype='int',delimiter=',', ndmin=2)
z = np.matmul(x,y)
out = np.reshape(z,-1)
np.savetxt('ans_one.txt',np.sort(out),fmt='%d')
