import numpy as np
import sys
x = np.loadtxt(sys.argv[1],dtype='int',delimiter=',')
y = np.loadtxt(sys.argv[2],dtype='int',delimiter=',')
z = np.matmul(x,y)
np.savetxt('ans_one.txt',np.sort(z),fmt='%d')
