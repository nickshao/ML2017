from PIL import Image
import math
from numpy import linalg as la
import matplotlib.pyplot as plt
import numpy as np
alphabet = ['A','B','C','D','E','F','G','H','I','J']
img_str = []
for alpha in alphabet:
    for number in range(10):
        _str = 'image_data/' + alpha+ '0' + str(number) + '.bmp'
        img_str.append(_str)
x = np.array([np.array(Image.open(fname)) for fname in img_str])
x = x.reshape(100, 64*64)
avr_face = np.mean(x, axis = 0)
#avr_face = avr_face.reshape(64,64)
center_data = x - avr_face
center = np.transpose(center_data)
#print('center_data',center_data.shape) #(4096, 100)
U, s, V = la.svd(center)
for i in range(4096):
    top_5_eigenface = U[:,0:i+1]
    eigen_coe = np.matmul(center_data, top_5_eigenface)
    final_img = np.matmul(eigen_coe, np.transpose(top_5_eigenface))
    final_img  = final_img + avr_face
    result = math.sqrt(np.sum((x - final_img) * (x - final_img))/(100*4096))
    result /= 256
    print(i+1,': ',result)
    if result < 0.01:
        print('answer:',i+1)
        break

#top_5_eigenface = U[:,0:9]
'''
eigen_coe = np.matmul(center_data, top_5_eigenface)
final_img = np.matmul(eigen_coe, np.transpose(top_5_eigenface))
final_img = final_img + avr_face

fig = plt.figure(figsize=(10,10))
for no_eigen in range(100):
    ax = fig.add_subplot(10,10,no_eigen+1)
    ax.imshow(final_img[no_eigen].reshape(64,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
fig.suptitle('reconstruct 100 faces')
fig.savefig('construct.png',dpi=300)
'''
'''
fig = plt.figure(figsize=(3,3))
for no_eigen in range(9):
    ax = fig.add_subplot(3,3,no_eigen+1)
    ax.imshow(top_5_eigenface[:, no_eigen].reshape(64,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
fig.savefig('eigenface.png',dpi=300)
'''
