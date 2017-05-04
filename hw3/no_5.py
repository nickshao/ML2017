import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
nb_filter = 64
NUM_STEPS = 100
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    for epoch in range(NUM_STEPS):
        print('epoch:',epoch)
        loss_value, grads_value = iter_func([input_image_data])
        input_image_data += grads_value * num_step 
    input_image_data = input_image_data.reshape(1, 48, 48)
    return input_image_data[0]

def main():
    emotion_classifier = load_model("weight_best.hdf5")
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[:])
    input_img = emotion_classifier.input

    name_ls = ["conv2d_1"]
    collect_layers = [ layer_dict[name].output for name in name_ls ]
    for c in collect_layers:
        filter_imgs = []
        for filter_idx in range(nb_filter):
            num_step = 1
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img], [target, grads])
            ###
            "You need to implement it."
            filter_imgs.append(grad_ascent(num_step, input_img_data, iterate))
            ###
        fig = plt.figure(figsize=(14, 8))
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(filter_imgs[i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

            plt.tight_layout()
        fig.suptitle('Filters of layer {} (# Ascent Epoch 100 )')
        fig.savefig('activation.png')

if __name__ == "__main__":
    main()
