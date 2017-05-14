from argparse import ArgumentParser

import word2vec
import numpy as np
import nltk

parser = ArgumentParser()
parser.add_argument('--train', action='store_true')
args = parser.parse_args()

if args.train:
    word2vec.word2vec(
        train='text_data/all.txt',
        output='model/model.bin',
        size = 100,
        min_count = 6,
        window = 5,
        #sample = 1e-5,
        verbose=True)
else:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from adjustText import adjust_text
    #load model
    model = word2vec.load('model/model.bin')
    vocabs = []
    vecs = []
    plot_num = 500
    count = 0
    for vocab in model.vocab:
        if count >= plot_num:
            break
        vocabs.append(vocab)
        vecs.append(model[vocab])
        count += 1
    vecs = np.array(vecs)
    tsne = TSNE(n_components=2) #convert dim of word embedding from 128 to 2 
    reduced = tsne.fit_transform(vecs)
    print('reduced shape: ',reduced.shape)


    use_tag = set(['JJ','NNP','NN','NNS'])
    simbols = ["'",".",":",";",",","?","!","’"]
    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tag
                and all(c not in label for c in simbols)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label, size = 5))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    plt.savefig('word2vec.png', dpi=600)
