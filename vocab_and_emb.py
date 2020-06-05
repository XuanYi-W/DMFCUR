import os
import numpy as np
import pickle
import gensim


EMB_DIR = 'data/music/dmfcur'

print("creat video voc matrix")



EMB_file = os.path.join('data', 'google.bin')

W_USER_file = os.path.join(EMB_DIR, 'W_user.pk')
W_ITEM_file = os.path.join(EMB_DIR, 'W_item.pk')

PARA_file = os.path.join(EMB_DIR, "data.para")


def get_embedding(vocab, embedding_dim, emb_file, W_file=None):
    w = 0
    initW = np.random.uniform(-1.0, 1.0, (len(vocab), embedding_dim))
    print("Load word2vec file")
    model = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=True)
    for word in vocab:
        if word in model:
            idx = vocab[word]
            initW[idx] = np.array(model[word])
            w += 1
    print("number of pre-trained words", w)
    print(initW)

    return initW

if __name__ == '__main__':
    pkl_file = open(PARA_file, 'rb')
    para = pickle.load(pkl_file)
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    embedding_dim = 300

    init_Wu = get_embedding(vocabulary_user, embedding_dim, EMB_file)
    init_Wi = get_embedding(vocabulary_item, embedding_dim, EMB_file)

    pickle.dump(init_Wu, open(W_USER_file, 'wb'))
    pickle.dump(init_Wi, open(W_ITEM_file, 'wb'))
    print("vocabulary_user_size: " ,len(vocabulary_user))
    print("vocabulary_item_size: ", len(vocabulary_item))
    print("get pre-trained words")
