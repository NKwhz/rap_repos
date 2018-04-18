import numpy as np


if __name__ == '__main__':
    train = np.load('x_train/doc_20000.npy')
    test = np.load('x_test/doc_1000.npy')

    for v in test[0]:
        print(np.argwhere(train==v))