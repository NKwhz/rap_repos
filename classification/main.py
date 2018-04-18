import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
import keras.backend as K
from classifiers import nn_classifier, svm, svm_1

# load labels
train_labels = np.load(open('label/train_labels.npy', 'rb'))
test_labels = np.load(open('label/test_labels.npy', 'rb'))



def label(args):
    x, y = args
    k = K.cast(K.greater_equal(x, y), 'float32')
    return k


def classify(model, threshold, dimension):
    # clf = svm_1(dimension, 9, 0.15)
    clf = nn_classifier(dimension, threshold=threshold)

    # x = Input((dimension,))
    # train_con, test_con = model
    # encoder = Dense(100, activation='relu', use_bias=True)(x)
    # decoder = Dense(9, activation='sigmoid', use_bias=True)(encoder)
    #
    # model_con = Model(inputs=x, outputs=decoder)
    #
    # model_con.compile(optimizer='rmsprop', loss='binary_crossentropy')
    # model_con.fit(train_con, train_labels,
    #               epochs=30,
    #               batch_size=100,
    #               verbose=0,
    #               )
    # preds = model_con.predict(test_con)

    # preds[preds >= threshold] = 1
    # preds[preds < threshold] = 0
    train_con, test_con = model
    clf.fit(train_con, train_labels)
    preds = clf.predict(test_con)
    micro = metrics.f1_score(test_labels, preds, average='micro')
    macro = metrics.f1_score(test_labels, preds, average='macro')
    jaccard = metrics.jaccard_similarity_score(test_labels, preds)
    hamming = metrics.hamming_loss(test_labels, preds)
    results = [micro, macro, jaccard, hamming]
    return results


if __name__ == '__main__':

    train_sum = np.sum(train_labels, axis=0)
    test_sum = np.sum(test_labels, axis=0)
    print(train_sum + test_sum)

    f = open('result', 'a')
    # model_names = ['HAN', 'VaeRL2_whole', 'RhymeAPP', 'Doc2vec', 'Rhyme2vec_whole']
    # dims = [100, 100, 24, 125, 100]
    model_names = ['VaeRL2_whole']
    dims = [100]
    thres = [0.05 * i for i in range(3, 8)]
    # thres = [0.1]
    metr = ['micro', 'macro', 'jaccard', 'hamming']

    result = [[[0 for i in range(len(model_names))] for j in range(len(thres))] for k in range(len(metr))]

    micros = [[0 for _ in range(len(model_names))] for i in range(len(thres))]
    macros = [[0 for _ in range(len(model_names))] for i in range(len(thres))]

    for i_idx, i in enumerate(thres):
        print('i = {}'.format(i))
        for m_idx, m in enumerate(model_names):
            print('testing {}'.format(m))
            train_data = np.load(open('{}/train.npy'.format(m), 'rb'))
            train_data = np.nan_to_num(train_data)
            test_data = np.load(open('{}/test.npy'.format(m), 'rb'))
            test_data = np.nan_to_num(test_data)
            total = []
            for j in range(5):
                # print ('{} turn'.format(j + 1))
                res = classify((train_data, test_data), i, dims[m_idx])
                # print('{}>: {}'.format(j + 1, res))
                total.append(res)
            r = np.mean(total, axis=0)
            micros[i_idx][m_idx] = r[0]
            macros[i_idx][m_idx] = r[1]
            print(r)
            for idx, score in enumerate(r):
                result[idx][i_idx][m_idx] = score

    with open('micros.csv', 'w') as res_tab:
        title = ''
        for m in model_names:
            title += ',{}'.format(m)
        res_tab.write('{}\n'.format(title))
        for i_idx, i in enumerate(thres):
            line = 't={:.2f}'.format(i)
            for m_idx, m in enumerate(model_names):
                line += ',{}'.format(micros[i_idx][m_idx])
            res_tab.write('{}\n'.format(line))

    with open('macros.csv', 'w') as res_tab:
        title = ''
        for m in model_names:
            title += ',{}'.format(m)
        res_tab.write('{}\n'.format(title))
        for i_idx, i in enumerate(thres):
            line = 't={:.2f}'.format(i)
            for m_idx, m in enumerate(model_names):
                line += ',{}'.format(macros[i_idx][m_idx])
            res_tab.write('{}\n'.format(line))

    for met_i, met in enumerate(metr):
        f.write('\n============' + met + '===============\n')
        for idx, i in enumerate(thres):
            f.write('{}: {}\n'.format(i, str(result[met_i][idx])))
    f.close()
