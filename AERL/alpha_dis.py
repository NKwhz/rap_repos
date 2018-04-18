# -*- coding: utf-8 -*-
import GetResult as GR
import numpy as np
from keras.layers import Input, Dense, Lambda, multiply, Layer, add, concatenate, Flatten
from keras.models import Model
from keras import backend as K
from keras import objectives
from pho_generate import get_weighted, de_attention
from attention_layer import AttentionWithContext

# 小数据
#
# train_doc = 'x_train/doc_20000.npy'
# train_pho0 = 'x_train/pho_0_20000.npy'
# train_pho1 = 'x_train/pho_1_20000.npy'
# train_y = 'x_train/y_20000.npy'
#
# test_num = 1000
# test_doc = 'x_test/doc_1000.npy'
# test_pho0 = 'x_test/pho_0_1000.npy'
# test_pho1 = 'x_test/pho_1_1000.npy'


# 大数据

train_doc = 'x_train/doc.npy'
train_pho0 = 'x_train/pho_0.npy'
train_pho1 = 'x_train/pho_1.npy'
train_y = 'x_train/y.npy'
#
test_num = 10000
test_doc = 'x_test/doc.npy'
test_pho0 = 'x_test/pho_0.npy'
test_pho1 = 'x_test/pho_1.npy'

# load training dataset
x_train_doc = np.load(train_doc)  # doc2vec feature file
x_train0 = np.load(train_pho0)
x_train1 = np.load(train_pho1)
y_train = np.load(train_y)

# val_doc = x_train_doc[-20000:]
# y_val = y_train[-20000:]
#
# x_train_doc = x_train_doc[:-20000]
# y_train = y_train[:-20000]

# load testing dataset
x_test_doc = np.load(test_doc)  # doc2vec feature file
x_test0 = np.load(test_pho0)  # feature file
x_test1 = np.load(test_pho1)

pho_dim = 125
doc_dim = 125
con_dim = 250

activ = 'tanh'
optim = 'adadelta'
binary = False


def slice(x, start, end):
    return x[:, start:end]


# class CustomVariationalLayer(Layer):
#     def __init__(self, alpha, beta, **kwargs):
#         self.is_placeholder = True
#         self.alpha = alpha
#         self.beta = beta
#         super(CustomVariationalLayer, self).__init__(**kwargs)

#     def loss(self, x, y):
#         return K.mean(self.alpha * x + self.beta * y)

#     def call(self, inputs):
#         x = inputs[0]
#         y = inputs[1]
#         loss = self.loss(x, y)
#         self.add_loss(loss, inputs=inputs)
#         # We won't actually use the output.
#         return x


class CustomVariationalLayer(Layer):
    def __init__(self, paras, **kwargs):
        self.is_placeholder = True
        self.hyperparas = paras
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def loss(self, losses):
        # print('self hypers', self.hyperparas)
        # l = losses[0]
        # print('loss list length is', len(losses))
        # for i, p in enumerate(self.hyperparas):
        #     if i > 0:
        #         print(i)
        #         l += p * losses[i]
        #         print(i)
        l = sum([p * losses[i] for i, p in enumerate(self.hyperparas)])
        return K.mean(l)

    def call(self, inputs):
        l = self.loss(inputs)
        self.add_loss(l, inputs=inputs)
        # We won't actually use the output.
        return inputs[0]


def VaeRL2(alpha=1.0, beta=1.0, activation=activ, use_bias=True, epochs=5, batch_size=1000,
           units=200, latent_dim=100, epsilon_std=1.0):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # Input
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho, att = get_weighted([x_0, x_1], pho_dim)

    # Attention Model
    # x_a = concatenate([x_doc, x_pho])
    # attention = Dense(units=doc_dim + pho_dim, activation='sigmoid')(x_a)
    # x_r = multiply([x_a, attention])

    # x_r = Dense(units=doc_dim + pho_dim, activation='relu')(x_a)
    x_r, _ = get_weighted([x_doc, x_pho], pho_dim)

    # VAE model
    z_mean = Dense(units=latent_dim, activation=activation, use_bias=use_bias, name='output')(x_r)
    z_log_var = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x_r)

    def sampling_z(args):
        _mean, _log_var = args
        # print("=========================================\n\n\n")
        # print("mean shape: {}".format(K.shape(_mean)))
        # print("\n\n\n=========================================")
        epsilon = K.random_normal(shape=(K.shape(_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return _mean + K.exp(_log_var / 2) * epsilon

    z = Lambda(sampling_z)([z_mean, z_log_var])

    de_mean = Dense(units=doc_dim, activation=activation, use_bias=use_bias)(z)
    de_log_var = Dense(units=doc_dim, activation=activation, use_bias=use_bias)(z)

    def sampling_d(args):
        _mean, _log_var = args
        # print("=========================================\n\n\n")
        # print("mean shape: {}".format(K.shape(_mean)))
        # print("\n\n\n=========================================")
        epsilon = K.random_normal(shape=(K.shape(_mean)[0], doc_dim + pho_dim), mean=0.,
                                  stddev=epsilon_std)
        return _mean + K.exp(_log_var / 2) * epsilon

    # decoder = Lambda(sampling_d)([de_mean, de_log_var])
    # _attention = Dense(units=doc_dim + pho_dim, activation='sigmoid')(decoder)
    # _x_a = multiply([decoder, _attention])
    #
    # # Output
    # _x_doc = Lambda(slice, arguments={'start': 0, 'end': 125})(_x_a)
    # _x_pho = Lambda(slice, arguments={'start': 125, 'end': 250})(_x_a)

    y = Input(shape=(1,), name='y_in')
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(z_mean)

    # Label loss
    def loss(args):
        x, y = args
        loss = objectives.binary_crossentropy(x, y)
        return loss

    label_loss = Lambda(loss)([y, sig])

    # Vae loss
    # x_doc_loss = Lambda(loss)([x_doc, _x_doc])
    # x_pho_loss = Lambda(loss)([x_pho, _x_pho])

    def vae_loss(args):
        zm, zl, dm, dl, xa = args
        kl_loss = - 0.5 * K.mean(1 + zl - K.square(zm) - K.exp(zl), axis=-1)
        pxz = - K.mean(-0.5 * (np.log(2 * np.pi) + dl) - 0.5 * K.square(xa - dm) / K.exp(dl))
        # xent_loss = x + y
        return kl_loss + pxz

    v_loss = Lambda(vae_loss)([z_mean, z_log_var, de_mean, de_log_var, x_r])

    # Custom loss layer

    L = CustomVariationalLayer([alpha, beta])([v_loss, label_loss])

    vaerl2 = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    # vaerl2.summary()

    vaerl2.compile(optimizer='adadelta', loss=None)
    vaerl2.fit([x_train_doc, x_train0, x_train1, y_train],
               shuffle=False,
               epochs=epochs,
               batch_size=batch_size,
               verbose=1
               )

    vaerl2_sig = Model(inputs=[x_doc, x_0, x_1, y], outputs=[sig, label_loss, v_loss])

    y_test = np.array(([1] + ([0] * 299)) * test_num)
    rank = vaerl2_sig.predict([x_test_doc, x_test0, x_test1, y_test])

    scores = rank[0].reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(scores)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)
    l_loss = np.mean(scores[1])
    v_loss = np.mean(scores[2])

    print('=======VAERL Result=======' + '\n')
    # K.clear_session()
    return result, vaerl2


def func(times, model, name, epoch, log_f, turns, dim, batch_size):
    r = []
    log_f.write('\n======{}======\n'.format(name))
    for j in range(turns):
        result, m_f = model(alpha=1, beta=1 * times, epochs=epoch, latent_dim=dim, batch_size=batch_size)
        line = '{}:> {}\n'.format(j, ' '.join([str(f1) for f1 in result]))
        # log_f.write(line)
        r.append(result)
        print(result)
        log_f.flush()
        model_filename = 'model/{}_{}.h5'.format(name, j)
        # m_f.save_weights(model_filename)
        # K.clear_session()
    # t = [0] * len(r[0])
    t = np.mean(r, axis=0)
    log_f.write('Batch_size: {}\n'.format(batch_size))
    log_f.write('Average: {}\n'.format(' '.join([str(fl) for fl in t])))
    log_f.write('======{} end.======\n'.format(name))
    log_f.flush()
    return t


if __name__ == '__main__':
    import time

    log = open('log', 'a')
    log.write("\n{}\n"
              "test size:{}\n".format(time.asctime(time.localtime(time.time())), test_num))

    # dims

    names = ['doc2vec', 'rhyme2vec', 'con', 'attention', 'conAE', 'conVAE', 'VaeRL2', 'AttAE', 'c-line', 'skip-line',
             'DA']
    turn = 5
    batch_sizes = 100
    res = []
    # dims = list(range(50, 251, 50))
    dims = [100]
    alp = [0.001, 0.003, 0.005, 0.007, 0.009]
    # alp = [1]
    times = []
    # iset = list(range(7))
    iset = [6]
    '''
    0: doc2vec
    1: rhyme2vec
    2: con
    3: att
    4: conAE
    5: conVAE
    6: VaeRL2
    7: AttAE
    8: C-Line
    9: Skip-Line
    10: DA
    '''
    print('================Dimension Discussion=================')
    for i in iset:
        print("\n\n*****************{}*****************\n\n".format(names[i]))
        for a in alp:
            time_start = time.time()
            res_t = func(a, VaeRL2, 'alpha discuss', 20, log, turn, 100, batch_sizes)
            # cross_val(models[i], names[i], log, turn[i], d, ['relu', 'tanh', 'sigmoid', 'softmax'], ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam'])
            time_used = time.time() - time_start
            log.write('alpha={} takes {} seconds in average.\n'.format(a, time_used))
            log.write('result: {}\n'.format(res_t))
            log.flush()
    log.close()
