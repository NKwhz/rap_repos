# -*- coding: utf-8 -*-
import GetResult as GR
import numpy as np
from keras.layers import Input, Dense, Lambda, multiply, Layer, add, concatenate, Flatten
from keras.models import Model
from keras import backend as K
from keras import objectives
from pho_generate import get_weighted, de_attention
from keras.callbacks import CSVLogger
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


def rhyme2vec(alpha=0, beta=0, activation=activ, use_bias=False, latent_dim=100, epochs=5, batch_size=200000):
    # recall@k
    rec_k_list = [1, 5, 30, 150]
    # the input dimension
    input_shape = pho_dim

    # x_train0 = np.load(train_pho0)
    # x_train1 = np.load(train_pho1)
    # y_train = np.load(train_y)  # label file
    #
    # # load testing dataset
    # x_test0 = np.load(test_pho0)  # feature file
    # x_test1 = np.load(test_pho1)
    print('====load dataset done====' + '\n')

    x_0 = Input(shape=(input_shape,))
    x_1 = Input(shape=(input_shape,))
    x, att = get_weighted([x_0, x_1], pho_dim)
    x = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x)

    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x)

    rhyme = Model(outputs=sig, inputs=[x_0, x_1])
    print('=======Model Information=======' + '\n')
    # rhyme.summary()
    rhyme.compile(optimizer=optim, loss='binary_crossentropy')
    rhyme.fit([x_train0, x_train1], y_train,
              shuffle=False,
              epochs=epochs,
              batch_size=batch_size,
              verbose=0
              )

    rank = rhyme.predict([x_test0, x_test1])

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======Rhyme2vec Result=======' + '\n')
    K.clear_session()
    return result, rhyme


def cl(alpha=0, beta=0, activation=activ, use_bias=False, latent_dim=0, epochs=5, batch_size=500000,
            test_doc_fname=test_doc):
    # recall@k
    rec_k_list = [1, 5, 30, 150]
    # the input dimension
    input_shape = doc_dim

    # x_train = np.load(train_doc)  # feature file
    # y_train = np.load(train_y)  # label file
    #
    # x_test = np.load(test_doc_fname)  # feature file

    # Model
    x = Input(shape=(input_shape,))
    x_encoder = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x)
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x_encoder)

    doc = Model(outputs=sig, inputs=x)
    print('=======Model Information=======' + '\n')
    # doc.summary()
    doc.compile(optimizer=optim, loss='binary_crossentropy')

    doc.fit(x_train0, y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
            )

    rank = doc.predict(x_test0)

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)
    print('=======Doc2vec Result=======' + '\n')
    K.clear_session()
    return result, doc


def sl(alpha=0, beta=0, activation=activ, use_bias=False, latent_dim=0, epochs=5, batch_size=500000,
            test_doc_fname=test_doc):
    # recall@k
    rec_k_list = [1, 5, 30, 150]
    # the input dimension
    input_shape = doc_dim

    # x_train = np.load(train_doc)  # feature file
    # y_train = np.load(train_y)  # label file
    #
    # x_test = np.load(test_doc_fname)  # feature file

    # Model
    x = Input(shape=(input_shape,))
    x_encoder = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x)
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x_encoder)

    doc = Model(outputs=sig, inputs=x)
    print('=======Model Information=======' + '\n')
    # doc.summary()
    doc.compile(optimizer=optim, loss='binary_crossentropy')

    doc.fit(x_train1, y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
            )

    rank = doc.predict(x_test1)

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)
    print('=======Doc2vec Result=======' + '\n')
    K.clear_session()
    return result, doc


def adding(alpha=0, beta=0, activation=activ, use_bias=False, latent_dim=0, epochs=5, batch_size=500000,
            test_doc_fname=test_doc):
    # recall@k
    rec_k_list = [1, 5, 30, 150]
    # the input dimension
    input_shape = doc_dim

    # x_train = np.load(train_doc)  # feature file
    # y_train = np.load(train_y)  # label file
    #
    # x_test = np.load(test_doc_fname)  # feature file

    # Model
    x0 = Input(shape=(input_shape,))
    x1 = Input(shape=(input_shape,))
    x = add([x0, x1])
    x_encoder = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x)
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x_encoder)

    doc = Model(outputs=sig, inputs=[x0, x1])
    print('=======Model Information=======' + '\n')
    # doc.summary()
    doc.compile(optimizer=optim, loss='binary_crossentropy')

    doc.fit([x_train0, x_train1], y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
            )

    rank = doc.predict([x_test0, x_test1])

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)
    print('=======Doc2vec Result=======' + '\n')
    K.clear_session()
    return result, doc


def doc2vec(alpha=0, beta=0, activation=activ, use_bias=False, latent_dim=0, epochs=5, batch_size=500000,
            test_doc_fname=test_doc):
    # recall@k
    rec_k_list = [1, 5, 30, 150]
    # the input dimension
    input_shape = doc_dim

    # x_train = np.load(train_doc)  # feature file
    # y_train = np.load(train_y)  # label file
    #
    # x_test = np.load(test_doc_fname)  # feature file

    # Model
    x = Input(shape=(input_shape,))
    x_encoder = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x)
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x_encoder)

    doc = Model(outputs=sig, inputs=x)
    print('=======Model Information=======' + '\n')
    # doc.summary()
    doc.compile(optimizer=optim, loss='binary_crossentropy')

    doc.fit(x_train_doc, y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
            )

    rank = doc.predict(x_test_doc)

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)
    print('=======Doc2vec Result=======' + '\n')
    K.clear_session()
    return result, doc


def con(alpha=0, beta=0, activation=activ, use_bias=False, latent_dim=0, epochs=5, batch_size=500000):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # load training dataset
    # x_train0 = np.load(train_pho0)
    # x_train1 = np.load(train_pho1)
    # x_train_doc = np.load(train_doc)
    # y_train = np.load(train_y)  # label file
    #
    # # load testing dataset
    # x_test_doc = np.load(test_doc)
    # x_test0 = np.load(test_pho0)  # feature file
    # x_test1 = np.load(test_pho1)

    # Model
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho, att = get_weighted([x_0, x_1], pho_dim)
    # x_doc_encoder = Dense(units=doc_dim, activation=activation, use_bias=use_bias)(x_doc)
    # x_pho_encoder = Dense(units=pho_dim, activation=activation, use_bias=use_bias)(x_pho)
    encoder = concatenate([x_doc, x_pho])
    encoder = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(encoder)
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(encoder)

    con = Model(outputs=sig, inputs=[x_doc, x_0, x_1])
    print('=======Model Information=======' + '\n')
    # con.summary()
    con.compile(optimizer=optim, loss='binary_crossentropy')

    con.fit([x_train_doc, x_train0, x_train1], y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
            )

    rank = con.predict([x_test_doc, x_test0, x_test1])
    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======Concatenate Result=======' + '\n')
    K.clear_session()
    return result, con

def att(alpha=0, beta=0, activation=activ, use_bias=False, latent_dim=0, epochs=5, batch_size=711240):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # load training dataset
    # x_train0 = np.load(train_pho0)
    # x_train1 = np.load(train_pho1)
    # x_train_doc = np.load(train_doc)
    # y_train = np.load(train_y)  # label file
    #
    # # load testing dataset
    # x_test_doc = np.load(test_doc)
    # x_test0 = np.load(test_pho0)  # feature file
    # x_test1 = np.load(test_pho1)

    # Model
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho, _ = get_weighted([x_0, x_1], pho_dim)
    # x_doc_encoder = Dense(units=doc_dim, activation=activation, use_bias=use_bias)(x_doc)
    # x_pho_encoder = Dense(units=pho_dim, activation=activation, use_bias=use_bias)(x_pho)
    encoder, _ = get_weighted([x_doc, x_pho], pho_dim)
    encoder = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(encoder)
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(encoder)

    con = Model(outputs=sig, inputs=[x_doc, x_0, x_1])
    print('=======Model Information=======' + '\n')
    con.summary()
    con.compile(optimizer=optim, loss='binary_crossentropy')

    con.fit([x_train_doc, x_train0, x_train1], y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
            )

    rank = con.predict([x_test_doc, x_test0, x_test1])
    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======Attention Result=======' + '\n')
    K.clear_session()
    return result, con

def conAE(alpha, beta, activation=activ, use_bias=False, latent_dim=100, epochs=5, batch_size=100000):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # load training dataset
    # x_train0 = np.load(train_pho0)
    # x_train1 = np.load(train_pho1)
    # x_train_doc = np.load(train_doc)
    # y_train = np.load(train_y)  # label file
    #
    # # load testing dataset
    # x_test0 = np.load(test_pho0)  # feature file
    # x_test1 = np.load(test_pho1)
    # x_test_doc = np.load(test_doc)

    # AE Model
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho, att = get_weighted([x_0, x_1], pho_dim)
    encoder = concatenate([x_doc, x_pho])

    answer = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(encoder)

    decoder = Dense(units=doc_dim + pho_dim, activation=activation, use_bias=use_bias)(answer)

    _x_doc = Lambda(slice, arguments={'start': 0, 'end': 125})(decoder)
    _x_pho = Lambda(slice, arguments={'start': 125, 'end': 250})(decoder)

    y = Input(shape=(1,))
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(answer)

    def bi_loss(args):
        x, y = args
        loss = objectives.mean_squared_error(x, y)
        return loss

    def ae_loss(args):
        xd, _xd, xp, _xp = args
        return objectives.binary_crossentropy(xd, _xd) \
               + objectives.binary_crossentropy(xp, _xp)

    # Label loss
    label_loss = Lambda(bi_loss)([y, sig])

    # AE loss
    sae_loss = Lambda(ae_loss)([x_doc, _x_doc, x_pho, _x_pho])

    # Custom loss layer
    L = CustomVariationalLayer([alpha, beta])([label_loss, sae_loss])

    AE = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    # AE.summary()
    AE.compile(optimizer=optim, loss=None)

    AE.fit([x_train_doc, x_train0, x_train1, y_train],
           shuffle=False,
           epochs=epochs,
           batch_size=batch_size,
           verbose=0
           )

    AE_sig = Model(inputs=[x_doc, x_0, x_1], outputs=sig)
    rank = AE_sig.predict([x_test_doc, x_test0, x_test1])

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======conAE Result=======' + '\n')
    K.clear_session()
    return result, AE


def conVAE(alpha, beta, activation=activ, use_bias=True, epochs=5, batch_size=100000, units=200, latent_dim=100,
           epsilon_std=1.0):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # load training dataset
    # x_train0 = np.load(train_pho0)
    # x_train1 = np.load(train_pho1)
    # x_train_doc = np.load(train_doc)
    # y_train = np.load(train_y)  # label file
    #
    # # load testing dataset
    # x_test0 = np.load(test_pho0)  # feature file
    # x_test1 = np.load(test_pho1)
    # x_test_doc = np.load(test_doc)

    # VAE Model
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho, att = get_weighted([x_0, x_1], pho_dim)
    concat = concatenate([x_doc, x_pho])

    z_mean = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(concat)
    z_log_var = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(concat)

    def sampling(args):
        _mean, _log_var = args
        epsilon = K.random_normal(shape=(K.shape(_mean)[0], K.shape(_mean)[1]), mean=0.,
                                  stddev=epsilon_std)
        return _mean + K.exp(_log_var / 2) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    de_mean = Dense(units=doc_dim + pho_dim, activation=activation, use_bias=use_bias)(z)
    de_log_var = Dense(units=doc_dim + pho_dim, activation=activation, use_bias=use_bias)(z)

    decoder = Lambda(sampling)([de_mean, de_log_var])

    # decoder = Dense(units=doc_dim + pho_dim, activation=activation, use_bias=use_bias)(z)

    _x_doc = Lambda(slice, arguments={'start': 0, 'end': 125})(decoder)
    _x_pho = Lambda(slice, arguments={'start': 125, 'end': 250})(decoder)

    y = Input(shape=(1,))
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(z_mean)

    # Label loss
    def bi_loss(args):
        x, y = args
        loss = objectives.mean_squared_error(x, y)
        return loss

    label_loss = Lambda(bi_loss)([y, sig])

    # VAE loss
    def vae_loss(args):
        zm, zl, dm, dl, c = args
        # xent_loss = objectives.binary_crossentropy(xd, _xd) \
        #             + objectives.binary_crossentropy(xp, _xp)
        pxz = - K.mean(-0.5 * (np.log(2 * np.pi) + dl) - 0.5 * K.square(c - dm) / K.exp(dl))
        kl_loss = - 0.5 * K.mean(1 + zl - K.square(zm) - K.exp(zl), axis=-1)
        return kl_loss + pxz

    vae_loss = Lambda(vae_loss)([z_mean, z_log_var, de_mean, de_log_var, concat])

    # Custom loss layer
    L = CustomVariationalLayer([alpha, beta])([label_loss, vae_loss])

    con_vae = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    # con_vae.summary()

    con_vae.compile(optimizer=optim, loss=None)
    con_vae.fit([x_train_doc, x_train0, x_train1, y_train],
                shuffle=False,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
                )

    con_vae_sig = Model(inputs=[x_doc, x_0, x_1], outputs=sig)
    rank = con_vae_sig.predict([x_test_doc, x_test0, x_test1])

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======conVAE Result=======' + '\n')
    K.clear_session()
    return result, con_vae


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

    vae_loss = Lambda(vae_loss)([z_mean, z_log_var, de_mean, de_log_var, x_r])

    # Custom loss layer

    L = CustomVariationalLayer([alpha, beta])([label_loss, vae_loss])

    vaerl2 = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    # vaerl2.summary()

    vaerl2.compile(optimizer='adadelta', loss=None, metrics=[])
    logger = CSVLogger('train.log')
    vaerl2.fit([x_train_doc, x_train0, x_train1, y_train],
               shuffle=False,
               epochs=epochs,
               batch_size=batch_size,
               verbose=1,
               callbacks=[logger]
               )

    vaerl2_sig = Model(inputs=[x_doc, x_0, x_1, y], outputs=[sig, label_loss, vae_loss])

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


def AERL(alpha=4, beta=2, gamma=1, delta=1, activation=activ, use_bias=False, epochs=20, batch_size=711240,
           units=128, latent_dim=100, epsilon_std=1.0):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # # load training dataset
    # x_train_doc = np.load(train_doc)  # doc2vec feature file
    # x_train0 = np.load(train_pho0)
    # x_train1 = np.load(train_pho1)
    # y_train = np.load(train_y)
    #
    # # load testing dataset
    # x_test_doc = np.load(test_doc)  # doc2vec feature file
    # x_test0 = np.load(test_pho0)  # feature file
    # x_test1 = np.load(test_pho1)

    # Input
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho, pho_att = get_weighted([x_0, x_1], pho_dim)

    doc_encoder = Dense(units=units, activation=activation, use_bias=use_bias)(x_doc)
    pho_encoder = Dense(units=units, activation=activation, use_bias=use_bias)(x_pho)

    # Attention Model
    u, u_att = get_weighted([doc_encoder, pho_encoder], units)

    v = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(u)

    _u = Dense(units=units, activation=activation, use_bias=use_bias)(v)
    # attention = Dense(units=doc_dim + pho_dim, activation='sigmoid')(x_a)
    # x_r = multiply([x_a, attention])
    _doc_decoder, _pho_decoder = de_attention([_u, u_att], units, 2)

    _x_doc = Dense(units=doc_dim, activation=activation, use_bias=use_bias)(_doc_decoder)
    _x_pho = Dense(units=pho_dim, activation=activation, use_bias=use_bias)(_pho_decoder)

    _x_0, _x_1 = de_attention([_x_pho, pho_att], pho_dim, 2)

    y = Input(shape=(1,), name='y_in')
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(v)

    # Label loss
    def loss(args):
        x, y = args
        loss = objectives.binary_crossentropy(x, y)
        return loss

    def ae_loss(args):
        x, y = args
        loss = objectives.mean_squared_error(x, y)
        return loss

    label_loss = Lambda(loss)([y, sig])

    # Vae loss
    x_doc_loss = Lambda(ae_loss)([x_doc, _x_doc])
    x_0_loss = Lambda(ae_loss)([x_0, _x_0])
    x_1_loss = Lambda(ae_loss)([x_1, _x_1])

    # Custom loss layer

    L = CustomVariationalLayer(paras=[alpha, beta, gamma, delta])([label_loss, x_doc_loss, x_0_loss, x_1_loss])

    aerl = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    aerl.summary()

    aerl.compile(optimizer=optim, loss=None)
    aerl.fit([x_train_doc, x_train0, x_train1, y_train],
               shuffle=False,
               epochs=epochs,
               batch_size=batch_size
               )

    aerl_sig = Model(inputs=[x_doc, x_0, x_1, y], outputs=sig)

    y_test = np.array(([1] + ([0] * 299)) * test_num)
    rank = aerl_sig.predict([x_test_doc, x_test0, x_test1, y_test])

    scores = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(scores)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======AERL Result=======' + '\n')
    # K.clear_session()
    return result, aerl


def func(model, name, epoch, log_f, turns, dim, batch_size):
    r = []
    log_f.write('\n======{}======\n'.format(name))
    for j in range(turns):
        result, m_f = model(alpha=10, beta=100, epochs=epoch, latent_dim=dim, batch_size=batch_size)
        line = '{}:> {}\n'.format(j, ' '.join([str(f1) for f1 in result]))
        log_f.write(line)
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


def cross_val(model, name, log_f, turns, dim, activs, optims):
    global activ
    global optim
    
    log_f.write('\n======{}======\n'.format(name))
    for opt in optims:
        for a in activs:
            log_f.write('\n=========={}   {}==========\n'.format(a, opt))
            r = []
            for j in range(turns):
                activ = a
                optim = opt
                result, m_f = model(alpha=1, beta=1, epochs=10, latent_dim=dim)
                line = '{}:> {}\n'.format(j, ' '.join([str(f1) for f1 in result]))
                log_f.write(line)
                r.append(result)
                print(a, opt, result)
                log_f.flush()
                model_filename = 'model/{}_{}.h5'.format(name, j)
                # m_f.save_weights(model_filename)
                # K.clear_session()
            # t = [0] * len(r[0])
            t = np.mean(r, axis=0)
            log_f.write('Average: {}\n'.format(' '.join([str(fl) for fl in t])))
    log_f.write('======{} end.======\n'.format(name))
    log_f.flush()


if __name__ == '__main__':
    import time

    log = open('log', 'a')
    log.write("\n{}\n"
              "test size:{}\n".format(time.asctime(time.localtime(time.time())), test_num))

    # dims
    models = [doc2vec, rhyme2vec, con, att, conAE, conVAE, VaeRL2, AERL, cl, sl, adding]
    names = ['doc2vec', 'rhyme2vec', 'con', 'attention', 'conAE', 'conVAE', 'VaeRL2', 'AttAE', 'c-line', 'skip-line', 'DA']
    turn = [1] * 11
    batch_sizes = [711240, 200000, 711240, 711240, 100000, 100000, 100, 711240, 711240, 711240, 711240]
    res = []
    # dims = list(range(50, 251, 50))
    dims = [100]
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
        for d in dims:
            time_start = time.time()
            res_t = func(models[i], names[i], 1, log, turn[i], d, batch_sizes[i])
            # cross_val(models[i], names[i], log, turn[i], d, ['relu', 'tanh', 'sigmoid', 'softmax'], ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam'])
            time_used = time.time() - time_start
            log.write('{} dims take {} seconds in average.\n'.format(d, time_used))
            log.write('result: {}\n'.format(res_t))
            log.flush()
    log.close()
