# -*- coding: utf-8 -*-
import GetResult as GR
import numpy as np
from keras.layers import Input, Dense, Lambda, multiply, Layer, add, concatenate, Flatten
from keras.models import Model
from keras import backend as K
from keras import objectives
from util import get_weighted, de_attention
from keras.callbacks import EarlyStopping

##############################################################################
# 分类
train_doc = 'class/train/train_doc.npy'
train_pho0 = 'class/train/train_pho0.npy'
train_pho1 = 'class/train/train_pho1.npy'
train_pho2 = 'class/train/train_pho2.npy'
train_pho3 = 'class/train/train_pho3.npy'
train_y = 'class/train/train_labels.npy'

test_doc = 'class/test/test_doc.npy'
test_pho0 = 'class/test/test_pho0.npy'
test_pho1 = 'class/test/test_pho1.npy'
test_pho2 = 'class/test/test_pho2.npy'
test_pho3 = 'class/test/test_pho3.npy'

# load training dataset
x_train_doc = np.load(train_doc)  # doc2vec feature file
x_train0 = np.load(train_pho0)
x_train1 = np.load(train_pho1)
x_train2 = np.load(train_pho2)
x_train3 = np.load(train_pho3)
y_train = np.load(train_y)

# load testing dataset
x_test_doc = np.load(test_doc)  # doc2vec feature file
x_test0 = np.load(test_pho0)  # feature file
x_test1 = np.load(test_pho1)
x_test2 = np.load(test_pho2)
x_test3 = np.load(test_pho3)
##############################################################################

pho_dim = 125
doc_dim = 125
con_dim = 250

activ = 'tanh'
optim = 'adagrad'
binary = False

cls_cnt = 9

def slice(x, start, end):
    return x[:, start:end]


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


def rhyme2vec(alpha=0, beta=0, activation=activ, use_bias=True, latent_dim=100, epochs=20, batch_size=10000):
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

    x_0 = Input(shape=(input_shape, ))
    x_1 = Input(shape=(input_shape, ))
    x_2 = Input(shape=(input_shape, ))
    # x_3 = Input(shape=(input_shape,))
    # x, att = get_weighted([x_0, x_1, x_2, x_3], pho_dim)
    x, att = get_weighted([x_0, x_1, x_2], pho_dim)

    x_encoder = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x)

    sig = Dense(cls_cnt, activation='sigmoid', use_bias=use_bias)(x_encoder)

    # rhyme = Model(outputs=sig, inputs=[x_0, x_1, x_2, x_3])
    rhyme = Model(outputs=sig, inputs=[x_0, x_1, x_2])
    print('=======Model Information=======' + '\n')
    # rhyme.summary()
    rhyme.compile(optimizer=optim, loss='binary_crossentropy')
    # rhyme.fit([x_train0, x_train1, x_train2, x_train3], y_train,
    #           shuffle=False,
    #           epochs=epochs,
    #           batch_size=batch_size
    #           )
    rhyme.fit([x_train0, x_train1, x_train2], y_train,
              # shuffle=False,
              epochs=epochs,
              batch_size=batch_size
              )

    # rep = Model(inputs=[x_0, x_1, x_2, x_3], outputs=x_encoder)
    # train_rep = rep.predict([x_train0, x_train1, x_train2, x_train3])
    # np.save('representations/rhyme2vec_train.npy', train_rep)
    # test_rep = rep.predict([x_test0, x_test1, x_test2, x_test3])
    # np.save('representations/rhyme2vec_test.npy', test_rep)

    rep = Model(inputs=[x_0, x_1, x_2], outputs=x_encoder)
    train_rep = rep.predict([x_train0, x_train1, x_train2])
    np.save('representations/rhyme2vec/train.npy', train_rep)
    test_rep = rep.predict([x_test0, x_test1, x_test2])
    np.save('representations/rhyme2vec/test.npy', test_rep)

    print('=======Rhyme2vec Result=======' + '\n')
    K.clear_session()



def doc2vec(alpha=0, beta=0, activation=activ, use_bias=False, latent_dim=100, epochs=10, batch_size=711240,
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
    sig = Dense(cls_cnt, activation='sigmoid', use_bias=use_bias)(x_encoder)

    doc = Model(outputs=sig, inputs=x)
    print('=======Model Information=======' + '\n')
    # doc.summary()
    doc.compile(optimizer=optim, loss='binary_crossentropy')

    doc.fit(x_train_doc, y_train,
            # shuffle=False,
            epochs=epochs,
            batch_size=batch_size
            )


    rep = Model(inputs=x, outputs=x_encoder)
    train_rep = rep.predict(x_train_doc)
    np.save('representations/doc2vec/train.npy', train_rep)
    test_rep = rep.predict(x_test_doc)
    np.save('representations/doc2vec/test.npy', test_rep)
    K.clear_session()

def VaeRL2(alpha=10, beta=100, activation=activ, use_bias=True, epochs=20, batch_size=16,
           units=200, latent_dim=100, epsilon_std=1.0):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # Input
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_2 = Input(shape=(pho_dim,))

    # x_3 = Input(shape=(pho_dim,))
    # x_pho, att = get_weighted([x_0, x_1, x_2, x_3], pho_dim)
    x_pho_d, _ = get_weighted([x_1, x_2], pho_dim)
    x_pho, att = get_weighted([x_0, x_pho_d], pho_dim)
    # x_pho, att = get_weighted([x_0, x_1, x_2], pho_dim)

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

    y = Input(shape=(cls_cnt,), name='y_in')
    sig = Dense(cls_cnt, activation='sigmoid', use_bias=use_bias)(z_mean)

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

    # vaerl2 = Model(outputs=L, inputs=[x_doc, x_0, x_1, x_2, x_3, y])
    vaerl2 = Model(outputs=L, inputs=[x_doc, x_0, x_1, x_2, y])
    print('=======Model Information=======' + '\n')
    # vaerl2.summary()

    vaerl2.compile(optimizer='adam', loss=None)
    # vaerl2.fit([x_train_doc, x_train0, x_train1, x_train2, x_train3, y_train],
    #            # shuffle=False,
    #            epochs=epochs,
    #            batch_size=batch_size,
    #            verbose=1
    #            )
    vaerl2.fit([x_train_doc, x_train0, x_train1, x_train2, y_train],
               # shuffle=False,
               epochs=epochs,
               batch_size=batch_size,
               verbose=1
               )

    # rep = Model(inputs=[x_doc, x_0, x_1, x_2, x_3], outputs=z_mean)
    # train_rep = rep.predict([x_train_doc, x_train0, x_train1, x_train2, x_train3])
    # np.save('representations/vaerl_train.npy', train_rep)
    # test_rep = rep.predict([x_test_doc, x_test0, x_test1, x_test2, x_test3])
    # np.save('representations/vaerl_test.npy', test_rep)

    rep = Model(inputs=[x_doc, x_0, x_1, x_2], outputs=z_mean)
    train_rep = rep.predict([x_train_doc, x_train0, x_train1, x_train2])
    np.save('representations/vaerl/train.npy', train_rep)
    test_rep = rep.predict([x_test_doc, x_test0, x_test1, x_test2])
    np.save('representations/vaerl/test.npy', test_rep)

    print('=======VAERL Result=======' + '\n')


def AERL_whole(paras=[6, 3, 1, 1, 1], activation=activ, use_bias=False, epochs=20, batch_size=1000,
               units=110, latent_dim=100, epsilon_std=1.0):
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
    x_2 = Input(shape=(pho_dim,))
    x_pho, pho_att = get_weighted([x_0, x_1, x_2], pho_dim)

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

    _x_0, _x_1, _x_2 = de_attention([_x_pho, pho_att], pho_dim, 3)

    y = Input(shape=(9,), name='y_in')
    sig = Dense(9, activation='softmax', use_bias=use_bias)(v)

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
    x_2_loss = Lambda(ae_loss)([x_2, _x_2])

    # Custom loss layer

    L = CustomVariationalLayer(paras=paras)([label_loss, x_doc_loss, x_0_loss, x_1_loss, x_2_loss])

    aerl = Model(outputs=L, inputs=[x_doc, x_0, x_1, x_2, y])
    print('=======Model Information=======' + '\n')
    # aerl.summary()

    aerl.compile(optimizer='rmsprop', loss=None)
    aerl.fit([x_train_doc, x_train0, x_train1, x_train2, y_train],
             shuffle=False,
             epochs=epochs,
             batch_size=batch_size
             )

    rep_gen = Model(inputs=[x_doc, x_0, x_1, x_2], outputs=v)

    reps = rep_gen.predict([x_train_doc, x_train0, x_train1, x_train2])
    np.save('train.npy', reps)
    reps = rep_gen.predict([x_test_doc, x_test0, x_test1, x_test2])
    np.save('test.npy', reps)

    print('=======ASVAE Result=======' + '\n')


if __name__ == '__main__':
    VaeRL2()

