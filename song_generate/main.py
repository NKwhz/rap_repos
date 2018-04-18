# -*- coding: utf-8 -*-
import os
import data_randomly_by_artist as dd
import pickle
import numpy as np
import feature
from keras import backend as K
import string
from keras.models import load_model, Model
from keras.layers import Input, Dense, Lambda, multiply, Layer, concatenate
from keras import objectives
from pho_generate import get_weighted

delset = string.punctuation
weights_fname = 'model/VaeRL2_0.h5'


def create_pho():
    model = load_model('model/pho9.h5')
    return model


def slice(x, start, end):
    return x[:, start:end]


# def create_vaerl2(activation='tanh', use_bias=True, units=200, latent_dim=100, epsilon_std=1.0, alpha=1.0, beta=1.0):
#     doc_dim = 125
#     pho_dim = 125
#     input_shape_doc = doc_dim
#     input_shape_pho = pho_dim
#
#     # Input
#     x_doc = Input(shape=(doc_dim,))
#     x_doc_ = Dense(units=doc_dim, activation=activation, use_bias=use_bias)(x_doc)
#     x_pho = Input(shape=(pho_dim,))
#     x_pho_ = Dense(units=pho_dim, activation=activation, use_bias=use_bias)(x_pho)
#
#     # Attention Model
#     x_a = concatenate([x_doc_, x_pho_])
#     attention = Dense(units=doc_dim + pho_dim)(x_a)
#     x_r = multiply([x_a, attention])
#
#     # VAE model
#     x = Dense(units=units, activation=activation, use_bias=use_bias)(x_r)
#     encoder = Dense(units=units, activation=activation, use_bias=use_bias)(x)
#     z_mean = Dense(units=latent_dim, activation=activation, use_bias=use_bias, name='output')(encoder)
#     z_log_var = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(encoder)
#
#     def sampling(args):
#         z_mean, z_log_var = args
#         epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
#                                   stddev=epsilon_std)
#         return z_mean + K.exp(z_log_var / 2) * epsilon
#
#     z = Lambda(sampling)([z_mean, z_log_var])
#
#     decoder = Dense(units=units, activation=activation, use_bias=use_bias)(z)
#     _x = Dense(units=units, activation=activation, use_bias=use_bias)(decoder)
#
#     # dimension increase
#     _x_r = Dense(units=input_shape_doc + input_shape_pho, activation=activation, use_bias=use_bias)(_x)
#
#     # Attention Model
#     _attention = Dense(units=input_shape_doc + input_shape_pho)(_x_r)
#     _x_a = multiply([_x_r, _attention])
#
#     # Output
#     _x_doc_ = Lambda(slice, arguments={'start': 0, 'end': 125})(_x_a)
#     _x_doc = Dense(units=doc_dim, activation=activation, use_bias=use_bias)(_x_doc_)
#     _x_pho_ = Lambda(slice, arguments={'start': 125, 'end': 250})(_x_a)
#     _x_pho = Dense(units=pho_dim, activation=activation, use_bias=use_bias)(_x_pho_)
#
#     y = Input(shape=(1,), name='y_in')
#     sig = Dense(1, activation='sigmoid', use_bias=use_bias)(z_mean)
#
#     # Label loss
#     def loss(args):
#         x, y = args
#         loss = objectives.binary_crossentropy(x, y)
#         return loss
#
#     label_loss = Lambda(loss)([y, sig])
#
#     # Vae loss
#     x_doc_loss = Lambda(loss)([x_doc, _x_doc])
#     x_pho_loss = Lambda(loss)([x_pho, _x_pho])
#
#     def vae_loss(args):
#         x, y = args
#         kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         xent_loss = x + y
#         return xent_loss + kl_loss
#
#     vae_loss = Lambda(vae_loss)([x_doc_loss, x_pho_loss])
#
#     # Custom loss layer
#     class CustomVariationalLayer(Layer):
#         def __init__(self, **kwargs):
#             self.is_placeholder = True
#             super(CustomVariationalLayer, self).__init__(**kwargs)
#
#         def loss(self, x, y):
#             return K.mean(alpha * x + beta * y)
#
#         def call(self, inputs):
#             x = inputs[0]
#             y = inputs[1]
#             loss = self.loss(x, y)
#             self.add_loss(loss, inputs=inputs)
#
#             return x
#
#     L = CustomVariationalLayer()([label_loss, vae_loss])
#
#     vae = Model(outputs=L, inputs=[x_doc, x_pho, y])
#     vae.load_weights(weights_fname)
#     vae_sig = Model(inputs=[x_doc, x_pho], outputs=sig)
#     return vae_sig


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


def create_vaerl2(doc_dim = 125, pho_dim = 125, activation='tanh', use_bias=True, units=100, latent_dim=100, epsilon_std=1.0, alpha=1.0, beta=1.0):
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
    vaerl2.load_weights(weights_fname)

    vaerl2_sig = Model(inputs=[x_doc, x_0, x_1], outputs=sig)

    return vaerl2_sig


def get_latent(model, doc, pho0, pho1):
    return model.predict([doc, pho0, pho1])


def creart_random_candidate(can_num, current_line, songid, last_word):
    f = open('dataset/all_data', 'rb')
    db = pickle.load(f)
    f.close()
    endrhyme_score = []
    for line in db:
        if not line[11].strip() == '' and not (line[12] in songid or line[11].strip().split()[-1] in last_word):
            f1 = feature.rhyming(query=current_line[0][2], rhyme_query=current_line[0][3], candidate=line[2],
                                 rhyme_can=line[3]).get_endrhyme()
            endrhyme_score.append(f1)
        else:
            endrhyme_score.append(-2)
    vals = np.array(endrhyme_score)
    candidate = []
    sort_index = np.argsort(vals)
    get_num = 0
    strat_index = -1
    while get_num < can_num:
        index = sort_index[strat_index]
        strat_index = strat_index - 1
        candidate.append((db[index], index))
        get_num += 1
    return candidate


def get_query_candidate_pair(song, songid, last_word):
    # first find 300 candidate next line except self
    current_line = song[-1]
    candidate = creart_random_candidate(candidate_num, current_line, songid, last_word)
    if len(song) > 1:
        last_line = song[-2]
    else:
        last_line = song[-1]
    query = (
        (current_line[0][2], current_line[0][3], current_line[0][11]),
        (last_line[0][2], last_line[0][3], last_line[0][11]))
    candidate_read = []
    for can in candidate:
        candidate_read.append((can[0][2], can[0][3], can[0][11]))
    return query, candidate_read, candidate


def writ_ranksvm_input_file(writfilename, metra, cannum):
    file_open = open(writfilename, 'w+')
    for i in range(metra.shape[0]):
        if i % cannum == 0:
            file_open.write('# query ' + str(i / cannum + 1) + '\n')
            file_open.write('2 qid:' + str(i / cannum + 1) + ' ')
        else:
            file_open.write('1 qid:' + str(i / cannum + 1) + ' ')
        writ = ''
        for k in range(len(metra[i, :])):
            writ = writ + str(k + 1) + ':' + str(metra[i][k]) + ' '
        file_open.write(writ + '\n')
    file_open.close()


def Determine(query, candidate):
    if query[0][12] == candidate[0][12] or query[0][11] == candidate[0][11]:
        return False
    else:
        return True


create_random_song_num = 100
candidate_num = 300
f = open('dataset/start_songs', 'rb')
start = pickle.load(f)[:]
f.close()
create_songs = []
create_song_length = 16

create_models = ['VaeRL2']

for create_model in create_models:
    asvae = create_vaerl2()
    # pho_model = create_pho()
    for song_indx, start_line in enumerate(start):
        song = []
        songid = []
        last_word = []
        song.append(start_line)
        songid.append(start_line[0][12])
        last_word.append(start_line[0][11].strip().split()[-1])
        if not os.path.exists(create_model):
            os.mkdir(create_model)
        if not os.path.exists(create_model + '/song'):
            os.mkdir(create_model + '/song')
        if not os.path.exists(create_model + '/info'):
            os.mkdir(create_model + '/info')
        f = open(create_model + '/song/' + str(song_indx), 'w')
        f2 = open(create_model + '/info/' + str(song_indx), 'w')
        f2.write('1\t' + start_line[0][4].strip().replace(" ", "_") + '\t' + start_line[0][6].strip().replace(" ",
                                                                                                              "_") + '\n')
        f.write(start_line[0][2].strip() + '\n')
        f.flush()
        f2.flush()
        length = 1
        while length < create_song_length:
            # struct query and candidate
            query, candidate_read, candidate = get_query_candidate_pair(song, songid, last_word)
            # computer the feature_vector
            feature_vector = dd.get_distribution_query_candidates_doc2vec(query, candidate_read, frontk=0,
                                                                          docv_du=125)
            feature_vector1 = dd.get_distribution_query_candidates_doc2vec(query, candidate_read, frontk=0,
                                                                           docv_du=125,
                                                                           is_phoneme=True,
                                                                           model='doc_model_/doc_phoneme_model_size125windows150')
            feature_vector2 = dd.get_distribution_query_candidates_doc2vec(query, candidate_read, frontk=1,
                                                                           docv_du=125, is_phoneme=True,
                                                                           model='doc_model_/doc_phoneme_model_size125windows150')

            # feature_vector3 = pho_model.predict([feature_vector1, feature_vector2])
            rank = get_latent(asvae, feature_vector, feature_vector1, feature_vector2)

            post = np.argmax(rank)

            song.append(candidate[post])
            songid.append(candidate[post][0][12])
            last_word.append(candidate[post][0][11].strip().split()[-1])
            f2.write(candidate[post][0][4].strip().replace(" ", "_") + '\t' + candidate[post][0][6].strip().replace(" ",
                                                                                                                    "_") + '\n')
            f.write(candidate[post][0][2].strip() + '\n')
            length += 1
        create_songs.append(song)
        f.close()
        f2.close()
f = open('songs', 'wb')
pickle.dump(create_songs, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

# read_strat_lines("C:/Users/Administrator/Desktop/dope_creat100_ly")
