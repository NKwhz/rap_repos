import GetResult as GR
import numpy as np
from keras.layers import Input, Dense, Lambda, multiply, Layer, add, concatenate
from keras.models import Model
from keras import backend as K
from keras import objectives
from pho_generate import get_weighted, de_attention, tensor_slice, tensor_split, inv_mul
import time

# # 小数据
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
# train_doc = 'x_train/doc.npy'
# train_pho0 = 'x_train/pho_0.npy'
# train_pho1 = 'x_train/pho_1.npy'
# train_y = 'x_train/y.npy'
#
# test_num = 10000
# test_doc = 'x_test/doc.npy'
# test_pho0 = 'x_test/pho_0.npy'
# test_pho1 = 'x_test/pho_1.npy'
#
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

##############################################################################
# 分类
train_doc = 'class/train/train_doc.npy'
train_pho0 = 'class/train/train_pho0.npy'
train_pho1 = 'class/train/train_pho1.npy'
train_pho2 = 'class/train/train_pho2.npy'
train_y = 'class/train/train_labels.npy'

test_doc = 'class/test/test_doc.npy'
test_pho0 = 'class/test/test_pho0.npy'
test_pho1 = 'class/test/test_pho1.npy'
test_pho2 = 'class/test/test_pho2.npy'

# load training dataset
x_train_doc = np.load(train_doc)  # doc2vec feature file
x_train0 = np.load(train_pho0)
x_train1 = np.load(train_pho1)
x_train2 = np.load(train_pho2)
y_train = np.load(train_y)

# load testing dataset
x_test_doc = np.load(test_doc)  # doc2vec feature file
x_test0 = np.load(test_pho0)  # feature file
x_test1 = np.load(test_pho1)
x_test2 = np.load(test_pho2)
##############################################################################


pho_dim = 125
doc_dim = 125
con_dim = 250

activ = 'tanh'
optim = 'sgd'
binary = False


def tensor_slice(x, start, end):
    return x[:, start:end]


def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = (ylen + batch_size - 1) // batch_size
    # idx = shuffle(list(range(loopcount + 1)))
    while (True):
        for i in range(loopcount):
            yield [x[0][i * batch_size:min((i + 1) * batch_size, ylen)], x[1][
                                                                        i * batch_size:min((i + 1) * batch_size, ylen)], \
                  x[2][i * batch_size:min((i + 1) * batch_size, ylen)], y[
                                                                        i * batch_size:min((i + 1) * batch_size, ylen)]], None


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


def AE(alpha=3.0, beta=1.0, gamma=1.0, delta=1.0, activation=activ, use_bias=True, epochs=20, batch_size=500,
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
    x_pho = add([x_0, x_1])

    doc_encoder = Dense(units=units, activation=activation, use_bias=use_bias)(x_doc)
    pho_encoder = Dense(units=units, activation=activation, use_bias=use_bias)(x_pho)

    # Attention Model
    u = add([doc_encoder, pho_encoder])

    v = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(u)

    _u = Dense(units=units, activation=activation, use_bias=use_bias)(v)
    # attention = Dense(units=doc_dim + pho_dim, activation='sigmoid')(x_a)
    # x_r = multiply([x_a, attention])
    _doc_decoder, _pho_decoder = inv_mul(_u, [[1, 1]], 2)

    _x_doc = Dense(units=doc_dim, activation=activation, use_bias=use_bias)(_doc_decoder)
    _x_pho = Dense(units=pho_dim, activation=activation, use_bias=use_bias)(_pho_decoder)

    _x_0, _x_1 = inv_mul(_x_pho, [[1, 1]], 2)

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

    ae = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    ae.summary()

    ae.compile(optimizer='adadelta', loss=None)
    ae.fit([x_train_doc, x_train0, x_train1, y_train],
           shuffle=False,
           epochs=epochs,
           batch_size=batch_size
           )

    ae_sig = Model(inputs=[x_doc, x_0, x_1, y], outputs=sig)

    y_test = np.array(([1] + ([0] * 299)) * test_num)
    rank = ae_sig.predict([x_test_doc, x_test0, x_test1, y_test])

    scores = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(scores)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======ASVAE Result=======' + '\n')
    # K.clear_session()
    return result, ae


def AERL(alpha=2, beta=1, gamma=1, delta=0, activation=activ, use_bias=False, epochs=20, batch_size=50000,
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
    # x_doc_loss = Lambda(ae_loss)([doc_encoder, _doc_decoder])
    # x_pho_loss = Lambda(ae_loss)([pho_encoder, _pho_decoder])

    # Custom loss layer

    L = CustomVariationalLayer(paras=[alpha, beta, gamma, delta])([label_loss, x_doc_loss, x_0_loss, x_1_loss])

    aerl = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    aerl.summary()

    aerl.compile(optimizer=optim, loss=None)

    aerl.fit_generator(generate_batch_data_random([x_train_doc, x_train0, x_train1], y_train, batch_size),
                       steps_per_epoch=(y_train.shape[0] + batch_size - 1) //
                                       batch_size,
                       epochs=epochs,
                       verbose=1)

    # aerl.fit([x_train_doc, x_train0, x_train1, y_train],
    #          shuffle=False,
    #          epochs=epochs,
    #          batch_size=batch_size
    #          )

    aerl_sig = Model(inputs=[x_doc, x_0, x_1, y], outputs=sig)

    y_test = np.array(([1] + ([0] * 299)) * test_num)
    rank = aerl_sig.predict([x_test_doc, x_test0, x_test1, y_test])

    scores = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(scores)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======ASVAE Result=======' + '\n')
    # K.clear_session()
    return result, aerl


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

    aerl.compile(optimizer=optim, loss=None)
    aerl.fit([x_train_doc, x_train0, x_train1, x_train2, y_train],
             shuffle=False,
             epochs=epochs,
             batch_size=batch_size
             )

    rep_gen = Model(inputs=[x_doc, x_0, x_1, x_2], outputs=v)

    reps = rep_gen.predict([x_train_doc, x_train0, x_train1, x_train2])
    np.save('E:\\Learning\\Papers\\calssification\\AERL\\train.npy', reps)
    reps = rep_gen.predict([x_test_doc, x_test0, x_test1, x_test2])
    np.save('E:\\Learning\\Papers\\calssification\\AERL\\test.npy', reps)

    print('=======ASVAE Result=======' + '\n')
    # K.clear_session()
    # return result, aerl


# res, model = AE()
# print(res)
# model_filename = 'model/aerl'
# model.save_weights(model_filename)

################################################################################
################################################################################
################################################################################

# res = []
# for i in range(4):
#     pra = [0.] * 4
#     pra[i] = 1.0
#     print('\n\n======================{}====================\n\n'.format(i))
#     r, _ = AERL(alpha=pra[0], beta=pra[1], gamma=pra[2], delta=pra[3])
#     res.append(r)

# for r in res:
#     print(r)

log = open('log', 'a')
log.write("\n{}\ntest size:{}\n".format(time.asctime(time.localtime(time.time())), test_num))
log.write('\n======AttAE======\n')
res = []
time = 5
for i in range(time):
    print('\n\n======================{}====================\n\n'.format(i))
    r, _ = AERL(alpha=3, beta=1, gamma=1, delta=1)
    res.append(r)
    print(r)
    log.write('{}:> {}\n'.format(i, ' '.join([str(f1) for f1 in r])))
    log.flush()

ave = [0, 0, 0, 0, 0, 0]
for r in res:
    print(r)
for i in range(6):
    summ = 0
    for j in range(time):
        summ += res[j][i]
    ave[i] = summ / time
print('============average===========')
print(ave)
log.write('Average: {}\n'.format(' '.join([str(fl) for fl in ave])))
log.write('======AttAE end.======\n')
log.flush()

###################################################################################
###################################################################################
###################################################################################

# AERL_whole()
