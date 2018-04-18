from sklearn.svm import SVC, SVR
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
import keras.backend as K


class classifier:
    def __init__(self, input_dim, output_dim):
        self.data_dim = input_dim
        self.class_num = output_dim

    def build(self):
        return

    def fit(self, x, y):
        return

    def predict(self, x):
        return


class nn_classifier(classifier):
    def __init__(self, input_dim, output_dim=9, encoder_dim=100,
                 encoder_act='relu', optimizer='rmsprop', epoch=30, batch=100, threshold=0.2):
        super(nn_classifier, self).__init__(input_dim, output_dim)
        self.encoder_dim = encoder_dim
        self.act = encoder_act
        self.opt = optimizer
        self.epoch = epoch
        self.batch = batch
        self.thres = threshold
        self.build()


    def build(self):
        x = Input((self.data_dim,))
        encoder = Dense(self.encoder_dim, activation=self.act, use_bias=True)(x)
        decoder = Dense(self.class_num, activation='sigmoid', use_bias=True)(encoder)

        self.model = Model(inputs=x, outputs=decoder)

        self.model.compile(optimizer=self.opt, loss='binary_crossentropy')


    def fit(self, x, y):
        self.model.fit(x, y, batch_size=self.batch, epochs=self.epoch, verbose=0)

    def predict(self, x):
        preds = self.model.predict(x)
        preds[preds >= self.thres] = 1
        preds[preds < self.thres] = 0
        return preds


class svm(classifier):
    def __init__(self, input_dim, output_dim):
        super(svm, self).__init__(input_dim, output_dim)
        self.build()


    def build(self):
        self.clfs = []
        for i in range(self.class_num):
            self.clfs.append(SVC(class_weight='balanced', max_iter=1200))

    def fit(self, x, y):
        for i in range(self.class_num):
            self.clfs[i].fit(x, y[:, i])

    def predict(self, x):
        result = []
        for i in range(self.class_num):
            result.append(np.expand_dims(self.clfs[i].predict(x), axis=1))
        return np.concatenate(result, axis=1)


class svm_1(classifier):
    def __init__(self, input_dim, output_dim, threshold):
        super(svm_1, self).__init__(input_dim, output_dim)
        self.thres = threshold
        self.build()


    def build(self):
        self.clfs = []
        for i in range(self.class_num):
            self.clfs.append(SVR())

    def fit(self, x, y):
        for i in range(self.class_num):
            self.clfs[i].fit(x, y[:, i])

    def predict(self, x):
        result = []
        for i in range(self.class_num):
            result.append(np.expand_dims(self.clfs[i].predict(x), axis=1))
        preds = np.concatenate(result, axis=1)
        preds[preds >= self.thres] = 1
        preds[preds < self.thres] = 0
        return preds

